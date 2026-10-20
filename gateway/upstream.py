"""
Upstream client — manages the HTTP connection to llama-server.

Responsibilities:
  - mDNS hostname resolution with TTL cache
  - httpx.AsyncClient lifecycle (startup / shutdown)
  - Token counting via /tokenize with LRU cache (max 1000 entries)
  - Auto-discovery of n_ctx from /props
  - Health check
"""
from __future__ import annotations

import collections
import logging
import re
import socket
import time
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

from gateway.config import GatewayConfig

_log = logging.getLogger("gateway.upstream")

# ---------------------------------------------------------------------------
# LRU dict (bounded OrderedDict)
# ---------------------------------------------------------------------------

class _LRUDict(collections.OrderedDict):
    """OrderedDict that evicts the oldest entry when full."""

    def __init__(self, maxsize: int = 1000):
        super().__init__()
        self._maxsize = maxsize

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self._maxsize:
            self.popitem(last=False)


# ---------------------------------------------------------------------------
# Reading GET /props
#
# Two callers need the same two facts out of one payload, and llama.cpp builds
# do not agree on where either lives. Keeping the reads here — rather than
# inline at each call site — is the whole fix for O-2026-03: `discover_n_ctx`
# had a chain of fallbacks, `health_check` twenty lines below had none, and so
# /health reported `null` and `"unknown"` for the life of the file.
# ---------------------------------------------------------------------------

def n_ctx_from_props(data: dict) -> tuple[int | None, str]:
    """Return (context window, where it was found) from a /props payload.

    The suffix is for the log line: knowing WHICH key a build answered on is
    what makes the next unfamiliar llama.cpp build cheap to diagnose.
    """
    n_ctx = data.get("n_ctx")
    if isinstance(n_ctx, int) and n_ctx:
        return n_ctx, ""
    for key in ("default_generation_settings", "props"):
        nested = data.get(key) or {}
        if isinstance(nested, dict):
            n_ctx = nested.get("n_ctx")
            if isinstance(n_ctx, int) and n_ctx:
                return n_ctx, f".{key}"
    return None, ""


def model_name_from_props(data: dict) -> str:
    """Return the served model's name, or "unknown".

    `model_alias` is what this server actually answers with;
    `default_generation_settings.model` is what the old code looked for and no
    build here returns. `model_path` is the last resort — a full path is not a
    name, so it is reduced to its basename, with the separator handled by hand
    because the path comes from whatever OS the upstream runs (a macOS path
    read on Windows, in this project's case).
    """
    alias = data.get("model_alias")
    if isinstance(alias, str) and alias.strip():
        return alias.strip()

    nested = data.get("default_generation_settings") or {}
    if isinstance(nested, dict):
        name = nested.get("model")
        if isinstance(name, str) and name.strip():
            return name.strip()

    path = data.get("model_path")
    if isinstance(path, str) and path.strip():
        return path.strip().replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]

    return "unknown"


# ---------------------------------------------------------------------------
# Upstream client
# ---------------------------------------------------------------------------

#: Fields the engine reports on /health that are worth surfacing, by name.
#: A BOUNDED list rather than the whole payload: an upstream is somebody else's
#: process -- A-2026-01 is this project's record of what happens when that is
#: forgotten -- and merging whatever it returns into this gateway's own /health
#: gives a stranger a say in what the dashboard shows.
#:
#: Empty: FastStrata's server reports its status, model and window, which
#: /health already carries in its own words, and no counters. A counter the
#: engine gains is added here by name.
ENGINE_HEALTH_FIELDS: tuple[str, ...] = ()


def engine_health_fields(health: object) -> dict:
    """The named engine fields present in an upstream /health payload."""
    if not isinstance(health, dict):
        return {}
    return {k: health[k] for k in ENGINE_HEALTH_FIELDS if health.get(k) is not None}


class UpstreamClient:
    """Async HTTP client for communication with the engine (llama.cpp's API)."""

    def __init__(self, config: GatewayConfig):
        self._config = config
        self._client: httpx.AsyncClient | None = None

        # mDNS resolver state
        self._resolved_url: str | None = None
        self._resolved_at: float = 0.0
        self._resolve_ttl: float = 300.0  # 5 minutes

        # Token counting LRU cache (fix unbounded dict memory leak from v1)
        self._token_cache: _LRUDict = _LRUDict(maxsize=1000)

    # -- Lifecycle -------------------------------------------------------------

    async def startup(self) -> None:
        """Create the httpx client.  Called once from the FastAPI lifespan."""
        limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
        transport = httpx.AsyncHTTPTransport(
            limits=limits,
            socket_options=[
                (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
                (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
            ],
        )
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(
                connect=30.0,
                read=float(self._config.stream_max_duration),
                write=60.0,
                pool=60.0,
            ),
        )
        _log.info("httpx client created (max_connections=500)")

    async def shutdown(self) -> None:
        """Close the httpx client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            _log.info("httpx client closed")

    @property
    def client(self) -> httpx.AsyncClient:
        """Raw httpx client (used by streaming code that needs low-level access)."""
        if not self._client:
            raise RuntimeError("UpstreamClient not started — call startup() first")
        return self._client

    # -- mDNS resolution -------------------------------------------------------

    def resolve(self, force: bool = False) -> str:
        """Resolve the upstream URL, caching .local mDNS lookups."""
        url = self._config.upstream
        now = time.time()

        if (not force
                and self._resolved_url
                and (now - self._resolved_at) < self._resolve_ttl):
            return self._resolved_url

        parsed = urlparse(url)
        hostname = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        if not hostname:
            return url

        # Already an IP — skip resolution
        if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname):
            self._resolved_url = url
            self._resolved_at = now
            return url

        try:
            results = socket.getaddrinfo(hostname, port, socket.AF_INET, socket.SOCK_STREAM)
            if results:
                ip = results[0][4][0]
                resolved = urlunparse(parsed._replace(netloc=f"{ip}:{port}"))
                _log.info("mDNS resolve success: %s -> %s", hostname, ip)
                self._resolved_url = resolved
                self._resolved_at = now
                return resolved
        except socket.gaierror as e:
            _log.warning("mDNS resolve failed for %s: %s. Using cached/original.", hostname, e)
            if self._resolved_url:
                return self._resolved_url

        return url

    # -- HTTP helpers ----------------------------------------------------------

    async def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Make a request to upstream with automatic mDNS retry on connection/timeout error."""
        subpath = path.lstrip("/")
        resolved = self.resolve()
        url = f"{resolved}/{subpath}"
        try:
            return await self.client.request(method, url, **kwargs)
        except (httpx.NetworkError, httpx.TimeoutException) as exc:
            _log.warning("Connection failure to %s: %s. Re-resolving and retrying.", url, exc)
            resolved = self.resolve(force=True)
            url = f"{resolved}/{subpath}"
            return await self.client.request(method, url, **kwargs)

    # -- Token counting --------------------------------------------------------

    async def count_tokens(self, text: str, base_url: str | None = None) -> int:
        """Count tokens via /tokenize. Falls back to a len/3 heuristic.

        **`base_url` exists because this asked the wrong server.** It used to
        go to `self.resolve()` — the CONFIGURED upstream — for every request,
        including ones the registry routes somewhere else. On a hot-swap rig
        that is wrong in two different ways, and both are silent:

        - While the text engine is UP and a vision model is being sized, the
          answer comes back from the wrong tokenizer. A number, confidently,
          for a different vocabulary.
        - While it is DOWN, because the vision model holds the card, the call
          fails — twice, since `request()` re-resolves and retries — and each
          attempt carries a 5 s timeout. Measured 2026-09-13: about ten
          seconds per vision turn, spent to reach a port nothing was on, ending
          in the character heuristic anyway.

        This is A-2026-01 in a third costume: **ask the party that knows.**
        Given a `base_url` the call goes there and is tried ONCE, because a
        retry against an engine that is not up yet is the pile-up this
        repository has paid for before. Anything else falls back to the
        heuristic without a round trip.

        The cache key includes the address, so two engines cannot share an
        answer.
        """
        cache_key = hash((base_url or "", text))
        cached = self._token_cache.get(cache_key)
        if cached is not None:
            return cached

        if not self._client:
            return len(text) // 3
        try:
            if base_url:
                r = await self.client.request(
                    "POST", f"{base_url.rstrip('/')}/tokenize",
                    json={"content": text}, timeout=httpx.Timeout(5.0))
            else:
                r = await self.request("POST", "/tokenize",
                                       json={"content": text},
                                       timeout=httpx.Timeout(5.0))
            if r.status_code == 200:
                count = len(r.json().get("tokens", []))
                self._token_cache[cache_key] = count
                return count
        except Exception as exc:
            # Debug, not warning: on a hot-swap rig the engine being asked is
            # routinely not up yet, and a WARNING per turn for the ordinary
            # case is how a log stops being read.
            _log.debug("Token counting via /tokenize failed: %s", exc)

        fallback = len(text) // 3
        self._token_cache[cache_key] = fallback
        return fallback

    # -- Discovery -------------------------------------------------------------

    async def discover_n_ctx(self) -> int | None:
        """Query GET /props to discover the model's context window size."""
        try:
            r = await self.request("GET", "/props", timeout=httpx.Timeout(5.0))
            if r.status_code == 200:
                n_ctx, where = n_ctx_from_props(r.json())
                if n_ctx:
                    _log.info("Discovered n_ctx=%d from upstream /props%s", n_ctx, where)
                    return n_ctx
        except Exception as exc:
            _log.warning("Failed to discover n_ctx from /props: %s", exc)
        return None

    async def health_check(self) -> dict:
        """Ping upstream and return status info.

        Reads the SAME payload through the SAME helpers as `discover_n_ctx`.
        Until 2026-09-03 it had its own inline reads of `n_ctx` and
        `default_generation_settings.model` — two keys llama-server does not
        return — so it answered `null` / `"unknown"` on every call it ever
        served, and the dashboard's context-size chip never once appeared
        (O-2026-03).
        """
        try:
            r = await self.request("GET", "/props", timeout=httpx.Timeout(5.0))
            if r.status_code == 200:
                data = r.json()
                n_ctx, _where = n_ctx_from_props(data)
                status = "ok"
                vram_free = None
                engine: dict = {}
                try:
                    hr = await self.request("GET", "/health", timeout=httpx.Timeout(2.0))
                    if hr.status_code == 200:
                        hdata = hr.json()
                        if hdata.get("status") == "sleeping":
                            status = "sleeping"
                            vram_free = hdata.get("vram_free_gb")
                        engine = engine_health_fields(hdata)
                except Exception:
                    pass

                result = {
                    "status": status,
                    "n_ctx": n_ctx,
                    "model": model_name_from_props(data),
                }
                if vram_free is not None:
                    result["vram_free_gb"] = vram_free
                result.update(engine)
                return result
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}
        return {"status": "error", "detail": "unexpected response"}
