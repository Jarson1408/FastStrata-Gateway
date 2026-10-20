"""
FastAPI application — routes, lifespan, request pipeline.

Replaces the monolithic proxy() handler from v1 with a cleaner structure
while preserving 100% functional parity.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
import dataclasses
from dataclasses import dataclass
from typing import IO, Any, AsyncIterator, Callable, ClassVar

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import (JSONResponse, PlainTextResponse, Response,
                               StreamingResponse)

from gateway.config import (GatewayConfig,
                            is_polystrata_upstream, POLYSTRATA_PORT,
                            upstream_can_constrain,
                            upstream_takes_json_schema)
from gateway.upstream import UpstreamClient
from gateway.qwen_xml_fsm import (
    BaseToolParser, QwenXmlFSM, EventType, FsmEvent, collect_events, create_parser,
    detect_model_family
)
from gateway.openai_sse import OpenAISSEEmitter
from gateway import (adaptive, client_params, destructive, engines, grammar,
                     metrics, normalizer, prompts, runaway, spill)
from gateway.model_registry import ENGINE_REMOTE, ModelSpec
from gateway.scheduler import (BACKGROUND, INTERACTIVE, PrioritySemaphore,
                               request_priority)
from gateway.tool_schema import (
    CRITICAL_FIELDS, ToolSchemaRegistry, coerce_value, heal_question_arguments,
    placeholder_for, should_buffer_tool,
)
from gateway.audit import AuditLogger
from gateway.shell_safety import build_embed_search_command, sanitize_search_query
from gateway.repair import (
    build_content_continuation_prompt as _build_content_continuation_prompt,
    build_continuation_prompt as _build_continuation_prompt,
    build_loop_recovery_prompt as _build_loop_recovery_prompt,
    check_text_backtick_state as _check_text_backtick_state,
    extract_conclusion_from_reasoning as _extract_conclusion_from_reasoning,
    get_initial_backtick_state,
    repair_truncated_structure as _repair_truncated_structure,
    validate_tool_args_json as _validate_tool_args_json,
)
from gateway.lodestar import client_contract
from gateway.lodestar.tool_names import (
    EDIT_TOOLS, READ_CANDIDATES, READ_TOOLS, SEARCH_TOOLS, SHELL_CANDIDATES,
    WRITE_CANDIDATES, first_available,
)
from gateway import (loop_breaker, path_repair, reasoning_code,
                     reasoning_loop)
from gateway.skills import resolve_skill_path
from gateway.lodestar import policy
from gateway.lodestar.policy import decide_tool_call
from gateway.lodestar.synthesis import (
    SUBAGENT_TASK_MARKER as _SUBAGENT_TASK_MARKER,
    embed_search_query as _embed_search_query,
    synthesize_active_context_write as _synthesize_active_context_write,
    synthesize_phase_d_nudge as _synthesize_phase_d_nudge,
    synthesize_stubborn_call as _synthesize_stubborn_call,
    path_param_for as _path_param_for,
)
from gateway.lodestar import LodestarState
from gateway.lodestar.budget import SynthesisBudget
from gateway.lodestar.state import cycle_turn_count, synthesis_count

_log = logging.getLogger("gateway.app")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(config: GatewayConfig) -> None:
    """Configure logging with rotating file handler."""
    from logging.handlers import RotatingFileHandler

    root = logging.getLogger("gateway")
    root.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    config.log_dir.mkdir(parents=True, exist_ok=True)

    # Rotating file handler (50MB, 3 backups)
    log_file = config.log_dir / "gateway.log"
    fh = RotatingFileHandler(
        str(log_file),
        maxBytes=50 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    # The version is on EVERY line, not only on the startup banner.
    #
    # `gateway.log` rotates at 50 MB across three files and survives restarts,
    # so a line read during an investigation is routinely from a build that is
    # no longer installed -- and the banner that would have said which one is
    # thousands of lines above it, or in a rotated-out file, or was never
    # written because the process was already running when the log was opened.
    # Every incident in AGENTS.md was reconstructed from these lines; several
    # of them turned on which revision produced them.
    #
    # Twelve characters a line against that: `prompts_version` in the audit
    # already ties a change in model behaviour to the prompt revision that
    # caused it, and this is the same idea for the code.
    log_format = f"%(asctime)s {GATEWAY_VERSION} %(levelname)s [%(name)s] %(message)s"
    fh.setFormatter(logging.Formatter(log_format))
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(log_format))
    root.addHandler(ch)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

class _HealthAccessFilter(logging.Filter):
    """Drop polling lines from uvicorn's access log.

    The desktop app polls /health every 15 s and a Prometheus scrape is
    typically every 15 s too; unfiltered, those lines flood the live log view
    (5000-line buffer) and can scroll real errors out of sight within hours of
    idle time.
    """
    _POLLED = ("/health", "/metrics")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(path in message for path in self._POLLED)


async def _report_embedding_service(config: GatewayConfig) -> None:
    """Say at startup whether the warm embedding service is up.

    LODESTAR shells out to embed.py twice per task — the Phase A search and the
    Phase D lesson. A cold call spends 7.2 s importing sentence-transformers
    before it does anything, so a missing service is ~16 s per task of pure
    process startup. Nothing breaks without it (embed.py loads the model
    in-process), which is exactly why it is easy to forget it was never
    started: the only symptom is that everything feels slow.

    Never fatal, never blocking for long, and silent when LODESTAR is off —
    the semantic memory is not used then.
    """
    if not getattr(config, "lodestar_enabled", True):
        return
    url = os.environ.get("EMBED_SERVICE_URL", "http://127.0.0.1:18001").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{url}/health")
        payload = resp.json() if resp.status_code == 200 else {}
    except Exception:
        payload = {}
    if payload.get("status") == "ok":
        _log.info("Embedding service up at %s (model=%s) — semantic memory calls "
                  "cost ~0.3 s", url, payload.get("model", "?"))
    else:
        _log.warning(
            "Embedding service NOT reachable at %s. The semantic memory still "
            "works (embed.py loads the model itself) but every call costs ~8 s "
            "instead of ~0.3 s, twice per task. Start it with: "
            "python .context/semantic-memory/embed.py serve  — or register the "
            "'faststrata-embeddings' task via install/windows/setup_autostart.ps1. "
            "See INSTALL.md.", url,
        )


# ---------------------------------------------------------------------------
# PolyStrata supervisor
# ---------------------------------------------------------------------------

_POLYSTRATA_HEALTH_URL = f"http://127.0.0.1:{POLYSTRATA_PORT}/health"
_POLYSTRATA_PROPS_URL = f"http://127.0.0.1:{POLYSTRATA_PORT}/props"
_POLYSTRATA_READY_ATTEMPTS = 60

_POLYSTRATA_PROC: "subprocess.Popen[bytes] | None" = None
_POLYSTRATA_LAST_SPAWN: float = 0.0
_POLYSTRATA_CURRENT_CONFIG: str | None = None
_POLYSTRATA_LOCK: "asyncio.Lock | None" = None
_POLYSTRATA_LOCK_LOOP: "asyncio.AbstractEventLoop | None" = None


def _polystrata_lock() -> asyncio.Lock:
    """The supervisor's lock, bound to the loop that is asking for it."""
    global _POLYSTRATA_LOCK, _POLYSTRATA_LOCK_LOOP
    loop = asyncio.get_running_loop()
    if _POLYSTRATA_LOCK is None or _POLYSTRATA_LOCK_LOOP is not loop:
        _POLYSTRATA_LOCK = asyncio.Lock()
        _POLYSTRATA_LOCK_LOOP = loop
    return _POLYSTRATA_LOCK


def _polystrata_reset_state() -> None:
    """Forget the tracked process, the cooldown and the lock. Tests only."""
    global _POLYSTRATA_PROC, _POLYSTRATA_LAST_SPAWN, _POLYSTRATA_CURRENT_CONFIG, _POLYSTRATA_LOCK, _POLYSTRATA_LOCK_LOOP
    _POLYSTRATA_PROC = None
    _POLYSTRATA_LAST_SPAWN = 0.0
    _POLYSTRATA_CURRENT_CONFIG = None
    _POLYSTRATA_LOCK = None
    _POLYSTRATA_LOCK_LOOP = None


async def _polystrata_probe_health() -> str | None:
    """The status PolyStrata reports, or None when it is not answering."""
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            r = await client.get(_POLYSTRATA_HEALTH_URL)
            if r.status_code == 200:
                return "ok"
            r2 = await client.get(_POLYSTRATA_PROPS_URL)
            if r2.status_code == 200:
                return "ok"
            return None
    except Exception:
        return None


async def _polystrata_wait_until_ready(
    proc: "subprocess.Popen[bytes]",
    attempts: int = _POLYSTRATA_READY_ATTEMPTS,
) -> "subprocess.Popen[bytes]":
    """Poll until PolyStrata answers on port 8080, it exits, or we give up."""
    for _ in range(attempts):
        await asyncio.sleep(1.0)
        if await _polystrata_probe_health() == "ok":
            _log.info("[+] PolyStrata server is online and ready (PID=%d) on port %d!",
                      proc.pid, POLYSTRATA_PORT)
            return proc
        if proc.poll() is not None:
            _log.warning(
                "PolyStrata server (PID=%d) exited during startup with code %s.",
                proc.pid, proc.returncode)
            return proc
    _log.warning("Timed out waiting for PolyStrata server to start on port %d.", POLYSTRATA_PORT)
    return proc


def _polystrata_spawn(config: GatewayConfig) -> "subprocess.Popen[bytes] | None":
    """Start one PolyStrata `serve/server.py` process."""
    global _POLYSTRATA_CURRENT_CONFIG
    polystrata_dir = config.polystrata_home()
    server_script = os.path.join(polystrata_dir, "serve", "server.py")
    python_bin = (os.path.join(polystrata_dir, ".venv", "Scripts", "python.exe") if os.name == "nt"
                  else os.path.join(polystrata_dir, ".venv", "bin", "python"))
    if not os.path.isfile(python_bin):
        python_bin = sys.executable

    cfg_name = getattr(config, "polystrata_config", "strata-qwen36-ud-q4_k_m.json")
    cfg_file = os.path.join(polystrata_dir, cfg_name) if not os.path.isabs(cfg_name) else cfg_name

    if not os.path.isfile(server_script):
        _log.warning("PolyStrata server.py not found at %s. Skipping auto-start.", server_script)
        return None

    _log.info("[*] Auto-starting PolyStrata server on http://127.0.0.1:%d (config=%s)...",
              POLYSTRATA_PORT, cfg_name)
    startupinfo = None
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0  # SW_HIDE

    log_path = os.path.join(polystrata_dir, "polystrata_server.log")
    log_file: "IO[str] | int"
    try:
        log_file = open(log_path, "a", encoding="utf-8")
    except Exception:
        log_file = subprocess.DEVNULL

    cmd = [
        python_bin,
        server_script,
        "--engine", "strata",
        "--config", cfg_file if os.path.isfile(cfg_file) else cfg_name,
        "--port", str(POLYSTRATA_PORT),
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=polystrata_dir,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        startupinfo=startupinfo,
    )
    _POLYSTRATA_CURRENT_CONFIG = os.path.basename(cfg_name)
    return proc


async def _ensure_polystrata_server(
    config: GatewayConfig,
    upstream: str | None = None,
    cfg_name: str | None = None,
) -> "subprocess.Popen[bytes] | None":
    """Ensure PolyStrata is running on port 8080 with requested config, swapping if needed."""
    global _POLYSTRATA_PROC, _POLYSTRATA_LAST_SPAWN, _POLYSTRATA_CURRENT_CONFIG

    if not getattr(config, "polystrata_autostart", True):
        return None

    target_upstream = upstream or getattr(config, "upstream", "")
    if not is_polystrata_upstream(target_upstream):
        return None

    target_cfg = os.path.basename(str(cfg_name or getattr(config, "polystrata_config", "strata-qwen36-ud-q4_k_m.json")))
    cooldown = float(getattr(config, "polystrata_restart_cooldown", 30.0))

    async with _polystrata_lock():
        status = await _polystrata_probe_health()
        if status == "ok":
            if _POLYSTRATA_CURRENT_CONFIG is not None and _POLYSTRATA_CURRENT_CONFIG != target_cfg:
                _log.info(
                    "PolyStrata is online with '%s', but '%s' was requested -- swapping model...",
                    _POLYSTRATA_CURRENT_CONFIG, target_cfg)
                _stop_polystrata_server()
                await asyncio.sleep(1.5)
            else:
                _POLYSTRATA_CURRENT_CONFIG = target_cfg
                _log.info("PolyStrata server is already online on port %d (config=%s).",
                          POLYSTRATA_PORT, target_cfg)
                return None

        if _POLYSTRATA_PROC is not None and _POLYSTRATA_PROC.poll() is None:
            if _POLYSTRATA_CURRENT_CONFIG == target_cfg:
                _log.info("PolyStrata server (PID=%d) is already starting -- not spawning another.",
                          _POLYSTRATA_PROC.pid)
                return _POLYSTRATA_PROC
            else:
                _log.info("PolyStrata server (PID=%d) is starting with different config -- stopping for swap.",
                          _POLYSTRATA_PROC.pid)
                _stop_polystrata_server()
                await asyncio.sleep(1.0)

        elapsed = time.monotonic() - _POLYSTRATA_LAST_SPAWN
        if _POLYSTRATA_LAST_SPAWN and cooldown > 0 and elapsed < cooldown:
            _log.warning(
                "PolyStrata server is down, but a start was attempted %.0f s ago "
                "(cooldown %.0f s) -- not starting another.",
                elapsed, cooldown)
            return None

        if getattr(config, "polystrata_config", None) != target_cfg:
            import dataclasses
            config = dataclasses.replace(config, polystrata_config=target_cfg)
        proc = _polystrata_spawn(config)
        if proc is None:
            return None
        _POLYSTRATA_PROC = proc
        _POLYSTRATA_CURRENT_CONFIG = target_cfg
        _POLYSTRATA_LAST_SPAWN = time.monotonic()

    return await _polystrata_wait_until_ready(proc)


#: Image names a lingering FastStrata process can have: the server is Python,
#: the engine strata-poly. Anything else listening on the port is somebody
#: else's program, and ending it would be this gateway's doing.
_POLYSTRATA_IMAGES = ("python.exe", "pythonw.exe", "strata-poly.exe", "strata-poly-kvq8.exe")


def _image_name(pid: str) -> str:
    """The executable name Windows reports for `pid`, lower-case; "" if unknown."""
    res = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True, check=False)
    first = res.stdout.strip().splitlines()[0] if res.stdout.strip() else ""
    return first.split(",")[0].strip('"').lower() if first.startswith('"') else ""


def _kill_polystrata_port(port: int = POLYSTRATA_PORT) -> None:
    """End a FastStrata process still listening on its port (Windows only).

    Only a process whose image is FastStrata's (`_POLYSTRATA_IMAGES`): the port
    is FastStrata's by default, not by right, and a PC that runs something else
    on 8080 must not lose it when a model is swapped.
    """
    if os.name != "nt":
        return
    try:
        res = subprocess.run(["netstat", "-ano", "-p", "tcp"],
                             capture_output=True, text=True, check=False)
        for line in res.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 5 and parts[3] == "LISTENING" and parts[1].endswith(f":{port}"):
                pid = parts[4]
                if pid and pid != "0" and pid != str(os.getpid()) and pid != "4":
                    image = _image_name(pid)
                    if image not in _POLYSTRATA_IMAGES:
                        _log.warning("Port %d is held by PID %s (%s), not by FastStrata -- "
                                     "leaving it alone.", port, pid, image or "unknown")
                        continue
                    _log.info("Terminating lingering process on port %d (PID=%s)...", port, pid)
                    subprocess.run(["taskkill", "/F", "/T", "/PID", pid],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        _log.debug("Failed to clean up port %d: %s", port, e)


def _stop_polystrata_server(proc: "subprocess.Popen[bytes] | None" = None) -> None:
    """Kill the PolyStrata server and free its VRAM/RAM.

    Defaults to the process the supervisor is tracking. On Windows, uses
    `taskkill /F /T /PID` to terminate the entire process tree (Python supervisor
    and child strata-poly.exe). Also checks and terminates any lingering process
    on port 8080 to ensure host RAM and VRAM are completely freed.
    """
    global _POLYSTRATA_PROC, _POLYSTRATA_LAST_SPAWN, _POLYSTRATA_CURRENT_CONFIG
    if proc is None:
        proc = _POLYSTRATA_PROC
    if proc is not None and proc.poll() is None:
        _log.info("Stopping auto-started PolyStrata server (PID=%d)...", proc.pid)
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                proc.terminate()
        except Exception as e:
            _log.warning("Failed to terminate PolyStrata server: %s", e)
    if proc is _POLYSTRATA_PROC:
        _POLYSTRATA_PROC = None
    _POLYSTRATA_LAST_SPAWN = 0.0
    _POLYSTRATA_CURRENT_CONFIG = None

    _kill_polystrata_port()


async def _hand_over_card(spec: "ModelSpec | None",
                         config: GatewayConfig) -> str | None:
    """Make the FastStrata model that serves `spec` be the one holding the card.

    Returns the base URL to send this request to, or None when the spec names
    nothing this gateway manages (a remote upstream, or no registry at all).
    Raises `engines.EngineError` with a readable sentence when the hand-over
    cannot happen -- the caller turns that into CONTENT, never an error object:
    a failed swap ends the turn, and an error object ends the session.

    The files are checked BEFORE the running model is stopped, so a request
    for a model that is not on disk leaves the working one loaded.
    """
    if spec is None or not spec.is_local:
        return None
    engines.check_installed(spec, config.polystrata_home())
    await _ensure_polystrata_server(config, upstream=spec.upstream, cfg_name=spec.target)
    return spec.upstream


#: How often a hand-over that is taking minutes reassures the client. A
#: constant rather than a literal so a test can make a slow swap fast, which
#: is the only honest way to exercise the keep-alive path -- a mocked
#: `asyncio.wait` can be made to report "done" for a task that is not.
_SWAP_KEEPALIVE_INTERVAL = 10.0


async def _hand_over_card_streaming(
    spec: "ModelSpec | None", config: GatewayConfig, out: dict[str, Any],
) -> AsyncIterator[bytes]:
    """`_hand_over_card`, yielding SSE keep-alives while it takes minutes.

    Loading a 13.6 GB GGUF off a cold page cache is the longest thing this
    gateway ever waits for, and a silent wait is exactly what a client's read
    timeout is for.

    The URL lands in `out["url"]` because an async generator cannot return a
    value, and the two alternatives are both worse: a sentinel item makes a
    byte stream carry something that is not bytes, and an attribute on the
    function object is process-global state that two concurrent requests
    would write over each other. `out` belongs to one caller.
    """
    task = asyncio.create_task(_hand_over_card(spec, config))
    try:
        while True:
            done, _pending = await asyncio.wait(
                {task}, timeout=_SWAP_KEEPALIVE_INTERVAL)
            if done:
                break
            yield b": keep-alive\n\n"
    except (GeneratorExit, asyncio.CancelledError):
        # The client went away mid-swap. The engine start itself is not
        # cancelled — a half-loaded model is worse than a loaded one nobody
        # asked for, and the next request will find it ready.
        raise
    out["url"] = task.result()


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = GatewayConfig.from_env()
    _setup_logging(config)

    # Installed at lifespan (after uvicorn has configured its loggers) so the
    # filter is not wiped by uvicorn's dictConfig.
    logging.getLogger("uvicorn.access").addFilter(_HealthAccessFilter())

    await _ensure_polystrata_server(config)

    upstream = UpstreamClient(config)
    await upstream.startup()

    # Auto-discover n_ctx from upstream
    discovered = await upstream.discover_n_ctx()
    if discovered:
        config.n_ctx = discovered
        _log.info(
            "n_ctx=%d discovered from upstream. prune_limit=%d (reserve=%d)",
            discovered, config.get_prune_limit(), config.context_reserve,
        )
    else:
        # Not fatal any more: /health re-reads /props every 15 s while the
        # desktop app is up and adopts the real window then (A-2026-01). Until
        # it does, the limit is the pessimistic fallback rather than the old
        # hardcoded 120 000, which was larger than some models' whole context.
        _log.warning(
            "Could not discover n_ctx from upstream. Assuming n_ctx=%d until the "
            "first /health poll — prune_limit=%d",
            # The FIELD, not the class constant. `get_prune_limit()` reads
            # `config.fallback_n_ctx`, which settings.json and the desktop app
            # can change; `FALLBACK_N_CTX` is only its default. The two agreed
            # while nobody had touched the setting, and stopped agreeing the
            # moment the eGPU preset carried 262144: this line would have
            # printed "Assuming n_ctx=32768 ... prune_limit=245760", a sentence
            # whose own two halves contradict each other. Same shape as the
            # `is_summary` incident — a log confidently reporting a number the
            # code does not use.
            config.fallback_n_ctx, config.get_prune_limit())

    await _report_embedding_service(config)

    app.state.config = config
    app.state.upstream = upstream
    app.state.audit = AuditLogger(config)
    # Not asyncio.Semaphore: that is FIFO, so a title generator holds the one
    # slot while the agentic turn a human is watching waits behind it.
    # gateway/scheduler.py carries the measurement.
    app.state.concurrency_semaphore = PrioritySemaphore(config.concurrency_limit)
    # Turns a runaway guard ended, per session. In-process rather than derived
    # from the conversation: the gateway owns the fact that it cut a turn, and
    # reading it back out of the `[SYSTEM GUARD]` text would make that marker
    # forgeable by anything the model can write or read.
    app.state.guard_cut_streak = runaway.GuardCutStreak()
    app.state.synthesis_budget = SynthesisBudget(
        max_per_window=config.max_synthesis_per_hour,
        window_seconds=config.synthesis_window_seconds,
    )
    app.state.spill_store = spill.SpillStore(config.log_dir / "spill")
    # Retention, enforced here and only here. These files are literal copies of
    # content from every project the gateway serves, and nothing deleted one
    # before 2026-09-03 (G-2026-01). Startup is enough: this gateway restarts
    # after every code change, so the interval between runs is far shorter than
    # the threshold — and a timer would add a coroutine to cancel on shutdown
    # and a race with the spill writer, for nothing.
    app.state.spill_store.prune(config.spill_retention_days)
    # The prompt dumps are the same kind of thing, only worse: a full prompt
    # carries the user's source. README.md says "turn it back off" without
    # saying what to clean up, so the same threshold applies here.
    spill.SpillStore(config.log_dir / "prompts").prune(config.spill_retention_days)

    # What the registry actually resolved to, said out loud at every start.
    #
    # An empty registry is SILENT by design -- it is the correct state for a
    # single-engine preset, and everything downstream of it degrades to the
    # old alias path without a word. That silence is what let a restarted
    # gateway serve no models at all on 2026-09-13 while the dashboard showed
    # four: the only way to find out was to ask `/v1/models` and notice what
    # was missing. A count costs one line and makes the next such gap loud.
    _registry = config.registry()
    if len(_registry):
        _log.info("MODEL REGISTRY: %d model(s) — %s", len(_registry),
                  ", ".join(f"{s.id} ({s.engine})" for s in _registry))
    else:
        _log.info("MODEL REGISTRY: empty — serving %s with the alias map only",
                  config.upstream)

    # GATEWAY_VERSION, never a literal. This line said "v6.1.0" for two
    # releases while `/health` said 7.0.0-Lodestar beside it, because
    # `test_version_is_one_number` compares the files a human EDITS and a
    # format string is not one of them. It is the startup banner, so it is
    # the first number anybody reads when diagnosing which build is running.
    _log.info("Gateway v%s started — %s:%d -> %s",
              GATEWAY_VERSION, config.host, config.port, config.upstream)

    yield

    await upstream.shutdown()
    _stop_polystrata_server()
    _log.info("Gateway shutdown complete")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

# One spelling. It was three -- this line, /health, and gateway/__init__.py,
# which said 2.4.0 because nothing read it.
GATEWAY_VERSION = "7.0.0-Lodestar"

app = FastAPI(title="faststrata-gateway", version=GATEWAY_VERSION, lifespan=lifespan)


# ---------------------------------------------------------------------------
# Auth middleware (optional)
# ---------------------------------------------------------------------------

# Reachable without a token even when one is configured. /health carries no
# model data and the desktop app polls it every 15 s.
PUBLIC_PATHS = frozenset({"/health"})


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Require the bearer token on every path except PUBLIC_PATHS.

    This used to guard only paths starting with /v1/, which left the catch-all
    proxy route wide open: POST /completion reached llama-server with no token
    at all, skipping LODESTAR, the FSM and tool validation along the way.
    """
    config: GatewayConfig = request.app.state.config
    if config.auth_token and request.url.path not in PUBLIC_PATHS:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {config.auth_token}":
            return JSONResponse(status_code=401, content={"error": {"message": "Unauthorized"}})
    return await call_next(request)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(request: Request):
    """Health check with upstream ping."""
    upstream: UpstreamClient = request.app.state.upstream
    config: GatewayConfig = request.app.state.config
    upstream_status = await upstream.health_check()
    if (upstream_status.get("status") not in ("ok", "sleeping")):
        if is_polystrata_upstream(getattr(config, "upstream", "")):
            await _ensure_polystrata_server(config)
            upstream_status = await upstream.health_check()

    # This poll already holds a fresh /props, and it is the only thing that
    # asks upstream anything on a schedule — so it is where a model swap gets
    # noticed. Without it `n_ctx` stayed whatever it was at startup for the
    # life of the process (A-2026-01). Ignored when the upstream is down: a
    # dead server reports nothing, which is not the same as a smaller window.
    if upstream_status.get("status") == "ok":
        previous = config.adopt_upstream_n_ctx(upstream_status.get("n_ctx"),
                                               upstream_status.get("model"))
        if previous is not None:
            _log.warning(
                "UPSTREAM CHANGED: n_ctx %s -> %s (model=%s). prune_limit is now "
                "%d — the next turn prunes against the new window, so expect one "
                "cold KV prefix.",
                previous, config.n_ctx, upstream_status.get("model"),
                config.get_prune_limit(),
            )

    return {
        "status": "ok",
        "version": GATEWAY_VERSION,
        "upstream": upstream_status,
        "config": {
            "n_ctx": config.n_ctx,
            "prune_limit": config.get_prune_limit(),
            # The preset NAME stays, because it is what settings.json and the
            # dashboard talk about -- but it is not an answer to "what is
            # this model sampled with", and for most of 2026 it was not even
            # close to one. See `effective_sampling`.
            "sampling_preset": config.sampling_preset,
            "sampling": config.effective_sampling(),
        },
    }


@app.get("/metrics")
async def metrics_endpoint(request: Request):
    """Prometheus exposition of what only the gateway can see.

    NOT in PUBLIC_PATHS: this reports queue depth, token volume, error types
    and how often the supervisor had to act for the model. That is operational
    detail about somebody's work, and a scraper can send a bearer token as
    easily as it can send none.

    The counters come from the audit path; the gauges are read here, because a
    queue depth is only true at the moment it is asked for.
    """
    config: GatewayConfig = request.app.state.config
    semaphore = getattr(request.app.state, "concurrency_semaphore", None)
    gauges: dict[str, float] = {}
    if isinstance(semaphore, PrioritySemaphore):
        gauges["llmgw_slot_in_use"] = float(
            max(0, config.concurrency_limit - semaphore.value))
        gauges["llmgw_queue_depth"] = float(semaphore.waiting())
        gauges["llmgw_queue_depth_interactive"] = float(
            semaphore.waiting_at(INTERACTIVE))

    body = metrics.REGISTRY.render(
        gauges=gauges,
        info={"version": GATEWAY_VERSION, "prompts_version": prompts.prompts_version()},
    )
    return PlainTextResponse(
        body, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Main chat completion pipeline."""
    config: GatewayConfig = request.app.state.config
    upstream: UpstreamClient = request.app.state.upstream
    audit: AuditLogger = request.app.state.audit
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    t_start = time.time()

    raw_body = await request.body()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }

    try:
        client_body = json.loads(raw_body or b"{}")
    except Exception:
        audit.log_error(request_id, "bad_json", "Invalid JSON in request body")
        return JSONResponse(status_code=400, content={"error": {"message": "bad JSON"}})

    # 1. Resolve model alias, then the registry entry it names.
    #
    # The registry is what a preset with more than one engine needs and an
    # alias map cannot express: an alias maps a name to another name, and the
    # thing that has to be decided here is which PROCESS gets the card. A
    # preset with no `models` list resolves nothing and everything below is
    # inert, which is what keeps a single-engine rig exactly as it was.
    original_model = client_body.get("model") or ""
    model_spec = config.spec_for(original_model)
    # Everything that scales with the window is now a property of the MODEL,
    # resolved here into a per-request copy so every reader below goes on
    # taking `config` exactly as it did. A model that overrides nothing gets
    # the same object back, so a single-model rig allocates nothing.
    #
    # It has to happen before ANY of it is read — the prune limit, the spill
    # threshold and the sampling are all decided further down this function,
    # and a copy made after the first of them would apply to some and not
    # others.
    config = config.for_spec(model_spec)
    if model_spec is not None:
        # The engine's own name for the model, which is not the name the
        # client asked for and must not be: the target changes when the quant
        # does, and the client should not have to follow.
        client_body["model"] = model_spec.target or model_spec.id
    else:
        client_body["model"] = config.resolve_model(original_model)

    wants_stream = bool(client_body.get("stream"))
    client_tools = client_body.get("tools") or []

    # Everything the client asked for that this gateway does not do. The body
    # sent upstream is composed from scratch, so a parameter nothing reads is
    # dropped in SILENCE -- and a client cannot tell "ignored" from "honoured
    # and had no effect". gateway/client_params.py carries the disposition of
    # every parameter and the argument for each; this is where the ones that
    # would answer a different question become a 400 instead.
    # The engine that will actually serve this request decides what it can
    # be asked for: the model's own upstream, not the preset's default.
    _serving_upstream = (model_spec.upstream if model_spec is not None
                         else getattr(config, "upstream", ""))
    can_constrain = upstream_can_constrain(_serving_upstream)
    # A JSON schema is a DIFFERENT question from a GBNF grammar, and the two
    # shared one flag until 2026-09-15 -- which refused `response_format` on an
    # engine that took a schema and no grammar.
    refusals = client_params.check(
        client_body, can_constrain=can_constrain,
        can_schema=upstream_takes_json_schema(_serving_upstream))
    if refusals:
        _log.warning("UNSUPPORTED PARAMS id=%s %s", request_id, "; ".join(refusals))
        audit.log_error(request_id, "unsupported_params", "; ".join(refusals))
        return JSONResponse(status_code=400, content={"error": {
            "message": "this gateway cannot serve the request as asked: "
                       + "; ".join(refusals),
            "type": "invalid_request_error",
        }})

    tool_mode, forced_tool = client_params.tool_choice_mode(client_body)

    # Audit: log incoming request. The workspace is resolved up front so every
    # event of one task shares a session_id — request_id is per HTTP call, so
    # without it the JSONL cannot be reassembled into turns.
    messages = client_body.get("messages") or []
    session_id = _extract_workspace_dir(messages)

    # Detected HERE rather than at step 4 below, where it is used. The flag was
    # computed 80 lines after it was logged, so the audit recorded False for
    # every request ever made — which is why a detection string that never
    # matched went unnoticed for months. A field that cannot be wrong is not
    # evidence. Detection reads only `messages`, which is already in hand.
    #
    # Reading through _message_text is what makes moving it here safe: this now
    # runs BEFORE normalization, so `content` is still whatever the client sent
    # — and OpenCode sends user messages as content-part LISTS. Calling .lower()
    # on one raises before the audit event is even written, which is exactly
    # what happened on 2026-09-01 14:04: every request 500'd and the JSONL had
    # no record of any of them, because the throw preceded the logging.
    summary_kind = summary_request_kind(messages)
    is_summary = summary_kind is not None
    audit.log_request(
        request_id=request_id,
        session_id=session_id,
        model=client_body["model"],
        stream=wants_stream,
        message_count=len(messages),
        tools_count=len(client_tools),
        is_summary=is_summary,
    )

    # LODESTAR applies only to agentic requests. Tool-less traffic — graph
    # builders (graphify/engram), title generators, plain completions — must
    # not touch the state machine at all: it cannot follow the protocol
    # anyway, and per-request state logging was flooding gateway.log with
    # corpus content during graph builds (2026-07-16).
    #
    # tool_choice="none" makes this non-agentic by the same argument: a request
    # that must not call tools cannot follow a protocol whose every step is a
    # tool call, and injecting the tool list would only invite what the client
    # just said it cannot handle.
    is_agentic = bool(client_tools) and tool_mode != "none"

    # 1b. Did the client throw the picture away on our behalf? Asked of the RAW
    # messages, and only for a model that could have served them: a text model
    # having its images stripped is the client behaving correctly. When the
    # gateway would have accepted an image and none arrived, the failure is
    # entirely upstream of this process and leaves no trace of its own -- the
    # 2026-09-14 incident was visible only as the ABSENCE of a MULTIMODAL line.
    # A log line is all this can be: the payload is gone, and the fix is a key
    # in the client's config that no request can supply.
    if model_spec is not None and model_spec.vision:
        for _rep in normalizer.stripped_media_reports(
                normalizer.detect_stripped_media(messages),
                vision=model_spec.vision, model=str(client_body["model"])):
            (_log.warning if _rep.level == "warning" else _log.info)(
                "%s", _rep.message)
            if _rep.audit_error:
                audit.log_error(request_id=request_id,
                                error_type=_rep.audit_error,
                                message=_rep.message)

    # 2. Normalize conversation (merge system, fix orphaned tools, strip reasoning)
    client_body["messages"] = normalizer.normalize_messages(
        messages, config, agentic=is_agentic,
        vision=bool(model_spec is not None and model_spec.vision))

    if is_agentic:
        lodestar_state = normalizer.determine_lodestar_state(client_body["messages"], config)
        _log.info("LODESTAR STATE MACHINE: state = %d", lodestar_state)
    else:
        lodestar_state = 3  # LodestarState.READY — plain proxy behaviour

    if lodestar_state < 3:
        # Remove skill-related tools from available tool declarations to hide them from model
        client_tools = [
            t for t in client_tools
            if t.get("function", {}).get("name") not in ("skill", "load_skill", "read_skills_metadata")
        ]
        if "tools" in client_body:
            client_body["tools"] = client_tools

    # 3a-. Clean the model's own corrupted paths out of the history before it
    # sees them again. Placed here, after LODESTAR derived its state, for the
    # same reason spill is: the state machine must judge what the client
    # actually sent. Measured 2026-09-01 — a prompt carrying the model's own
    # `PROJEKTOACME_HW` twice produced `PROJKTOACME_HW` next, that string minus
    # one character. Left in place, every failed path becomes the template for
    # a worse one.
    if is_agentic and session_id:
        client_body["messages"], _path_fixes = path_repair.repair_history(
            client_body["messages"], session_id)
        if _path_fixes:
            _log.warning("PATH REPAIR: cleaned %d corrupted path(s) out of the "
                         "history so they stop being examples", _path_fixes)

    # 3a. Spill oversized tool results before pruning decides what to drop.
    # Order matters: pruning evicts whole conversational blocks, so shrinking
    # the fat ones first means fewer turns have to go. LODESTAR ran above and
    # its evidence is exempt from spilling, so the state it derived stays
    # valid for the rest of the pipeline.
    spill_count = 0
    spill_store = getattr(request.app.state, "spill_store", None)
    if config.spill_max_inline_bytes > 0 and spill_store is not None:
        bytes_before = _messages_bytes(client_body["messages"])
        client_body["messages"], spill_count = spill.apply_spill_policy(
            client_body["messages"],
            config.spill_max_inline_bytes,
            spill_store,
            session_id=session_id,
        )
        if spill_count:
            audit.log_spill(
                request_id=request_id, session_id=session_id,
                spilled_count=spill_count, bytes_before=bytes_before,
                bytes_after=_messages_bytes(client_body["messages"]),
            )

    # 3. Prune context
    client_body["messages"] = await normalizer.prune_messages(
        client_body["messages"], upstream, max_tokens=config.get_prune_limit(),
        audit=audit, request_id=request_id, session_id=session_id,
        compaction=config.compaction_enabled,
        image_tokens=(config.image_tokens_estimate
                      if model_spec is not None and model_spec.vision else 0),
        # The tokenizer of the model this request is GOING to, not of whichever
        # engine the preset names. Pruning runs before the hand-over, so on the
        # first turn of a swap that address is not answering yet and the
        # character heuristic is used -- which is what happened silently on
        # every vision turn before this, after ten seconds of timeouts.
        tokenizer_url=(model_spec.upstream
                       if model_spec is not None and model_spec.upstream
                       else None),
    )
    compacted = any(
        normalizer.COMPACTION_MARKER in str(m.get("content") or "")
        for m in client_body["messages"]
    )

    # 4. Summarization request (detected above, before the audit event)
    if summary_kind == SUMMARY_ANCHORED:
        # The system prompt and the tool list stay byte for byte what the
        # agent's turns had, because they ARE the cached prefix the client
        # kept them for. The instruction goes where nothing is cached yet --
        # the end of the request itself -- and a tool call the model tries
        # anyway is dropped at emission (`emit_event(is_summary=True)`).
        _log.info("SUMMARY MODE (anchored): prompt prefix kept, tool calls "
                  "suppressed at emission, every agentic guard off "
                  "for request_id=%s", request_id)
        _append_to_last_user(client_body.get("messages", []),
                             "\n\n" + prompts.get("summary_mode"))
    elif is_summary:
        _log.info("SUMMARY MODE: tools stripped and every agentic guard off "
                  "for request_id=%s", request_id)
        client_tools = []
        if "tools" in client_body:
            client_body["tools"] = []
        # Append strict warning
        sys_msg = None
        for m in client_body.get("messages", []):
            if m.get("role") == "system":
                sys_msg = m
                break
        warning = prompts.get("summary_mode")
        if sys_msg:
            sys_msg["content"] = str(sys_msg.get("content", "")) + "\n\n" + warning
        else:
            client_body.setdefault("messages", []).insert(
                0, {"role": "system", "content": warning})

    # 5. Inject tools into system prompt (bypass native parser).
    # Nothing is injected under tool_choice="none": the client said it cannot
    # execute a call, so describing the tools can only produce one. That also
    # strips the `tools` array from the body, which is what keeps llama-server's
    # own parser out of it either way.
    normalizer.inject_tools_to_prompt(
        client_body, [] if tool_mode == "none" else client_tools,
        dialect=config.tool_dialect)
    client_body.pop("tools", None)

    workspace_dir = _extract_workspace_dir(client_body.get("messages") or [])

    # 5b. Tiered routing. Inert unless an escalation model is configured.
    #
    # A registry entry sets the address first: on a hot-swap rig the model IS
    # the routing decision. Escalation may still override it below, which is
    # the right precedence — an escalation target is an explicit instruction
    # and a registry entry is a default.
    upstream_override: str | None = (
        model_spec.upstream if model_spec is not None else None)
    if is_agentic and config.escalation_model:
        depth = adaptive.should_escalate(
            client_body.get("messages") or [], config.escalation_retry_depth)
        if depth:
            target = config.resolve_model(config.escalation_model)
            upstream_override = config.escalation_upstream
            _log.warning(
                "ESCALATION: retry depth %d — routing to %s%s",
                depth, target,
                f" via {upstream_override}" if upstream_override else "",
            )
            audit.log_escalation(
                request_id=request_id, session_id=session_id,
                from_model=client_body["model"], to_model=target,
                retry_depth=depth, upstream=upstream_override,
            )
            client_body["model"] = target
            # Appended last so it is the most recent thing in the system
            # prompt. This does move the KV prefix — acceptable for an event
            # that only happens once a loop has already cost several turns.
            directive = prompts.get("escalation_directive", depth=depth)
            for m in client_body.get("messages") or []:
                if m.get("role") == "system":
                    m["content"] = f"{str(m.get('content') or '')}\n\n{directive}"
                    break

    # 6. Stream or non-stream
    # Open this turn's books on the guard-cut streak: `begin` also closes the
    # PREVIOUS turn, so a session whose last request finished without a cut
    # starts again at zero. One call site rather than a `clear()` that every
    # path a turn can end by would have to remember.
    guard_streak = getattr(request.app.state, "guard_cut_streak", None)
    if guard_streak is not None:
        guard_streak.begin(workspace_dir, request_id)

    if wants_stream:
        return StreamingResponse(
            _stream_chat_v2(client_body, headers, client_tools, is_summary,
                           upstream, config, audit, request_id,
                           semaphore=request.app.state.concurrency_semaphore,
                           workspace_dir=workspace_dir,
                           budget=request.app.state.synthesis_budget,
                           spill_count=spill_count,
                           compaction=compacted,
                           upstream_override=upstream_override,
                           model_spec=model_spec,
                           guard_streak=guard_streak),
            media_type="text/event-stream",
        )

    # --- Non-streaming path (also uses FSM for unified parsing) ---
    return await _non_stream_completion(
        client_body, headers, client_tools, is_summary, upstream, config,
        audit, request_id, t_start,
        semaphore=request.app.state.concurrency_semaphore,
        workspace_dir=workspace_dir,
        budget=request.app.state.synthesis_budget,
        spill_count=spill_count,
        compaction=compacted,
        upstream_override=upstream_override,
        model_spec=model_spec,
        guard_streak=guard_streak,
    )


# ---------------------------------------------------------------------------
# Streaming pipeline (FSM + SSE emitter)
# ---------------------------------------------------------------------------

class _SlotTimeout(Exception):
    """The upstream concurrency slot never freed up within the timeout."""


@dataclass
class TurnState:
    """Everything about ONE attempt at a turn, in one place.

    These fields used to be fifteen locals rebound through a `nonlocal` list
    inside `_retry_out_of_the_loop`, and the risk was concrete rather than
    aesthetic: **a new guard had to know to clear its own state there**, or
    after a silent retry it would fire immediately on the abandoned
    generation's text — or never fire at all. That is not hypothetical.
    `ttft_ms` was already missing from that list, so for any turn that went
    through a retry the audit log's time-to-first-token described a generation
    that had been thrown away — and `check_prefill_health.py` reads exactly
    that field to judge whether the KV prefix was warm. O-2026-04, R-2026-02.

    `reset_for_retry()` is the whole point: adding a field now means adding it
    here, and `TestTurnStateResets` compares against `dataclasses.fields`, so a
    field that is not cleared fails the suite rather than a production turn.
    """

    #: Survives a retry by design — it is the count OF retries.
    PERSIST: ClassVar[frozenset[str]] = frozenset(
        {"reasoning_retries", "content_continues", "loop_recoveries",
         "answer_rescues"})

    finish_reason: str = "stop"
    raw_tail: str = ""
    ttft_ms: int | None = None

    reasoning_retries: int = 0
    content_continues: int = 0
    loop_recoveries: int = 0
    answer_rescues: int = 0
    reasoning_guard_tripped: bool = False
    #: What runaway.RunawayWatch said, kept so the final flush can render the
    #: guard's own text. The throttles that produced it live in the watch
    #: itself, which is reset alongside this on a retry.
    runaway_verdict: runaway.Verdict | None = None
    #: The upstream ended this attempt at its token limit (n_predict), not by
    #: choice. A tool call the final flush has to close was cut off by it.
    cut_by_limit: bool = False
    #: Such a cut-off call had already streamed, so it cannot be withdrawn and
    #: the turn has to end `length`, which tells the client not to run it.
    truncated_call_streamed: bool = False

    def reset_for_retry(self) -> None:
        """Return every per-attempt field to its starting value.

        Everything except `PERSIST`: the guards must judge the NEW generation
        on its own text, and the counter must keep counting or the retry limit
        would never be reached.
        """
        fresh = TurnState()
        for f in dataclasses.fields(self):
            if f.name in self.PERSIST:
                continue
            setattr(self, f.name, getattr(fresh, f.name))


def _finish_on_upstream_stop(st: TurnState, chunk: dict, tool_stats: dict) -> None:
    """The finish reason when the upstream ends a generation. All four readers.

    A limit stop used to overwrite `tool_calls` with `length`, and OpenCode
    runs NO call from a turn that ended `length`. Measured 2026-10-17 12:08: a
    turn that had finished eight calls (five of them edits) reached
    n_predict while writing on, and 310 s of work was thrown away whole. Calls
    the model closed itself are complete and are kept; a call the limit cut
    in half is the final flush's to judge (`_flush_truncated_call`).
    """
    limit = bool(chunk.get("stopped_limit") or chunk.get("stop_type") == "limit")
    if limit and not (tool_stats.get("turn_budget_chunk_sent")
                      or tool_stats.get("turn_loop_stop")):
        st.cut_by_limit = True
        if st.finish_reason == "tool_calls":
            _log.warning("LENGTH CUT after complete tool calls -- finishing "
                         "`tool_calls` so the client runs the calls that closed")
        else:
            st.finish_reason = "length"
    elif st.finish_reason != "tool_calls":
        st.finish_reason = "stop"


def _flush_truncated_call(fsm_event: FsmEvent, st: TurnState,
                          emitter: OpenAISSEEmitter) -> bool:
    """True when the final flush is closing a call the token limit cut off.

    Anything the flush closes after a limit stop was closed by the GATEWAY,
    not the model -- `auto_closed` or a JSON tail `repair_json` completed --
    so its arguments end wherever the limit fell. Measured on the parser: a
    `write` cut mid-content came out as a well-formed call with half a file in
    it, and the turn finished `tool_calls`. A call still held back is dropped
    unseen; one that streamed cannot be, so the turn is made to end `length`,
    which is the client's own rule for not running anything.
    """
    if not st.cut_by_limit or fsm_event.type != EventType.TOOL_CALL_END:
        return False
    idx = fsm_event.data.get("index", -1)
    name = fsm_event.data.get("name") or "?"
    if emitter.has_streamed(idx):
        st.truncated_call_streamed = True
        _log.warning("LENGTH CUT inside '%s', which has already streamed -- "
                     "ending the turn `length` so the client does not run it", name)
    else:
        emitter.suppress(idx)
        _log.warning("LENGTH CUT inside '%s' -- dropping the half-written call", name)
    return True


async def _acquire_slot(
    semaphore: PrioritySemaphore,
    config: GatewayConfig,
    priority: int = INTERACTIVE,
) -> AsyncIterator[bytes]:
    """Wait for the upstream slot, yielding SSE keep-alives while queued.

    Only one request may reach llama-server at a time, and a queued client will
    hit its own read timeout long before that slot frees up — hence the
    keep-alive comments. Raises _SlotTimeout when the wait runs out; the caller
    turns that into a 503-shaped error chunk.
    """
    total_wait = config.semaphore_timeout
    wait_interval = config.semaphore_keepalive_interval
    elapsed = 0.0
    while elapsed < total_wait:
        this_timeout = min(wait_interval, total_wait - elapsed)
        try:
            await asyncio.wait_for(semaphore.acquire(priority), timeout=this_timeout)
            return
        except asyncio.TimeoutError:
            elapsed += this_timeout
            if elapsed >= total_wait:
                # from None: the queue timing out is the signal, and the
                # asyncio.TimeoutError underneath it is an implementation
                # detail of how the wait was chunked.
                raise _SlotTimeout from None
            yield b": keep-alive\n\n"
    raise _SlotTimeout


async def _fetch_upstream_to_queue(
    q: asyncio.Queue,
    upstream: UpstreamClient,
    generate_body: dict,
    headers: dict,
    chat_id: str,
    upstream_override: str | None = None,
    stop: asyncio.Event | None = None,
    resp_holder: dict | None = None,
) -> None:
    """Pump the upstream SSE stream into a queue.

    Runs as its own task so the response generator can keep emitting
    heartbeats while waiting on the model. Queue items are tagged tuples:
    ("headers", ...), ("line", str), ("error", bytes), ("exception", exc) and
    a final ("done", None) that always arrives, however the task ended.

    `stop` is the cooperative shutdown signal: set it and the read loop breaks
    at the next line, so this task runs its own `finally` and closes the httpx
    stream properly. That close is what makes llama-server abort the slot —
    see _shutdown_upstream_worker. `resp_holder` publishes the live response so
    the caller can sever it by force if the loop is wedged.
    """
    try:
        server_body = generate_body.copy()
        model_name = generate_body.get("model") or ""
        resolved = upstream_override or upstream.resolve()
        if "messages" in server_body:
            # The chat endpoint takes `model` as part of its contract, unlike
            # /completion where the engine serves whatever it loaded and the
            # field is noise.
            server_body["model"] = model_name
        else:
            server_body.pop("model", None)
        payload = json.dumps(server_body).encode("utf-8")
        request_headers = {**headers, "content-type": "application/json", "x-model-name": str(model_name or "")}

        # A chat body has `messages`, a completion body has `prompt`; the
        # endpoint follows from the BODY rather than from a flag somebody
        # has to remember to pass. `/completion` is where this gateway has
        # always gone, and the only reason the other exists is that it is
        # the only one that sees an image -- see build_chat_body.
        path = "/v1/chat/completions" if "messages" in server_body else "/completion"
        url = f"{resolved}{path}"
        try:
            stream_ctx = upstream.client.stream("POST", url, content=payload,
                                                headers=request_headers)
            upstream_resp = await stream_ctx.__aenter__()
        except (httpx.NetworkError, httpx.TimeoutException) as exc:
            # A sleeping Mac resolves fine but refuses the connection; force a
            # fresh mDNS lookup before giving up.
            _log.warning("Stream connection failed to %s: %s. Re-resolving and retrying.",
                         url, exc)
            if is_polystrata_upstream(url):
                await _ensure_polystrata_server(upstream._config)
            # An escalation target is an explicit address, not a discovered
            # one — re-resolving would silently fall back to the primary.
            resolved = upstream_override or upstream.resolve(force=True)
            url = f"{resolved}{path}"
            stream_ctx = upstream.client.stream("POST", url, content=payload,
                                                headers=request_headers)
            upstream_resp = await stream_ctx.__aenter__()

        if resp_holder is not None:
            resp_holder["response"] = upstream_resp
        try:
            await q.put(("headers", (upstream_resp.status_code, dict(upstream_resp.headers))))
            if upstream_resp.status_code == 200:
                async for line in upstream_resp.aiter_lines():
                    await q.put(("line", line))
                    if stop is not None and stop.is_set():
                        _log.info(
                            "STREAM: upstream read stopped on request chat_id=%s", chat_id)
                        break
            else:
                await q.put(("error", await upstream_resp.aread()))
        finally:
            await stream_ctx.__aexit__(None, None, None)
    except Exception as exc:
        _log.error("STREAM FETCH_UPSTREAM EXCEPTION chat_id=%s error=%s", chat_id, exc)
        await q.put(("exception", exc))
    finally:
        await q.put(("done", None))


class _StreamPacing:
    """When the client last heard anything, so a heartbeat can be timed.

    An object rather than a local, because `_upstream_events` needs to read it
    while the caller keeps updating it — and because the silent retry rebinds
    its locals, which a captured value would not follow.
    """

    __slots__ = ("last_yield",)

    def __init__(self) -> None:
        self.last_yield = time.time()

    def touch(self) -> None:
        self.last_yield = time.time()


_REASONING_FIELD_WARNED = False


def _merge_final_usage(final_usage: dict | None, chunk: dict) -> dict:
    """One place where an upstream stop chunk becomes a usage object.

    It stood written out four times, once per upstream reader, and the four
    copies had already drifted: two REPLACED the usage and two ACCUMULATED.
    So a turn whose generation was abandoned and resumed reported only its
    last attempt's output, while a turn that merely continued reported all of
    it -- the same turn shape, two different numbers, depending on which of
    the four readers happened to see the stop. Accumulating is the correct
    half: every attempt's tokens were decoded on the one slot this rig has,
    and `tokens_out` is what the audit prices a turn with.

    `prompt_tokens` stays the FIRST reader's. A continuation re-sends the
    conversation plus everything generated so far, so adding those up would
    report a prompt that was never sent.

    `cached_tokens` rides in OpenAI's own `prompt_tokens_details`, so it is a
    field a client already knows how to read rather than a private key it has
    to ignore. It is the FIRST observation for the same reason: the first
    attempt is the one that paid for the prefill, and a continuation against a
    now-warm prefix would otherwise overwrite that with a flattering number.
    Only ExLlamaV3 reports it today (llama.cpp's `tokens_evaluated` is the
    whole prompt whether it read it or reused it), and absent stays absent --
    a zero here would be read as a cold prefix.
    """
    prompt_tokens = int(chunk.get("tokens_evaluated", 0) or 0)
    completion_tokens = int(chunk.get("tokens_predicted", 0) or 0)
    if final_usage is None:
        final_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    else:
        final_usage["completion_tokens"] = (
            final_usage.get("completion_tokens", 0) + completion_tokens)
        final_usage["total_tokens"] = (
            final_usage.get("prompt_tokens", 0) + final_usage["completion_tokens"])
    cached = chunk.get("cached_tokens")
    if cached is not None:
        details = final_usage.setdefault("prompt_tokens_details", {})
        details.setdefault("cached_tokens", int(cached))
    return final_usage


def cached_tokens_of(usage: dict | None) -> int | None:
    """The engine's own prefix-cache hit, or None when nobody reported one."""
    if not usage:
        return None
    detail = usage.get("prompt_tokens_details") or {}
    value = detail.get("cached_tokens")
    return int(value) if value is not None else None


class UpstreamReportedError(Exception):
    """The upstream said, mid-stream, that it had failed.

    Distinct from `httpx.TransportError` (the socket died) and from a non-200
    body (the request was refused before generating anything). Here the
    connection is healthy and the server is telling us, in band, that the
    generation is over -- which until 2026-09-18 reached the user as an
    EMPTY TURN.
    """

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


def upstream_error_text(chunk: dict) -> str | None:
    """The message out of an upstream SSE chunk that reports a failure.

    Both engines report a mid-generation failure the same way: a chunk
    carrying `error` and nothing usable, followed by `[DONE]`. ExLlamaV3's
    `sse_stream` catches the exception and emits
    `{"error": {"message": "eGPU inference error: ..."}}`; llama.cpp emits the
    OpenAI shape, with `code` and `type` beside the message.

    Neither carries `content` or `stop`, so before this existed every reader
    simply ran out of stream and closed the turn `finish=stop` with nothing in
    it. Measured 2026-09-18: a CUDA out-of-memory during prefill reached the
    user as a model that answered nothing, `gateway.log` said not one word
    about it, and the only record was in the engine's own log -- which is why
    it was read as a model that would not start.

    ONE recognizer, because there are six readers and the shape is the same
    for all of them; `TestEveryReaderNoticesAnUpstreamError` counts the call
    sites in SOURCE, since a reader is covered when it CALLS this and never
    when prose says so.
    """
    err = chunk.get("error")
    if not err:
        return None
    if isinstance(err, str):
        return err.strip() or None
    if isinstance(err, dict):
        for key in ("message", "detail", "msg"):
            val = err.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return str(err)[:500] or None


def _as_completion_chunk(chunk: dict) -> dict:
    """One chunk shape for every reader, whichever endpoint produced it.

    Six places in this file read a chunk -- the main loop, the auto-resume,
    the content continuation, the loop recovery, `_upstream_events` for all
    four of those, and `_non_stream_completion`, which parses its own JSON.
    So the SHAPE is normalised here and every reader is untouched by the
    second transport existing at all.

    **This docstring said FIVE and named the non-streaming path among them
    while that path never called this function.** It reads
    `chunk["content"]`, which a chat chunk does not have, so the first live
    vision request returned an empty answer with `finish_reason: stop` --
    nothing logged, nothing raised. That is this file's own recurring
    finding, and the docstring asserting the coverage is what made it hard to
    see. A reader is covered when it CALLS this, never when prose says so.

    Told apart by the payload rather than by a flag: only the chat endpoint
    emits `choices`.
    """
    if "choices" not in chunk:
        return chunk

    global _REASONING_FIELD_WARNED
    choices = chunk.get("choices") or [{}]
    choice = choices[0] if choices else {}
    delta = choice.get("delta") or {}
    content = delta.get("content") or ""

    if not content and delta.get("reasoning_content"):
        # The vision server is started with `--reasoning-format none` so that
        # `<think>` reaches the FSM as text, the way it does on every other
        # path. If a build ignores that, the reasoning would otherwise be
        # dropped silently -- so it is passed through and said out loud once.
        if not _REASONING_FIELD_WARNED:
            _REASONING_FIELD_WARNED = True
            _log.warning(
                "Upstream is splitting reasoning into `reasoning_content` "
                "despite --reasoning-format none; the FSM will see it as "
                "content and cannot label it as thinking.")
        content = delta["reasoning_content"]

    out: dict[str, Any] = {"content": content}
    finish = choice.get("finish_reason")
    if finish:
        out["stop"] = True
        if finish == "length":
            out["stopped_limit"] = True
    usage = chunk.get("usage") or {}
    timings = chunk.get("timings") or {}
    if usage:
        out["tokens_evaluated"] = usage.get("prompt_tokens", 0)
        out["tokens_predicted"] = usage.get("completion_tokens", 0)
    elif timings:
        # `usage` needs `stream_options.include_usage`, which this gateway
        # does not send; measured against b10869, the final chat chunk carries
        # `timings` and no `usage` at all. Without this every vision turn
        # reported zero tokens in and out, to the audit log and to the client.
        #
        # **cache_n + prompt_n, not prompt_n.** Measured on the same server:
        # /completion reports `tokens_evaluated` 203 on a cold AND a warm run
        # of the same prompt while `prompt_n` falls from 203 to 4 -- so the
        # field this gateway has always recorded is the WHOLE prompt, and
        # `prompt_n` is only what was re-read. Mapping `prompt_n` would put a
        # different quantity under the same name for one transport, which
        # `check_prefill_health.py` reads as a per-turn delta and would have
        # reported as a permanently cold prefix.
        out["tokens_evaluated"] = (
            timings.get("cache_n", 0) + timings.get("prompt_n", 0))
        out["tokens_predicted"] = timings.get("predicted_n", 0)
    return out


async def _upstream_events(
    get_queue: Callable[[], asyncio.Queue],
    config: GatewayConfig,
    pacing: _StreamPacing,
) -> AsyncIterator[tuple[str, Any]]:
    """The queue-reading half of a stream, written once for every reader.

    `_stream_chat_v2` reads an upstream stream in FOUR places — the main
    generation, the in-stream auto-resume, the content continuation and the
    loop recovery — and each carried its own copy of this: the heartbeat
    timing, the sentinel dispatch, the `data:` prefix, the `[DONE]` line, the
    JSON decode. Copies drift, and the drift is not cosmetic: audit finding
    W-2026-03 was the loop-recovery copy having silently omitted both runaway
    guards, and when this was extracted two MORE copies turned out to have
    neither. A guard that has to be remembered four times is a guard that will
    be present three times.

    Yields, in the caller's terms rather than the queue's:

        ("tick", None)        before every wait — where a caller runs its
                              guards, because the decision to stop belongs to
                              whoever knows what this attempt is for
        ("heartbeat", None)   nothing arrived within the interval; the caller
                              emits and calls `pacing.touch()`
        ("error", bytes)      upstream answered with a non-200 body
        ("chunk", dict)       one decoded SSE payload

    and simply returns when the stream ends. An upstream exception is RAISED
    rather than yielded, so it lands in the caller's own `except` with its type
    intact — which is what lets a crashed upstream be told apart from a proxy
    bug (`httpx.TransportError`).

    The queue arrives as a CALLABLE. The silent retry abandons the generation
    and rebinds `q` to a fresh queue mid-loop, so a captured value would keep
    reading the dead one; `lambda: q` reads the caller's current binding on
    every pass. That is the whole reason this is not a plain parameter.
    """
    while True:
        yield ("tick", None)

        wait_timeout = max(0.1, config.heartbeat_interval
                           - (time.time() - pacing.last_yield))
        try:
            item = await asyncio.wait_for(get_queue().get(), timeout=wait_timeout)
        except asyncio.TimeoutError:
            yield ("heartbeat", None)
            continue

        if item is None:
            return
        event_type, val = item

        if event_type == "done":
            return
        if event_type == "exception":
            raise val
        if event_type == "error":
            yield ("error", val)
            return
        if event_type != "line":
            continue  # "headers", and anything a future producer adds

        if not val.startswith("data: "):
            continue
        payload = val[6:].strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except Exception:
            continue  # a partial line on a live stream is normal
        detail = upstream_error_text(chunk)
        if detail is not None:
            # RAISED rather than yielded, for the reason this docstring gives
            # above: it lands in the caller's own `except`, which already
            # knows how to end a turn without ending the session. There are
            # FOUR callers, so a new event type would be a new branch to
            # forget in three of them.
            raise UpstreamReportedError(detail)
        yield ("chunk", _as_completion_chunk(chunk))


#: Which text the model is given when a runaway guard ends its turn. Keyed on
#: the verdict so the two paths cannot disagree about which guard says what.
_RUNAWAY_PROMPTS = {
    runaway.CODE: "reasoning_code_guard",
    runaway.LOOP: "reasoning_loop_guard",
    runaway.BUDGET: "reasoning_budget_guard",
    runaway.CONTENT_LOOP: "content_loop_guard",
}


def _stream_time_exhausted(stream_start: float, config: GatewayConfig) -> bool:
    """Has this turn held the single upstream slot past the safety valve?

    Extracted so the main read loop and the loop-recovery read loop ask the
    same question. They did not: the recovery loop, added 2026-09-09, was a
    copy of the reader that omitted both this check and the reasoning budget,
    so a turn that entered recovery lost every runaway guard except
    max_n_predict (audit finding W-2026-03).

    0 disables the guard, which is what every other limit in `config.py`
    means by 0 — and what this one did NOT mean until 2026-09-10. The check
    was a bare `elapsed > limit`, so the only value a reader would set to
    switch the safety valve off instead made it fire on the first loop
    iteration, roughly 100 ms into every stream. A switch that cannot be
    switched off is not a switch; one that does the opposite is a trap.
    """
    if config.stream_max_duration <= 0:
        return False
    return time.time() - stream_start > config.stream_max_duration


def wants_chat_transport(spec: "ModelSpec | None") -> bool:
    """Does this model need the chat endpoint, or can it keep the prompt?

    **Vision is no longer the question; the ENGINE is.** llama.cpp was
    measured on 2026-09-13 to ignore `multimodal_data` beside the prompt and
    to reject the documented object form, so a GGUF vision model has to go to
    `/v1/chat/completions` and give up the byte-exact ChatML this gateway
    renders. An engine that honours the marker contract on `/completion`
    keeps the prompt -- and with it the LODESTAR prefill, the `tool_choice`
    grammar prefill, the silent reasoning retry, the in-stream content
    continuation and the loop recovery, all five of which ask
    `prompt_transport` and are simply off on the chat shape.

    Asked through ONE function rather than spelled at each call site, because
    the predicate stood in two places as `model_spec.vision` and a third
    reader would have inherited the old answer. `TestTheTransportIsAskedOnce`
    reads this file as source.
    """
    if spec is None or not spec.vision:
        return False
    # A remote server is reached the way llama.cpp is, which sees an image only
    # on the chat route (the measurement above). FastStrata keeps the prompt.
    return spec.engine == ENGINE_REMOTE


def _build_upstream_body(client_body: dict, config: GatewayConfig,
                         use_chat: bool, guard_cuts: int) -> dict:
    """The upstream request body, by transport.

    Extracted because the same two-line ternary stood in the streaming and the
    non-streaming path, and `guard_cuts` would have had to be remembered in
    both -- which is how this file has lost a guard three times. One call site
    each now, and a new argument reaches both or neither.
    """
    if use_chat:
        return normalizer.build_chat_body(client_body, config, stream=True,
                                          guard_cuts=guard_cuts)
    return normalizer.build_generate_body(client_body, config, stream=True,
                                          guard_cuts=guard_cuts)


async def _stream_chat_v2(
    client_body: dict,
    headers: dict,
    tools: list[dict],
    is_summary: bool,
    upstream: UpstreamClient,
    config: GatewayConfig,
    audit: AuditLogger | None = None,
    request_id: str = "",
    semaphore: PrioritySemaphore | None = None,
    workspace_dir: str | None = None,
    budget: SynthesisBudget | None = None,
    spill_count: int = 0,
    compaction: bool = False,
    upstream_override: str | None = None,
    model_spec: "ModelSpec | None" = None,
    guard_streak: "runaway.GuardCutStreak | None" = None,
) -> AsyncIterator[bytes]:
    """FSM-based streaming pipeline.

    Replaces _legacy.stream_chat_completion with clean event-driven architecture:
      QwenXmlFSM → FsmEvent → OpenAISSEEmitter → bytes
    """
    # Which transport serves this model. It follows the MODEL, not the turn:
    # a vision model's text-only turns could use the cheaper prompt path, but
    # alternating shapes inside one conversation would invalidate the
    # upstream's prompt cache on every switch, and the model is chosen by the
    # user precisely when images are expected.
    #
    # `/completion` cannot be given an image on these builds -- measured, see
    # `normalizer.build_chat_body` -- so a vision model goes to the chat
    # endpoint and everything else keeps the byte-exact ChatML it has always
    # had.
    use_chat = wants_chat_transport(model_spec)
    # How many turns in a row a guard has already ended for this session. It
    # feeds the same escalation a failing tool does -- see
    # `adaptive.effective_loop_depth` for why a cut never maps to the greedy
    # step that a failing tool's second try does.
    guard_cuts = guard_streak.depth(workspace_dir) if guard_streak else 0
    generate_body = _build_upstream_body(client_body, config, use_chat, guard_cuts)
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    # Which transport this turn uses, DERIVED rather than remembered: a
    # completion body carries `prompt`, a chat body carries `messages`. Five
    # features here work by appending to the prompt -- the prefill, the silent
    # retry, the auto-resume, the content continuation and the loop recovery
    # -- and none of them can exist on an endpoint that takes no prompt. Each
    # asks this, so a sixth cannot be written without noticing.
    #
    # **A prompt in another chat format counts as no prompt.** All five append
    # ChatML-and-Qwen markup -- `<think>`, a closed `</think>`, an open
    # `<tool_call>`, a compacted history re-rendered by `format_chatml` -- so on
    # a Gemma 4 prompt (`<|turn>model`, a `<|channel>` thought) they would write
    # text the model was never trained to continue. They take the documented
    # chat-transport behaviour instead: the guards still fire and still say
    # so, and end the turn rather than retrying invisibly.
    prompt_transport = ("prompt" in generate_body
                        and config.prompt_format == "chatml")
    model = client_body.get("model", "unknown")
    # The parser dialect, declared by the registry rather than sniffed out of
    # the model's file name. Computed once here because FOUR parsers are built
    # in this function -- the main generation, the silent retry, the content
    # continuation and the loop recovery -- and a dialect that only three of
    # them knew about is exactly the shape of defect this file keeps recording.
    parser_family = model_spec.family if model_spec is not None else None
    created = int(time.time())

    _log.info("STREAM START model=%s chat_id=%s", model, chat_id)

    # Build set of registered tool names for hallucination filtering
    registered_names: set[str] = set()
    for t in tools:
        fn = (t or {}).get("function") or {}
        n = fn.get("name")
        if n:
            registered_names.add(n)

    if tools:
        state = normalizer.determine_lodestar_state(client_body.get("messages") or [], config)
    else:
        # Non-agentic request (graph builder, title generator...) — LODESTAR
        # machinery stays fully out of the way.
        state = int(LodestarState.READY)
    # v2.4: build the forced-order todo tracker for this state
    from gateway.lodestar import LodestarTodo
    from gateway.lodestar.context_injector import user_requested_skills
    _wants_skills = user_requested_skills(client_body.get("messages") or [])
    lodestar_todo = LodestarTodo.from_messages(
        client_body.get("messages") or [],
        LodestarState(state),
        user_wants_skills=_wants_skills,
        workspace_dir=workspace_dir,
    )
    emitter = OpenAISSEEmitter(chat_id, model, created, tool_names=registered_names, allow_skills=True)
    in_triple, in_single = get_initial_backtick_state(client_body.get("messages") or [])

    # v2.5: assistant prefill (report: "Sockpuppeting"). In states 1-2 the
    # prompt is extended with an open <think> block naming the next required
    # LODESTAR action, so the model mathematically continues the protocol
    # instead of jumping to an unrelated skill/tool. The prefill lives only
    # in the prompt (never echoed to the client); the FSM starts in THINKING.
    prefill_tool_call = False
    start_in_thinking = False
    if (
        getattr(config, "lodestar_prefill", True)
        and prompt_transport
        and state < int(LodestarState.READY)
        and not is_summary
        and workspace_dir
        and tools  # auxiliary requests (title generator etc.) send no tools
    ):
        next_req = lodestar_todo.next_required()
        if next_req is not None:
            _, next_text = next_req
            # v2.5.3: the prefill ends with an OPEN <tool_call> tag. The
            # earlier version ended with an announcement sentence ("I will
            # emit that tool call now") — and the model's strong
            # announce-then-stop habit made it emit EOS as the very first
            # token (audit: tokens_out=1). With the tool-call block already
            # open, the only plausible continuation is <function=...>.
            generate_body["prompt"] += (
                "<think>\nLODESTAR protocol, state "
                f"{state}. Required next action: {next_text}\n"
                "I am emitting exactly this required tool call now.\n"
                "</think>\n\n<tool_call>\n"
            )
            prefill_tool_call = True
            _log.info("LODESTAR PREFILL: state=%d next=%s", state, next_text[:80])
    forced_mode, forced_name = client_params.tool_choice_mode(client_body)
    if prompt_transport and not prefill_tool_call and tools and forced_mode == "function":
        # A named function needs both halves: the prefill opens <tool_call> so
        # a call is the only plausible continuation, and the grammar pins WHICH
        # call -- which the prefill alone cannot do. Everything after the
        # opener is left unconstrained; gateway/grammar.py says why the rest of
        # the call must NOT be, and why this is not parity with the eGPU.
        gbnf = grammar.forced_function(str(forced_name))
        if gbnf:
            generate_body["prompt"] += "<tool_call>\n"
            generate_body["grammar"] = gbnf
            prefill_tool_call = True
            _log.info("TOOL_CHOICE function=%s: prefilled and pinned by grammar",
                      forced_name)

    if prompt_transport and not prefill_tool_call and tools and forced_mode == "required":
        # tool_choice="required" gets the treatment LODESTAR's states 1-2 get,
        # and for the same measured reason: an announcement sentence made the
        # model emit EOS as its very first token (audit: tokens_out=1), while
        # an already-open <tool_call> leaves <function=...> as the only
        # plausible continuation. Text asking for a tool call is what this
        # model ignores; taking the alternative away is what works.
        generate_body["prompt"] += "<tool_call>\n"
        prefill_tool_call = True
        _log.info("TOOL_CHOICE required: prefilled an open tool call")

    if (
        not prefill_tool_call
        and prompt_transport
        and detect_model_family(model) == "qwen3"
        and not is_summary
    ):
        # Auto-prefill <think> for Qwen 3.x reasoning models so thoughts stream as reasoning_content
        generate_body["prompt"] += "<think>\n"
        start_in_thinking = True

    # Everything that shapes the prompt has now happened — injection, pruning,
    # adaptive sampling, prefill. Record it here rather than at request entry:
    # this is the string llama-server sees, and it is the only one worth
    # comparing against what the model then did.
    if audit:
        audit.log_prompt(
            request_id=request_id,
            prompt=generate_body.get("prompt", ""),
            session_id=workspace_dir,
            model=model,
            lodestar_state=state,
            prefill=prefill_tool_call or start_in_thinking,
            sampling=generate_body,
            spill_count=spill_count,
            compaction=compaction,
            dump_dir=(config.log_dir / "prompts") if config.audit_full_prompts else None,
        )

    fsm = create_parser(
        model_name=model,
        tools=tools,
        in_triple_backticks=in_triple,
        in_single_backtick=in_single,
        start_in_thinking=start_in_thinking,
        start_in_tool_call=prefill_tool_call,
        family=parser_family,
    )

    final_usage: dict | None = None

    # The audit keeps the real numbers whatever the client asked for; only what
    # goes out on the wire is gated. tokens_in/tokens_out in gateway_events
    # are read by check_prefill_health.py and session_report.py, and a request
    # that opted out of usage must not make the log lie about what it cost.
    def client_usage(usage: dict | None) -> dict | None:
        return usage if client_params.wants_usage(client_body) else None
    # Per-request tally of what the enforcer did, summarised into the response
    # audit event. "synthesized" is the one that matters: it counts the calls
    # the gateway had to make because the model would not.
    tool_stats: dict[str, Any] = {}
    # The per-turn call budget travels in `stats` rather than as a parameter,
    # because `stats` already reaches every `_handle_fsm_event` call site and a
    # new argument would be a new thing to forget at one of seven of them --
    # which is the shape this file has paid for three times. It also SURVIVES
    # the parser rebuild a continuation does: `fsm.tool_call_count` restarts at
    # zero there, which is exactly how 22 calls accumulated on 2026-09-14 with
    # nothing counting them.
    tool_stats["turn_call_budget"] = config.max_tool_calls_per_turn
    # How much the gateway has already done for the model in this cycle. Past
    # config.max_synthesis_per_cycle it stops and says so.
    synthesis_before = synthesis_count(client_body.get("messages") or [])
    # Subagents skip LODESTAR, so every limit derived from cycle state misses
    # them entirely. This one is counted from their own conversation.
    subagent_hard_stop = False
    subagent_report_floor = 0
    if config.max_subagent_turns and not is_summary:
        from gateway.lodestar import subagent as _sub
        _sub_msgs = client_body.get("messages") or []
        if _sub.is_subagent_request(None, _sub_msgs):
            subagent_report_floor = config.subagent_min_report_chars
            _stage, _sub_turns = _sub.budget_state(
                _sub_msgs, config.max_subagent_turns, config.subagent_turn_grace)
            subagent_hard_stop = _stage == _sub.HARD
            if _stage is not None:
                _log.warning(
                    "SUBAGENT BUDGET: %s — %d turns of %d, tool calls %s",
                    _stage, _sub_turns, config.max_subagent_turns,
                    "stopped" if subagent_hard_stop else "still running")

    # Until 2026-09-01 this only wrote a log line: a configured limit that
    # stopped nothing, which is worse than no limit because it reads as cover.
    # Now it behaves like the subagent budget — asked first, then enforced —
    # with one difference that makes the hard stop safe: the turn ends with a
    # plain answer to the USER, and a fresh cycle resets the count. Nothing is
    # lost, only interrupted.
    #
    # **That reset is LODESTAR's, which is why this asks whether LODESTAR is
    # running at all.** `effective_messages` slices the history at a tool call
    # that writes a closing Status into active-context.md, and at nothing
    # else. With `lodestar_enabled` off the model is never asked for that
    # write, so no boundary is ever laid down and the count stops meaning
    # "turns spent on this task" and starts meaning "assistant turns in this
    # conversation, ever" -- a per-session ceiling wearing a per-task limit's
    # name. Measured 2026-09-21 on a rig that had served without the protocol
    # since 2026-09-10: one client session climbed 60 -> 81 and three `bash`
    # calls were suppressed across two turns, minutes apart.
    #
    # The silence on top of it is the half that cost the afternoon. BOTH
    # notices -- the soft close-out ask and the [SYSTEM GUARD] line that tells
    # the person the stop is recoverable -- come from `context_injector`,
    # which `_inject_context_files` skips when LODESTAR is off. So the
    # ENFORCEMENT ran while the EXPLANATION did not: the calls vanished with
    # no word to the model or to the user, and it read as the model giving up.
    # Asking the same question here is what puts the two on one switch;
    # `TestTheBudgetAndItsNoticeShareOneSwitch` reads this file as source,
    # because a conjunct is silently droppable.
    cycle_hard_stop = False
    if config.max_cycle_turns and config.lodestar_enabled and not is_summary:
        from gateway.lodestar.state import CYCLE_HARD, cycle_budget_state

        _cycle_stage, turns_so_far = cycle_budget_state(
            client_body.get("messages") or [],
            config.max_cycle_turns, config.cycle_turn_grace)
        cycle_hard_stop = _cycle_stage == CYCLE_HARD
        if _cycle_stage is not None:
            _log.warning(
                "LODESTAR LIMIT: cycle has run %d assistant turns (budget %d) "
                "chat_id=%s — tool calls %s",
                turns_so_far, config.max_cycle_turns, chat_id,
                "stopped" if cycle_hard_stop else "still running",
            )

    # Wait for upstream concurrency slot
    sem_acquired = False
    worker = None
    # The live httpx response, published by the reader task so the shutdown
    # path can sever the socket even when the reader will not wind down.
    upstream_holder: dict[str, Any] = {}
    stop_upstream: asyncio.Event | None = None
    # Everything a retry has to clear lives in one object, so adding a guard
    # means adding a field rather than remembering a `nonlocal` list.
    st = TurnState()
    # The three reasoning guards and their throttles, for THIS attempt.
    # Outside TurnState because it needs config; reset beside it, because a
    # new generation must be judged on its own text.
    watch = runaway.RunawayWatch(
        config, is_summary,
        on_cut=(lambda g: guard_streak.record(workspace_dir, request_id, g))
        if guard_streak else None)
    stream_start = time.time()
    # Three clocks, because duration_ms alone cannot say why a turn was slow:
    # time spent queued for the single slot, time until upstream produces its
    # first token (prefill), and the total.
    queue_start = stream_start
    queue_wait_ms: int | None = None

    try:
        if semaphore:
            try:
                # Both halves of INTERACTIVE: a turn with tools has a human
                # watching its loop, and a compaction has the client blocked
                # mid-session. Everything else yields to them.
                async for keepalive in _acquire_slot(
                        semaphore, config,
                        request_priority(has_tools=bool(tools),
                                         is_summary=is_summary)):
                    yield keepalive
            except _SlotTimeout:
                # The non-streaming twin has logged this since it was written;
                # this path did not, so a queue long enough to time out looked
                # in the audit log like a request that simply stopped
                # (O-2026-01).
                if audit:
                    audit.log_error(request_id, "upstream_busy",
                                    "concurrency slot timeout")
                yield emitter.error_chunk("upstream busy - request queued for too long")
                yield emitter.done()
                return
            sem_acquired = True

        # The card, AFTER the slot: holding it is what guarantees nothing is
        # generating on the engine about to be unloaded. A cold 13.6 GB load
        # takes minutes, so the wait yields keep-alives rather than going
        # silent on a client that is counting seconds.
        if model_spec is not None and model_spec.is_local:
            _swap: dict[str, Any] = {}
            try:
                async for keepalive in _hand_over_card_streaming(
                        model_spec, config, _swap):
                    yield keepalive
                if _swap.get("url"):
                    upstream_override = _swap["url"]
            except engines.EngineError as exc:
                # Content, never error_chunk. A failed hand-over ends the TURN;
                # an error object carries no `choices` and ends the SESSION --
                # OpenCode abandons the agent loop 14 ms later. The model also
                # needs to read this next turn, and it cannot read an error.
                _log.error("ENGINE SWAP FAILED chat_id=%s: %s", chat_id, exc)
                if audit:
                    audit.log_error(request_id, "engine_swap_failed", str(exc))
                yield emitter.role_chunk()
                yield emitter.content_chunk(
                    f"[SYSTEM GUARD] The engine for `{model_spec.id}` could not "
                    f"be started, so this turn did not reach a model.\n\n{exc}")
                yield emitter.finish_chunk("stop")
                yield emitter.done()
                return

        # Generation starts now: reset stream_start so queue wait time does not
        # count against stream_max_duration.
        stream_start = time.time()
        queue_wait_ms = int((stream_start - queue_start) * 1000)

        yield emitter.role_chunk()

        q: asyncio.Queue = asyncio.Queue()
        # `stop_upstream` is what makes a cancelled turn actually stop costing
        # slot time. See _shutdown_upstream_worker for why cancel() alone is
        # not enough.
        stop_upstream = asyncio.Event()
        worker = asyncio.create_task(
            _fetch_upstream_to_queue(q, upstream, generate_body, headers, chat_id,
                                     upstream_override=upstream_override,
                                     stop=stop_upstream, resp_holder=upstream_holder))
        pacing = _StreamPacing()
        base_prompt = generate_body.get("prompt", "")

        async def _retry_out_of_the_loop(reason: str) -> None:
            """Abandon this generation and start another, invisibly.

            The alternative — cut the turn and explain — leaves the user
            answering for a problem the gateway diagnosed itself. Here the
            client sees one continuous response instead.

            The retry prompt closes `</think>` before the model resumes, so it
            physically cannot go back to deliberating. Text telling a looping
            model to stop is exactly what it has already been ignoring; taking
            the block away is the part that works. And because the prompt is an
            APPEND to the original, llama-server reuses the cached prefix and
            prefills only the correction.

            Reasoning already streamed stays streamed — the client has it, and
            rewriting history would be worse than a long "Thought". What the
            user is spared is the dead turn.
            """
            # Only what is REBOUND needs to be here. Everything the retry has
            # to CLEAR is a field of `st`, which is mutated, not rebound — so a
            # new guard cannot be forgotten on this list any more.
            nonlocal worker, q, stop_upstream, upstream_holder, fsm
            nonlocal generate_body

            st.reasoning_retries += 1
            _log.warning(
                "REASONING RETRY %d/%d (%s): abandoning the generation and "
                "steering out of the loop chat_id=%s",
                st.reasoning_retries, config.reasoning_retry_max, reason, chat_id)
            if audit:
                audit.log_error(
                    request_id, "reasoning_retry",
                    f"attempt {st.reasoning_retries} ({reason}) after "
                    f"{len(emitter.reasoning_text)} chars of reasoning")

            await _shutdown_upstream_worker(
                worker, stop_upstream, upstream_holder, chat_id,
                grace=config.upstream_close_grace)

            try:
                correction = prompts.get("reasoning_retry_prefill")
            except KeyError:
                correction = "I will stop deliberating and act now."
            generate_body = {**generate_body,
                             "prompt": f"{base_prompt}{correction}\n</think>\n\n"}

            # A fresh parser, and NOT in thinking: the block is closed in the
            # prompt above, so the next token is content or a tool call.
            fsm = create_parser(
                model_name=model, tools=tools,
                in_triple_backticks=in_triple, in_single_backtick=in_single,
                start_in_thinking=False, start_in_tool_call=False,
                family=parser_family)
            emitter.reset_reasoning()
            emitter.reset_tool_call_state()
            # One call instead of eight assignments — and `ttft_ms`, which was
            # never on the old list, is cleared with the rest (O-2026-04).
            st.reset_for_retry()
            watch.reset()

            q = asyncio.Queue()
            stop_upstream = asyncio.Event()
            upstream_holder = {}
            worker = asyncio.create_task(
                _fetch_upstream_to_queue(
                    q, upstream, generate_body, headers, chat_id,
                    upstream_override=upstream_override,
                    stop=stop_upstream, resp_holder=upstream_holder))
            pacing.touch()

        # One reader for all four stream loops (see _upstream_events).
        # `lambda: q` and not `q`: the silent retry rebinds the queue
        # mid-loop, and a captured value would keep reading the dead one.
        async for _ev, _val in _upstream_events(
                    # by reference on purpose: the retry rebinds q
                    lambda: q, config, pacing):  # noqa: B023
            if _ev == "tick":
                # Safety valve — Stream Guard Timeout
                if _stream_time_exhausted(stream_start, config):
                    _log.warning(
                        "STREAM MAX DURATION (%ds) exceeded chat_id=%s request_id=%s — triggering stream guard",
                        config.stream_max_duration, chat_id, request_id,
                    )
                    # Flush the FSM to output closing characters first (to prevent JSON Parse error)
                    for fsm_event in fsm.finish("length"):
                        chunks = _handle_fsm_event(
                            fsm_event, emitter, tools, is_summary,
                            workspace_dir=workspace_dir, lodestar_state=state,
                            registered_names=frozenset(registered_names),
                            lodestar_todo=lodestar_todo, client_body=client_body,
                            audit=audit, request_id=request_id, session_id=workspace_dir,
                            stats=tool_stats, max_synthesis=config.max_synthesis_per_cycle,
                            synthesis_so_far=synthesis_before, budget=budget,
                            destructive_action=config.destructive_action,
                            subagent_hard_stop=subagent_hard_stop,
                            cycle_hard_stop=cycle_hard_stop,
                        )
                        for c in chunks:
                            yield c

                    # Emit self-correcting system guard directive so the model divides the task in the next step
                    try:
                        guard_msg = prompts.get("stream_timeout_guard")
                    except KeyError:
                        guard_msg = ""
                    if guard_msg:
                        yield emitter.content_chunk(f"\n\n{guard_msg}\n")

                    if audit:
                        audit.log_error(
                            request_id, "stream_timeout_guard",
                            f"Stream exceeded safety limit ({config.stream_max_duration}s), guard message emitted",
                        )

                    yield emitter.finish_chunk("length")
                    yield emitter.done()
                    return

                continue
            if _ev == "heartbeat":
                yield emitter.heartbeat()
                pacing.touch()
                continue
            if _ev == "error":
                err_msg = _val.decode("utf-8", errors="ignore").strip()
                if audit:
                    audit.log_error(request_id, "upstream_stream_error",
                                    err_msg[:500])
                try:
                    err_json = json.loads(err_msg)
                    if not isinstance(err_json, dict) or "error" not in err_json:
                        err_json = {"error": {"message": err_msg}}
                except Exception:
                    err_json = {"error": {"message": err_msg}}
                yield b"data: " + json.dumps(err_json).encode("utf-8") + b"\n\n"
                yield emitter.done()
                return

            chunk = _val
            if chunk.get("stop"):
                _finish_on_upstream_stop(st, chunk, tool_stats)
                final_usage = _merge_final_usage(final_usage, chunk)
                break

            piece = chunk.get("content", "")
            if st.ttft_ms is None and piece:
                # First token out of upstream: everything before it was
                # prompt processing. Large here with a large tokens_in
                # means the KV prefix was missed.
                st.ttft_ms = int((time.time() - stream_start) * 1000)
            if not piece:
                continue
            st.raw_tail = (st.raw_tail + piece)[-800:]

            # --- Feed piece to FSM and emit SSE ---
            for fsm_event in fsm.feed(piece):
                chunks = _handle_fsm_event(fsm_event, emitter, tools, is_summary, workspace_dir=workspace_dir, lodestar_state=state, registered_names=frozenset(registered_names), lodestar_todo=lodestar_todo, client_body=client_body, audit=audit, request_id=request_id, session_id=workspace_dir, stats=tool_stats, max_synthesis=config.max_synthesis_per_cycle, synthesis_so_far=synthesis_before, budget=budget, destructive_action=config.destructive_action,
                        subagent_hard_stop=subagent_hard_stop,
                        cycle_hard_stop=cycle_hard_stop)
                for c in chunks:
                    yield c
                    pacing.touch()
                if fsm_event.type == EventType.TOOL_CALL_END and not is_summary:
                    if not emitter.is_suppressed(fsm_event.data.get("index", -1)):
                        st.finish_reason = "tool_calls"

            if tool_stats.get("turn_budget_chunk_sent") or tool_stats.get("turn_loop_stop"):
                _reason = (
                    "turn call budget reached"
                    if tool_stats.get("turn_budget_chunk_sent")
                    else "intra-turn tool loop detected"
                )
                _log.warning(
                    "STREAM: %s — stopping upstream generation cleanly", _reason
                )
                if emitter.has_emitted_tool_calls:
                    st.finish_reason = "tool_calls"
                else:
                    st.finish_reason = "stop"
                stop_upstream.set()
                break

            # --- The three reasoning guards, asked as one question ---
            # The policy behind them -- the config knobs, `not is_summary`,
            # `fsm.in_thinking`, and the per-detector throttles -- lives in
            # gateway/runaway.py, because policy written out by hand in each
            # caller is policy that will be missing from one. It WAS missing
            # from one: _non_stream_completion had none of it, and seven turns
            # ran to the n_predict ceiling at fourteen minutes apiece with
            # nothing watching (measured 2026-09-12).
            #
            # Only the ACTION stays here, and it is the same for all three:
            # hand it to the silent retry while there is one, then cut and say
            # why. `fsm.in_thinking` is what keeps a started tool call safe --
            # cutting mid-<tool_call> truncates the arguments and hands the
            # client a JSON parse error, the one failure this codebase never
            # trades for another.
            verdict = watch.check_stream(emitter.reasoning_text, fsm.in_thinking,
                                              emitter.content_text,
                                              fsm.tool_call_count > 0)
            if verdict is not None:
                st.reasoning_guard_tripped = True
                st.runaway_verdict = verdict
                if verdict.guard == runaway.CODE:
                    _lines, _first = verdict.detail
                    _log.warning(
                        "REASONING CODE: %d lines of source written inside "
                        "<think> after %d chars — cutting the turn. First "
                        "line: %r chat_id=%s",
                        _lines, verdict.chars, _first[:80], chat_id)
                    detail = f"{_lines} code lines in reasoning after {verdict.chars} chars"
                elif verdict.guard == runaway.LOOP:
                    line, count = verdict.detail
                    _log.warning(
                        "REASONING LOOP: %r written %d times in %d chars — "
                        "cutting the turn chat_id=%s request_id=%s",
                        line[:80], count, verdict.chars, chat_id, request_id)
                    detail = (f"repeated {count}x after {verdict.chars} chars: "
                              f"{line[:120]}")
                else:
                    _log.warning(
                        "REASONING BUDGET EXCEEDED (%d chars, limit %d) chat_id=%s "
                        "request_id=%s — cutting the turn. Reasoning tail: %r",
                        verdict.chars, config.reasoning_budget_chars,
                        chat_id, request_id, emitter.reasoning_text[-300:])
                    detail = (f"reasoning reached {verdict.chars} chars (limit "
                              f"{config.reasoning_budget_chars}) without leaving <think>")
                if audit:
                    audit.log_error(
                        request_id, f"reasoning_{verdict.guard}_guard", detail)
                if (st.reasoning_retries < config.reasoning_retry_max
                        and prompt_transport):
                    await _retry_out_of_the_loop(f"reasoning {verdict.guard}")
                    continue
                st.finish_reason = "length"
                break

        # --- Final flush ---
        for fsm_event in fsm.finish(st.finish_reason):
            _cut_off = _flush_truncated_call(fsm_event, st, emitter)
            if _cut_off and emitter.is_suppressed(fsm_event.data.get("index", -1)):
                continue    # held back and cut off: never seen, never judged
            chunks = _handle_fsm_event(fsm_event, emitter, tools, is_summary, workspace_dir=workspace_dir, lodestar_state=state, registered_names=frozenset(registered_names), lodestar_todo=lodestar_todo, client_body=client_body, audit=audit, request_id=request_id, session_id=workspace_dir, stats=tool_stats, max_synthesis=config.max_synthesis_per_cycle, synthesis_so_far=synthesis_before, budget=budget, destructive_action=config.destructive_action,
                        subagent_hard_stop=subagent_hard_stop,
                        cycle_hard_stop=cycle_hard_stop)
            for c in chunks:
                yield c
            if fsm_event.type == EventType.TOOL_CALL_END and not is_summary and not _cut_off:
                if not emitter.is_suppressed(fsm_event.data.get("index", -1)):
                    st.finish_reason = "tool_calls"
        if st.truncated_call_streamed:
            st.finish_reason = "length"

        # The guard fired: say so and end the turn. Deliberately placed ahead
        # of the enforcer blocks below — the stubborn guide and the Phase D
        # nudge both react to "no tool calls in this turn", and this turn had
        # none because the gateway cut it, not because the model refused.
        # Synthesizing a call on top of a truncation would bury the one
        # message the model needs to read.
        if st.reasoning_guard_tripped:
            try:
                # Order matters: code-in-reasoning is the most specific
                # diagnosis and the one the model can act on directly.
                _v = st.runaway_verdict
                if _v is not None and _v.guard == runaway.CODE:
                    _lines, _first = _v.detail
                    guard_msg = prompts.get("reasoning_code_guard",
                                            lines=_lines, first_line=_first)
                elif _v is not None and _v.guard == runaway.LOOP:
                    _line, _count = _v.detail
                    guard_msg = prompts.get("reasoning_loop_guard",
                                            count=_count, line=_line[:200])
                else:
                    guard_msg = prompts.get("reasoning_budget_guard")
            except KeyError:
                guard_msg = ""
            if guard_msg:
                yield emitter.content_chunk(f"\n\n{guard_msg}\n")
            # The turn the gateway cut for being expensive was the one turn
            # with no `response` event, so its cost was the one cost the audit
            # log could not report (O-2026-01). The token counts are whatever
            # upstream managed to send: a cut turn never receives its usage
            # block, and the paired guard `error` event carries the character
            # counts that ARE known.
            if audit:
                audit.log_response(
                    request_id=request_id,
                    finish_reason="length",
                    tokens_in=(final_usage or {}).get("prompt_tokens", 0),
                    tokens_out=(final_usage or {}).get("completion_tokens", 0),
                    duration_ms=int((time.time() - stream_start) * 1000),
                    ttft_ms=st.ttft_ms,
                    cached_tokens=cached_tokens_of(final_usage),
                    queue_wait_ms=queue_wait_ms,
                    tool_call_count=fsm.tool_call_count,
                    session_id=workspace_dir,
                    synthesized_count=tool_stats.get("synthesized", 0),
                )
            yield emitter.finish_chunk("length", client_usage(final_usage))
            yield emitter.done()
            return

        # v2.4.7: stubborn guide — if the model made no tool calls and
        # there's a required LODESTAR action pending, SYNTHESIZE the
        # call on the model's behalf. The model hallucinates the intent
        # ("I'll read active-context.md") but doesn't actually emit the
        # tool call — it just responds with text. The bramka must
        # force the action through so the chat doesn't end with no
        # work done. The client sees a real tool call, executes it,
        # the result comes back, the model continues.
        if (
            not is_summary
            and tools  # skip auxiliary requests (title generator etc.)
            and fsm.tool_call_count == 0
            and lodestar_todo is not None
            and st.finish_reason != "tool_calls"
        ):
            next_req = lodestar_todo.next_required()
            if next_req is not None:
                next_key, _ = next_req
                stub_chunks = _synthesize_stubborn_call(
                    next_key=next_key,
                    workspace_dir=workspace_dir,
                    tools=tools,
                    emitter=emitter,
                    client_body=client_body,
                )
                if stub_chunks:
                    _log.info(
                        "LODESTAR ENFORCER: stubborn guide — model made no tool calls, "
                        "synthesizing %s for state %d. Model output tail: %r",
                        next_key, state, st.raw_tail[-400:],
                    )
                    tool_stats["synthesized"] = tool_stats.get("synthesized", 0) + 1
                    if budget is not None:
                        budget.record(workspace_dir)
                    if audit:
                        audit.log_tool_call(
                            request_id=request_id, session_id=workspace_dir,
                            tool_name=next_key, parse_status="synthesized",
                            blocked_reason="stubborn_guide", lodestar_state=state,
                            arguments=_args_from_chunks(stub_chunks))
                    for c in stub_chunks:
                        yield c
                    st.finish_reason = "tool_calls"
                else:
                    # Couldn't synthesize. Log the model's raw output tail so
                    # the stall cause is visible in the log.
                    _log.warning(
                        "LODESTAR ENFORCER: stubborn guide — no synthesized call "
                        "available for %s, sending noop. Model output tail: %r",
                        next_key, st.raw_tail[-400:],
                    )

        # v4.1: Phase D nudge — the model ended an OPEN task (Goal committed,
        # no closing write) with a plain-text answer. Remind it via a
        # synthetic echo call to save the lesson and close active-context.
        if (
            not is_summary
            and tools
            and workspace_dir
            and state >= int(LodestarState.READY)
            and fsm.tool_call_count == 0
            and st.finish_reason != "tool_calls"
        ):
            nudge = _synthesize_phase_d_nudge(client_body, tools, emitter,
                                              st.raw_tail, workspace_dir)
            if nudge:
                _log.info(
                    "LODESTAR ENFORCER: Phase D nudge — open task ended with "
                    "text-only turn; reminding model to save lesson and close "
                    "active-context. Output tail: %r", st.raw_tail[-200:],
                )
                tool_stats["synthesized"] = tool_stats.get("synthesized", 0) + 1
                if budget is not None:
                    budget.record(workspace_dir)
                if audit:
                    audit.log_tool_call(
                        request_id=request_id, session_id=workspace_dir,
                        tool_name="bash", parse_status="synthesized",
                        blocked_reason="phase_d_nudge", lodestar_state=state,
                        arguments=_args_from_chunks(nudge))
                for c in nudge:
                    yield c
                st.finish_reason = "tool_calls"

        # Auto-closed tool calls at stream end → finish_reason=length
        if st.finish_reason != "tool_calls" and fsm.tool_call_count > 0:
            # FSM auto-closed a tool call due to truncation
            pass

        # In-stream auto-resume: if model exhausted its generation in <think> without emitting content or tools,
        # continue generation within the SAME HTTP stream instead of ending the turn.
        while (
            not is_summary
            and st.finish_reason != "tool_calls"
            and not emitter.has_emitted_content
            and not emitter.has_emitted_tool_calls
            and emitter.reasoning_text
            and (st.finish_reason == "length" or len(emitter.reasoning_text) >= 400)
            and st.reasoning_retries < config.reasoning_retry_max
            and prompt_transport
        ):
            st.reasoning_retries += 1
            _log.warning(
                "STREAM: in-stream auto-resume (%d/%d): model completed in <think> without content/tools "
                "(%d chars reasoning). Steering into action chat_id=%s request_id=%s",
                st.reasoning_retries, config.reasoning_retry_max, len(emitter.reasoning_text),
                chat_id, request_id,
            )
            if audit:
                audit.log_error(
                    request_id, "instream_auto_resume",
                    f"attempt {st.reasoning_retries}/{config.reasoning_retry_max}: auto-resuming after "
                    f"{len(emitter.reasoning_text)} chars of reasoning with 0 content/tools",
                )

            await _shutdown_upstream_worker(
                worker, stop_upstream, upstream_holder, chat_id,
                grace=config.upstream_close_grace,
            )

            continuation_prompt = _build_continuation_prompt(
                base_prompt, emitter.reasoning_text, max_tokens=config.get_prune_limit(),
            )
            generate_body = {**generate_body, "prompt": continuation_prompt}

            fsm = create_parser(
                model_name=model, tools=tools,
                in_triple_backticks=in_triple, in_single_backtick=in_single,
                start_in_thinking=False, start_in_tool_call=False,
                family=parser_family,
            )
            st.reset_for_retry()
            watch.reset()

            q = asyncio.Queue()
            stop_upstream = asyncio.Event()
            upstream_holder = {}
            worker = asyncio.create_task(
                _fetch_upstream_to_queue(
                    q, upstream, generate_body, headers, chat_id,
                    upstream_override=upstream_override,
                    stop=stop_upstream, resp_holder=upstream_holder,
                )
            )
            pacing.touch()

            # Read continuation stream until completion
            # W-2026-03 applies to this reader too, and it had NEITHER
            # guard. Extracting _upstream_events is what made that
            # visible: the plumbing existed four times, the guards twice.
            guard_prompt: str | None = None
            async for _ev, _val in _upstream_events(
                    # by reference on purpose: the retry rebinds q
                    lambda: q, config, pacing):  # noqa: B023
                if _ev == "tick":
                    if _stream_time_exhausted(stream_start, config):
                        _log.warning(
                            "STREAM MAX DURATION (%ds) exceeded during auto-resume "
                            "chat_id=%s request_id=%s", config.stream_max_duration,
                            chat_id, request_id)
                        if audit:
                            audit.log_error(request_id, "stream_timeout_guard",
                                            "auto-resume exceeded the safety limit")
                        guard_prompt = "stream_timeout_guard"
                        st.finish_reason = "length"
                        break
                    if (_v := watch.check_stream(emitter.reasoning_text, fsm.in_thinking,
                                              emitter.content_text,
                                              fsm.tool_call_count > 0)) is not None:
                        _log.warning(
                            "RUNAWAY (%s) after %d chars during auto-resume "
                            "chat_id=%s request_id=%s", _v.guard, _v.chars,
                            chat_id, request_id)
                        if audit:
                            audit.log_error(request_id, f"reasoning_{_v.guard}_guard",
                                            f"auto-resume: {_v.chars} chars")
                        guard_prompt = _RUNAWAY_PROMPTS[_v.guard]
                        st.finish_reason = "length"
                        break
                    continue
                if _ev == "heartbeat":
                    yield emitter.heartbeat()
                    pacing.touch()
                    continue
                if _ev == "error":
                    err_msg = _val.decode("utf-8", errors="ignore").strip()
                    if audit:
                        audit.log_error(request_id, "upstream_stream_error", err_msg[:500])
                    break
                chunk = _val
                if chunk.get("stop"):
                    _finish_on_upstream_stop(st, chunk, tool_stats)
                    final_usage = _merge_final_usage(final_usage, chunk)
                    break

                piece = chunk.get("content", "")
                if not piece:
                    continue
                st.raw_tail = (st.raw_tail + piece)[-800:]

                for fsm_event in fsm.feed(piece):
                    chunks = _handle_fsm_event(
                        fsm_event, emitter, tools, is_summary,
                        workspace_dir=workspace_dir, lodestar_state=state,
                        registered_names=frozenset(registered_names),
                        lodestar_todo=lodestar_todo, client_body=client_body,
                        audit=audit, request_id=request_id, session_id=workspace_dir,
                        stats=tool_stats, max_synthesis=config.max_synthesis_per_cycle,
                        synthesis_so_far=synthesis_before, budget=budget,
                        destructive_action=config.destructive_action,
                        subagent_hard_stop=subagent_hard_stop,
                        cycle_hard_stop=cycle_hard_stop,
                    )
                    for c in chunks:
                        yield c
                        pacing.touch()
                    if fsm_event.type == EventType.TOOL_CALL_END and not is_summary:
                        if not emitter.is_suppressed(fsm_event.data.get("index", -1)):
                            st.finish_reason = "tool_calls"

                if tool_stats.get("turn_budget_chunk_sent") or tool_stats.get("turn_loop_stop"):
                    _reason = (
                        "turn call budget reached"
                        if tool_stats.get("turn_budget_chunk_sent")
                        else "intra-turn tool loop detected"
                    )
                    _log.warning(
                        "STREAM: %s — stopping upstream generation cleanly", _reason
                    )
                    if emitter.has_emitted_tool_calls:
                        st.finish_reason = "tool_calls"
                    else:
                        st.finish_reason = "stop"
                    stop_upstream.set()
                    break

            # Flush parser after continuation
            for fsm_event in fsm.finish(st.finish_reason):
                _cut_off = _flush_truncated_call(fsm_event, st, emitter)
                if _cut_off and emitter.is_suppressed(fsm_event.data.get("index", -1)):
                    continue    # held back and cut off: never seen, never judged
                chunks = _handle_fsm_event(
                    fsm_event, emitter, tools, is_summary,
                    workspace_dir=workspace_dir, lodestar_state=state,
                    registered_names=frozenset(registered_names),
                    lodestar_todo=lodestar_todo, client_body=client_body,
                    audit=audit, request_id=request_id, session_id=workspace_dir,
                    stats=tool_stats, max_synthesis=config.max_synthesis_per_cycle,
                    synthesis_so_far=synthesis_before, budget=budget,
                    destructive_action=config.destructive_action,
                    subagent_hard_stop=subagent_hard_stop,
                    cycle_hard_stop=cycle_hard_stop,
                )
                for c in chunks:
                    yield c
                if fsm_event.type == EventType.TOOL_CALL_END and not is_summary and not _cut_off:
                    if not emitter.is_suppressed(fsm_event.data.get("index", -1)):
                        st.finish_reason = "tool_calls"
            if st.truncated_call_streamed:
                st.finish_reason = "length"

            # After the flush, never before it: the flush closes a half-written
            # tool call, and a guard paragraph inside one is the JSON parse
            # error this codebase refuses to trade for anything.
            if guard_prompt:
                try:
                    guard_msg = prompts.get(guard_prompt)
                except KeyError:
                    guard_msg = ""
                if guard_msg:
                    yield emitter.content_chunk(f"\n\n{guard_msg}\n")
                break

        def _answer_stuck_in_think() -> bool:
            """Did the turn end with the answer left in the discarded channel?

            The subagent floor catches this for a subagent, where a short last
            turn is ALWAYS a failure -- the client ends it the moment a turn
            carries no tool call, and that text becomes the report its parent
            reads. A main session is different: "Gotowe." after a moment's
            thought is a correct answer, and the same rule there would nag
            constantly. Measured 2026-09-12 over 736 real turns that ended
            with no tool call, 301 were exactly that.

            So the trigger is the CONJUNCTION -- substantial reasoning AND
            almost no content -- which isolates 11 of those 736, eight of them
            with no content at all.

            Continuing beats extracting the conclusion out of <think>. This
            project already decided, at the code-in-reasoning guard, that the
            gateway must not fabricate work out of a reasoning block: it does
            not know where the conclusion ends, and an extraction hands the
            reader the model's THINKING voice ("Let me check...") rather than
            an answer. Asking costs one short generation against a warm
            prefix; the model is the only party that knows what it concluded.
            """
            return bool(
                config.answer_rescue_min_reasoning
                and st.answer_rescues < config.answer_rescue_max
                and st.finish_reason != "tool_calls"
                and len(emitter.reasoning_text) >= config.answer_rescue_min_reasoning
                and emitter.emitted_content_chars < config.subagent_min_report_chars
            )

        # In-stream content continuation. Two cases, one mechanism: a turn cut
        # off mid-content by the token limit, and a turn that ended with its
        # answer still inside <think>. Both resume with a compacted history and
        # a CLOSED reasoning block; only the trailing nudge differs.
        while (
            not is_summary
            and prompt_transport
            and not emitter.has_emitted_tool_calls
            and fsm.tool_call_count == 0
            and (
                (st.finish_reason == "length"
                 and emitter.has_emitted_content
                 and st.content_continues < config.content_continue_max)
                or _answer_stuck_in_think()
            )
        ):
            rescue = _answer_stuck_in_think()
            if rescue:
                st.answer_rescues += 1
                _log.info(
                    "STREAM: answer rescue (%d/%d): %d chars of reasoning, %d "
                    "of content, no tool call. Asking for the answer itself "
                    "chat_id=%s request_id=%s",
                    st.answer_rescues, config.answer_rescue_max,
                    len(emitter.reasoning_text), emitter.emitted_content_chars,
                    chat_id, request_id)
                if audit:
                    audit.log_error(
                        request_id, "answer_rescue",
                        f"{len(emitter.reasoning_text)} chars of reasoning, "
                        f"{emitter.emitted_content_chars} of content")
            else:
                st.content_continues += 1
            _log.info(
                "STREAM: in-stream content continuation (%d/%d): model cut off at length "
                "(%d chars content). Resuming with compacted history chat_id=%s request_id=%s",
                st.content_continues, config.content_continue_max, len(emitter.content_text),
                chat_id, request_id,
            )
            if audit:
                audit.log_error(
                    request_id, "instream_content_continue",
                    f"attempt {st.content_continues}/{config.content_continue_max}: resuming after "
                    f"{len(emitter.content_text)} chars of content cut off by length",
                )

            await _shutdown_upstream_worker(
                worker, stop_upstream, upstream_holder, chat_id,
                grace=config.upstream_close_grace,
            )

            continuation_prompt = _build_content_continuation_prompt(
                client_body.get("messages") or [],
                content_text=emitter.content_text,
                reasoning_text=emitter.reasoning_text,
                max_tokens=config.get_prune_limit(),
                nudge=prompts.get("answer_rescue") if rescue else None,
            )
            generate_body = {**generate_body, "prompt": continuation_prompt}

            cont_in_triple, cont_in_single = _check_text_backtick_state(emitter.content_text)
            fsm = create_parser(
                model_name=model, tools=tools,
                in_triple_backticks=cont_in_triple, in_single_backtick=cont_in_single,
                start_in_thinking=False, start_in_tool_call=False,
                family=parser_family,
            )
            st.reset_for_retry()
            watch.reset()

            q = asyncio.Queue()
            stop_upstream = asyncio.Event()
            upstream_holder = {}
            worker = asyncio.create_task(
                _fetch_upstream_to_queue(
                    q, upstream, generate_body, headers, chat_id,
                    upstream_override=upstream_override,
                    stop=stop_upstream, resp_holder=upstream_holder,
                )
            )
            pacing.touch()

            # Read continuation stream until completion
            # W-2026-03 applies to this reader too, and it had NEITHER
            # guard. Extracting _upstream_events is what made that
            # visible: the plumbing existed four times, the guards twice.
            guard_prompt = None
            async for _ev, _val in _upstream_events(
                    # by reference on purpose: the retry rebinds q
                    lambda: q, config, pacing):  # noqa: B023
                if _ev == "tick":
                    if _stream_time_exhausted(stream_start, config):
                        _log.warning(
                            "STREAM MAX DURATION (%ds) exceeded during content continuation "
                            "chat_id=%s request_id=%s", config.stream_max_duration,
                            chat_id, request_id)
                        if audit:
                            audit.log_error(request_id, "stream_timeout_guard",
                                            "content continuation exceeded the safety limit")
                        guard_prompt = "stream_timeout_guard"
                        st.finish_reason = "length"
                        break
                    if (_v := watch.check_stream(emitter.reasoning_text, fsm.in_thinking,
                                              emitter.content_text,
                                              fsm.tool_call_count > 0)) is not None:
                        _log.warning(
                            "RUNAWAY (%s) after %d chars during content continuation "
                            "chat_id=%s request_id=%s", _v.guard, _v.chars,
                            chat_id, request_id)
                        if audit:
                            # The guard that actually tripped, not the one this
                            # line used to name. `Verdict.guard` is CODE, LOOP
                            # or BUDGET, and hardcoding BUDGET here meant a loop
                            # or a code trip in a content continuation was
                            # recorded as a budget trip -- in the JSONL and in
                            # `llmgw_errors_total{type=...}`, which reads this
                            # exact field. Two of the three readers already
                            # passed it through; this was the third. O-2026-06.
                            audit.log_error(
                                request_id, f"reasoning_{_v.guard}_guard",
                                f"content continuation: {_v.chars} chars")
                        guard_prompt = _RUNAWAY_PROMPTS[_v.guard]
                        st.finish_reason = "length"
                        break
                    continue
                if _ev == "heartbeat":
                    yield emitter.heartbeat()
                    pacing.touch()
                    continue
                if _ev == "error":
                    err_msg = _val.decode("utf-8", errors="ignore").strip()
                    if audit:
                        audit.log_error(request_id, "upstream_stream_error", err_msg[:500])
                    break
                chunk = _val
                if chunk.get("stop"):
                    _finish_on_upstream_stop(st, chunk, tool_stats)
                    final_usage = _merge_final_usage(final_usage, chunk)
                    break

                piece = chunk.get("content", "")
                if not piece:
                    continue
                st.raw_tail = (st.raw_tail + piece)[-800:]

                for fsm_event in fsm.feed(piece):
                    chunks = _handle_fsm_event(
                        fsm_event, emitter, tools, is_summary,
                        workspace_dir=workspace_dir, lodestar_state=state,
                        registered_names=frozenset(registered_names),
                        lodestar_todo=lodestar_todo, client_body=client_body,
                        audit=audit, request_id=request_id, session_id=workspace_dir,
                        stats=tool_stats, max_synthesis=config.max_synthesis_per_cycle,
                        synthesis_so_far=synthesis_before, budget=budget,
                        destructive_action=config.destructive_action,
                        subagent_hard_stop=subagent_hard_stop,
                        cycle_hard_stop=cycle_hard_stop,
                    )
                    for c in chunks:
                        yield c
                        pacing.touch()
                    if fsm_event.type == EventType.TOOL_CALL_END and not is_summary:
                        if not emitter.is_suppressed(fsm_event.data.get("index", -1)):
                            st.finish_reason = "tool_calls"

                if tool_stats.get("turn_budget_chunk_sent") or tool_stats.get("turn_loop_stop"):
                    _reason = (
                        "turn call budget reached"
                        if tool_stats.get("turn_budget_chunk_sent")
                        else "intra-turn tool loop detected"
                    )
                    _log.warning(
                        "STREAM: %s — stopping upstream generation cleanly", _reason
                    )
                    if emitter.has_emitted_tool_calls:
                        st.finish_reason = "tool_calls"
                    else:
                        st.finish_reason = "stop"
                    stop_upstream.set()
                    break

            # Flush parser after continuation
            for fsm_event in fsm.finish(st.finish_reason):
                _cut_off = _flush_truncated_call(fsm_event, st, emitter)
                if _cut_off and emitter.is_suppressed(fsm_event.data.get("index", -1)):
                    continue    # held back and cut off: never seen, never judged
                chunks = _handle_fsm_event(
                    fsm_event, emitter, tools, is_summary,
                    workspace_dir=workspace_dir, lodestar_state=state,
                    registered_names=frozenset(registered_names),
                    lodestar_todo=lodestar_todo, client_body=client_body,
                    audit=audit, request_id=request_id, session_id=workspace_dir,
                    stats=tool_stats, max_synthesis=config.max_synthesis_per_cycle,
                    synthesis_so_far=synthesis_before, budget=budget,
                    destructive_action=config.destructive_action,
                    subagent_hard_stop=subagent_hard_stop,
                    cycle_hard_stop=cycle_hard_stop,
                )
                for c in chunks:
                    yield c
                if fsm_event.type == EventType.TOOL_CALL_END and not is_summary and not _cut_off:
                    if not emitter.is_suppressed(fsm_event.data.get("index", -1)):
                        st.finish_reason = "tool_calls"
            if st.truncated_call_streamed:
                st.finish_reason = "length"

            # After the flush, never before it: the flush closes a half-written
            # tool call, and a guard paragraph inside one is the JSON parse
            # error this codebase refuses to trade for anything.
            if guard_prompt:
                try:
                    guard_msg = prompts.get(guard_prompt)
                except KeyError:
                    guard_msg = ""
                if guard_msg:
                    yield emitter.content_chunk(f"\n\n{guard_msg}\n")
                break

        if tool_stats.get("loop_message_emitted") and guard_streak is not None:
            guard_streak.record(workspace_dir, request_id, runaway.TOOL_LOOP)

        # In-stream loop recovery: if loop_breaker suppressed a repeated call and left the turn
        # with 0 emitted tool calls, steer the model into an alternative tool call or valid file action.
        while (
            not is_summary
            and tool_stats.get("loop_message_emitted")
            and not emitter.has_emitted_tool_calls
            and st.loop_recoveries < config.loop_recovery_max
            and prompt_transport
        ):
            st.loop_recoveries += 1
            tool_name = str(tool_stats.get("loop_tool_name") or "that")
            loop_msg = str(tool_stats.get("loop_message_text") or "")
            _log.warning(
                "STREAM: in-stream loop recovery (%d/%d): repeating tool call '%s' suppressed. "
                "Steering model into alternative tool call chat_id=%s request_id=%s",
                st.loop_recoveries, config.loop_recovery_max, tool_name, chat_id, request_id,
            )
            if audit:
                audit.log_error(
                    request_id, "instream_loop_recovery",
                    f"attempt {st.loop_recoveries}/{config.loop_recovery_max}: steering after '{tool_name}' suppressed",
                )

            await _shutdown_upstream_worker(
                worker, stop_upstream, upstream_holder, chat_id,
                grace=config.upstream_close_grace,
            )

            # Clear the flag so subsequent iterations don't re-trigger unless loop breaker trips again
            tool_stats["loop_message_emitted"] = 0

            # No `max_tokens` here: the parameter existed, was never read,
            # and the call site passed the prune limit into it — so anyone
            # reading this had grounds to believe the recovery prompt was
            # bounded by the context window. It was not (R-2026-04).
            continuation_prompt = _build_loop_recovery_prompt(
                base_prompt=base_prompt,
                loop_message=loop_msg,
                tool_name=tool_name,
            )
            generate_body = {**generate_body, "prompt": continuation_prompt}

            fsm = create_parser(
                model_name=model, tools=tools,
                in_triple_backticks=in_triple, in_single_backtick=in_single,
                start_in_thinking=False, start_in_tool_call=False,
                family=parser_family,
            )
            emitter.reset_tool_call_state()
            st.reset_for_retry()
            watch.reset()
            tool_stats.pop("current_turn_calls", None)
            tool_stats.pop("turn_loop_stop", None)

            q = asyncio.Queue()
            stop_upstream = asyncio.Event()
            upstream_holder = {}
            worker = asyncio.create_task(
                _fetch_upstream_to_queue(
                    q, upstream, generate_body, headers, chat_id,
                    upstream_override=upstream_override,
                    stop=stop_upstream, resp_holder=upstream_holder,
                )
            )
            pacing.touch()

            # Read recovery stream until completion.
            #
            # W-2026-03: this loop is a second reader, and the guards live in
            # the first one. Both are asked here through the same predicates
            # the main loop uses, so the two cannot drift apart again. The
            # ACTION differs on purpose: the main loop can still hand a
            # reasoning runaway to the silent retry, but that machinery
            # rebuilds the main loop's worker, and by here the turn has
            # already spent its one recovery. So the guard ends the attempt
            # and says why, which is what the model and the user need.
            guard_prompt = None
            async for _ev, _val in _upstream_events(
                    # by reference on purpose: the retry rebinds q
                    lambda: q, config, pacing):  # noqa: B023
                if _ev == "tick":
                    if _stream_time_exhausted(stream_start, config):
                        _log.warning(
                            "STREAM MAX DURATION (%ds) exceeded during loop "
                            "recovery chat_id=%s request_id=%s",
                            config.stream_max_duration, chat_id, request_id)
                        if audit:
                            audit.log_error(
                                request_id, "stream_timeout_guard",
                                f"loop recovery exceeded the safety limit "
                                f"({config.stream_max_duration}s)")
                        guard_prompt = "stream_timeout_guard"
                        st.finish_reason = "length"
                        break
                    if (_v := watch.check_stream(emitter.reasoning_text, fsm.in_thinking,
                                              emitter.content_text,
                                              fsm.tool_call_count > 0)) is not None:
                        _log.warning(
                            "RUNAWAY (%s) after %d chars during loop recovery "
                            "chat_id=%s request_id=%s",
                            _v.guard, _v.chars, chat_id, request_id)
                        if audit:
                            audit.log_error(
                                request_id, f"reasoning_{_v.guard}_guard",
                                f"loop recovery: {_v.chars} chars of reasoning")
                        guard_prompt = _RUNAWAY_PROMPTS[_v.guard]
                        st.finish_reason = "length"
                        break
                    continue
                if _ev == "heartbeat":
                    yield emitter.heartbeat()
                    pacing.touch()
                    continue
                if _ev == "error":
                    err_msg = _val.decode("utf-8", errors="ignore").strip()
                    if audit:
                        audit.log_error(request_id, "upstream_stream_error", err_msg[:500])
                    break
                chunk = _val
                if chunk.get("stop"):
                    _finish_on_upstream_stop(st, chunk, tool_stats)
                    final_usage = _merge_final_usage(final_usage, chunk)
                    break

                piece = chunk.get("content", "")
                if not piece:
                    continue
                st.raw_tail = (st.raw_tail + piece)[-800:]

                for fsm_event in fsm.feed(piece):
                    chunks = _handle_fsm_event(
                        fsm_event, emitter, tools, is_summary,
                        workspace_dir=workspace_dir, lodestar_state=state,
                        registered_names=frozenset(registered_names),
                        lodestar_todo=lodestar_todo, client_body=client_body,
                        audit=audit, request_id=request_id, session_id=workspace_dir,
                        stats=tool_stats, max_synthesis=config.max_synthesis_per_cycle,
                        synthesis_so_far=synthesis_before, budget=budget,
                        destructive_action=config.destructive_action,
                        subagent_hard_stop=subagent_hard_stop,
                        cycle_hard_stop=cycle_hard_stop,
                    )
                    for c in chunks:
                        yield c
                        pacing.touch()
                    if fsm_event.type == EventType.TOOL_CALL_END and not is_summary:
                        if not emitter.is_suppressed(fsm_event.data.get("index", -1)):
                            st.finish_reason = "tool_calls"

                if tool_stats.get("turn_budget_chunk_sent") or tool_stats.get("turn_loop_stop"):
                    _reason = (
                        "turn call budget reached"
                        if tool_stats.get("turn_budget_chunk_sent")
                        else "intra-turn tool loop detected"
                    )
                    _log.warning(
                        "STREAM: %s — stopping upstream generation cleanly", _reason
                    )
                    if emitter.has_emitted_tool_calls:
                        st.finish_reason = "tool_calls"
                    else:
                        st.finish_reason = "stop"
                    stop_upstream.set()
                    break

            # Flush parser after recovery
            for fsm_event in fsm.finish(st.finish_reason):
                _cut_off = _flush_truncated_call(fsm_event, st, emitter)
                if _cut_off and emitter.is_suppressed(fsm_event.data.get("index", -1)):
                    continue    # held back and cut off: never seen, never judged
                chunks = _handle_fsm_event(
                    fsm_event, emitter, tools, is_summary,
                    workspace_dir=workspace_dir, lodestar_state=state,
                    registered_names=frozenset(registered_names),
                    lodestar_todo=lodestar_todo, client_body=client_body,
                    audit=audit, request_id=request_id, session_id=workspace_dir,
                    stats=tool_stats, max_synthesis=config.max_synthesis_per_cycle,
                    synthesis_so_far=synthesis_before, budget=budget,
                    destructive_action=config.destructive_action,
                    subagent_hard_stop=subagent_hard_stop,
                    cycle_hard_stop=cycle_hard_stop,
                )
                for c in chunks:
                    yield c
                if fsm_event.type == EventType.TOOL_CALL_END and not is_summary and not _cut_off:
                    if not emitter.is_suppressed(fsm_event.data.get("index", -1)):
                        st.finish_reason = "tool_calls"
            if st.truncated_call_streamed:
                st.finish_reason = "length"

            # After the flush, never before it: the flush is what closes a
            # half-written tool call, and a guard paragraph emitted into the
            # middle of one is the JSON parse error this codebase refuses to
            # trade for anything.
            if guard_prompt:
                try:
                    guard_msg = prompts.get(guard_prompt)
                except KeyError:
                    guard_msg = ""
                if guard_msg:
                    yield emitter.content_chunk(f"\n\n{guard_msg}\n")
                break

        # Fallback auto-heal: if model STILL exhausted its generation without content or tools
        if (
            not is_summary
            and st.finish_reason != "tool_calls"
            and not emitter.has_emitted_content
            and not emitter.has_emitted_tool_calls
            and emitter.reasoning_text
        ):
            conclusion = _extract_conclusion_from_reasoning(emitter.reasoning_text)
            _log.info(
                "STREAM: auto-healed empty content from reasoning tail (%d chars) chat_id=%s request_id=%s",
                len(conclusion), chat_id, request_id,
            )
            yield emitter.content_chunk(f"{conclusion}\n")

        # A subagent's last turn is its entire deliverable. The client ends the
        # subagent the moment a turn carries no tool call, and whatever text is
        # in that turn becomes the task_result its parent reads — so the
        # announce-then-stop habit this model is documented for costs the
        # parent everything. Measured 2026-09-01: two subagents "completed"
        # after two minutes of work and handed back
        #
        #     "Let me check the full 0_RESOURCES/TPS7A4501 folder ..."
        #     "No calculator files in 3V9_LDO itself. Let me now read ..."
        #
        # while their <think> held the answer in full: which folders exist,
        # that 3V9_LDO has no calculator, and the three calculate.py files that
        # do. The main session learned nothing and delegated the same job
        # again. In a main session the stubborn guide catches this; a subagent
        # skips all of that machinery by design, so nothing did.
        #
        # The threshold is on length rather than on phrasing: an announcement
        # is a sentence, a report is not, and matching intent by wording would
        # be guessing at the model's mood.
        if (
            not is_summary
            and st.finish_reason != "tool_calls"
            and subagent_report_floor
            and emitter.reasoning_text
            and emitter.emitted_content_chars < subagent_report_floor
        ):
            conclusion = _extract_conclusion_from_reasoning(emitter.reasoning_text)
            # The model sometimes copies its own conclusion out of <think>
            # already. An exact substring test misses that — the extraction
            # trims differently, so 275 characters of content and 282 of
            # conclusion say the same thing and match nowhere. Compare the
            # opening of each with whitespace collapsed instead.
            # Tags are stripped as well as whitespace: reasoning_text carries
            # the literal `<think>` opener, which is enough on its own to make
            # two identical texts fail to match.
            def _key(text: str) -> str:
                return " ".join(re.sub(r"<[^>]{1,20}>", " ", text).split())

            _said = _key(emitter.content_text)
            _new = _key(conclusion)
            if conclusion and _new[:80] not in _said:
                _log.warning(
                    "SUBAGENT REPORT: turn ended with %d chars for its parent — "
                    "completing it from the reasoning (%d chars) chat_id=%s",
                    emitter.emitted_content_chars, len(conclusion), chat_id,
                )
                yield emitter.content_chunk(f"\n\n{conclusion}\n")

        _log.info("STREAM END chat_id=%s finish=%s tool_calls=%d request_id=%s",
                  chat_id, st.finish_reason, fsm.tool_call_count, request_id)

        # Audit: log response
        if audit and final_usage:
            audit.log_response(
                request_id=request_id,
                finish_reason=st.finish_reason,
                tokens_in=final_usage.get("prompt_tokens", 0),
                tokens_out=final_usage.get("completion_tokens", 0),
                duration_ms=int((time.time() - stream_start) * 1000),
                ttft_ms=st.ttft_ms,
                cached_tokens=cached_tokens_of(final_usage),
                queue_wait_ms=queue_wait_ms,
                tool_call_count=fsm.tool_call_count,
                session_id=workspace_dir,
                synthesized_count=tool_stats.get("synthesized", 0),
            )

        yield emitter.finish_chunk(st.finish_reason, client_usage(final_usage))
        yield emitter.done()

    except asyncio.CancelledError:
        _log.warning("STREAM CANCELLED chat_id=%s request_id=%s", chat_id, request_id)
        if audit:
            audit.log_error(request_id, "stream_cancelled", f"Client disconnected: {chat_id}")
        raise
    except Exception as e:
        _log.error("STREAM ERROR chat_id=%s error=%s request_id=%s", chat_id, e, request_id, exc_info=True)
        if audit:
            audit.log_error(request_id, "stream_error", str(e))

        # A transport error here means the upstream went away mid-response —
        # the model process crashed or was restarted under us. That is not a
        # failed REQUEST, and `error_chunk` tells the client it is: an SSE
        # error object carries no `choices`, so OpenCode abandons the agent
        # loop and the session is over (measured: 14 ms). Recoverable failures
        # go out as content, which is a turn — the user reads a sentence
        # instead of a raw JSON blob, and the model sees it on the next one.
        #
        # Measured 2026-09-09: the whole visible symptom of the eGPU crash was
        # `proxy stream error: ReadError - `, with an empty message because
        # httpx.ReadError carries no text. The message the user needed was
        # only ever in gateway.log.
        #
        # Only when nothing of a tool call has reached the client. Past that
        # point its arguments are already truncated on the wire, and finishing
        # the turn as if it were text would hand the client a half-written
        # tool call to parse — the one thing worse than the error object.
        # An upstream that REPORTED a failure in band is the same trade as
        # one whose socket died: the turn is lost, the session need not be.
        # A separate branch because the model is told a different thing --
        # the engine's own words, which may name the cause -- and because
        # it is audited under its own status: `stream_error` alone cannot
        # tell a CUDA out-of-memory from a proxy bug, and `/metrics` reads
        # that field.
        recovered = False
        if (isinstance(e, (httpx.TransportError, UpstreamReportedError))
                and not emitter.has_emitted_tool_calls):
            if isinstance(e, UpstreamReportedError):
                prompt_name = "upstream_reported_error"
                audit_status = "upstream_reported_error"
                prompt_params: dict[str, Any] = {"detail": e.detail}
                logged = e.detail[:300]
            else:
                prompt_name = "upstream_stream_lost"
                audit_status = "upstream_stream_lost"
                prompt_params = {}
                logged = type(e).__name__
            try:
                lost_msg = prompts.get(prompt_name, **prompt_params)
            except KeyError:
                lost_msg = ""
            if lost_msg:
                _log.warning(
                    "UPSTREAM FAILED: %s mid-stream — ending the turn with "
                    "a guard message instead of killing the session chat_id=%s",
                    logged, chat_id)
                if audit:
                    audit.log_error(request_id, audit_status, logged)
                yield emitter.content_chunk(f"\n\n{lost_msg}\n")
                yield emitter.finish_chunk("stop")
                recovered = True

        if not recovered:
            yield emitter.error_chunk(f"proxy stream error: {type(e).__name__} - {e}")
        yield emitter.done()
    finally:
        # `sem_acquired` is only ever True when `semaphore` is not None, but
        # that is an invariant of the code rather than of the types.
        if sem_acquired and semaphore is not None:
            semaphore.release()
        if worker and not worker.done():
            await _shutdown_upstream_worker(
                worker, stop_upstream, upstream_holder, chat_id,
                grace=config.upstream_close_grace,
            )


async def _shutdown_upstream_worker(
    worker: "asyncio.Task[Any] | None",
    stop: "asyncio.Event | None",
    resp_holder: dict,
    chat_id: str,
    grace: float = 2.0,
) -> None:
    """Stop the upstream reader so llama-server stops generating.

    `worker.cancel()` on its own does not do this, and the difference is not
    academic: measured 2026-08-31, a turn the client cancelled at 20:56:02 was
    still decoding at 21:00:50 — 10,742 tokens into a 32,000-token budget,
    holding the single upstream slot throughout, so the user's next message
    would have queued behind a generation nobody would ever read.

    The reason is that cancellation lands on the `await` inside the reader's
    `async for`, and the `await stream_ctx.__aexit__(...)` that would close the
    connection is itself an await inside a task that is already cancelled.
    llama-server aborts a slot when the socket it is writing to goes away; if
    nothing closes the socket, nothing aborts.

    So: ask the reader to break out of its loop and run its own `finally` (the
    clean path, which closes httpx properly), and only if it will not wind down
    within `grace` sever the response by hand and then cancel.

    `worker` is Optional because one of the two call sites reaches here from a
    retry path where the reader may never have started; there is nothing to
    shut down then.
    """
    if worker is None:
        return
    if stop is not None:
        stop.set()
        try:
            await asyncio.wait_for(asyncio.shield(worker), timeout=grace)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        except Exception:
            pass
    if worker.done():
        return

    _log.warning(
        "STREAM: upstream reader did not wind down in %.1fs chat_id=%s — "
        "closing the connection by force", grace, chat_id,
    )
    resp = resp_holder.get("response")
    if resp is not None:
        try:
            await resp.aclose()
        except Exception as exc:  # already broken is the outcome we wanted
            _log.debug("STREAM: forced aclose raised %s chat_id=%s", exc, chat_id)
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


@dataclass
class _CallContext:
    """Everything the TOOL_CALL_END handlers need, gathered once.

    Passing this instead of thirteen positional arguments is what makes the
    handlers below small enough to read; `record` is the audit/stats sink the
    dispatcher closes over.
    """
    emitter: OpenAISSEEmitter
    tools: list[dict]
    workspace_dir: str | None
    lodestar_state: int
    registered_names: frozenset[str] | None
    lodestar_todo: Any
    client_body: dict | None
    record: Callable[..., None]
    max_synthesis: int | None = None
    synthesis_so_far: int = 0
    budget: "SynthesisBudget | None" = None


def _handle_fsm_event(
    event: FsmEvent,
    emitter: OpenAISSEEmitter,
    tools: list[dict],
    is_summary: bool,
    workspace_dir: str | None = None,
    lodestar_state: int = 3,
    registered_names: frozenset[str] | None = None,
    lodestar_todo=None,
    client_body: dict | None = None,
    audit: AuditLogger | None = None,
    request_id: str = "",
    session_id: str | None = None,
    stats: dict | None = None,
    max_synthesis: int | None = None,
    synthesis_so_far: int = 0,
    budget: SynthesisBudget | None = None,
    destructive_action: str = destructive.WARN,
    subagent_hard_stop: bool = False,
    cycle_hard_stop: bool = False,
) -> list[bytes]:
    """Turn one FSM event into SSE bytes.

    Everything except TOOL_CALL_END is pure framing and goes straight to the
    emitter. TOOL_CALL_END is where the gateway decides what actually reaches
    the client, and it has five distinct outcomes — gated, rewritten as a
    subagent task, rewritten as a skill read, dropped, or emitted as-is. Each
    lives in its own function below; this one only routes.

    `audit`/`stats` record what happened to each call. Until 2026-07-29
    AuditLogger.log_tool_call existed but was never called from anywhere, so
    the question "how often does the model actually break the protocol?" could
    only be answered by reading gateway.log by hand.
    """
    def _record(name: str, status: str, reason: str | None = None,
                emitted: bool = True, arguments: str | None = None) -> None:
        if stats is not None:
            stats[status] = stats.get(status, 0) + 1
            # One writer for the turn's call count, in the single place every
            # outcome passes -- the same discipline as GuardCutStreak. Only
            # calls that actually reach the client are counted: a call the
            # gateway itself suppressed never ran, and charging the model for
            # the gateway's own interventions is how a budget becomes a tax.
            if emitted:
                stats["turn_tool_calls"] = stats.get("turn_tool_calls", 0) + 1
        if audit is not None:
            audit.log_tool_call(
                request_id=request_id, session_id=session_id, tool_name=name,
                parse_status=status, blocked_reason=reason, emitted=emitted,
                lodestar_state=lodestar_state, arguments=arguments,
            )

    # v2.5: buffer the whole call in states < READY instead of dropping it by
    # name at START. The old pre-emptive drop suppressed EVERY bash call in
    # state 1 — including the `embed.py search` the state directive itself
    # demands — which deadlocked the protocol, since state 1 cannot advance
    # without that search. Buffering means no partial chunks reach the client
    # and the enforcer decides at END with the full parameters in hand, which
    # also removed the "JSON Parse error: Expected '}'" class of client errors.
    if event.type == EventType.TOOL_CALL_START and not is_summary:
        # A subagent past budget + grace has every call stopped, so suppress at
        # START: nothing about this decision depends on the parameters, which
        # is what separates it from the name-based pre-emptive drop this file
        # forbids. Deciding at END would let the argument chunks reach the
        # client first, and a call dropped after partial emission arrives as a
        # JSON parse error.
        if (subagent_hard_stop or cycle_hard_stop) and isinstance(event.data, dict):
            emitter.suppress(event.data.get("index", -1))
            return []
        # A turn that has executed more calls than any healthy turn ever does.
        # Suppressed at START for the same reason the subagent stop is: nothing
        # about the decision reads the parameters, so this is a count and not
        # the name-based pre-emptive drop this file forbids -- and deciding at
        # END would let the argument chunks reach the client before the call
        # was dropped.
        #
        # This is the backstop for the loop that does NOT talk. The content
        # loop detector catches a model saying the same sentence while it
        # hammers; nothing catches one that hammers in silence, and the log
        # holds turns of 34, 36, 60 and 140 calls that did exactly that.
        _budget = (stats or {}).get("turn_call_budget") or 0
        if _budget and isinstance(event.data, dict) \
                and (stats or {}).get("turn_tool_calls", 0) >= _budget:
            emitter.suppress(event.data.get("index", -1))
            if stats is None or stats.get("turn_budget_chunk_sent"):
                return []
            # Told ONCE, like the loop breaker: a turn whose every remaining
            # call is suppressed would otherwise answer with the same
            # paragraph a dozen times, trading the model's loop for a loop in
            # the output.
            stats["turn_budget_chunk_sent"] = 1
            _log.warning(
                "TURN CALL BUDGET: %d tool calls already executed in this turn "
                "(limit %d) — suppressing the rest and telling the model "
                "request_id=%s", stats.get("turn_tool_calls", 0), _budget,
                request_id)
            if audit is not None:
                audit.log_error(request_id, "turn_call_budget",
                                f"{stats.get('turn_tool_calls', 0)} calls "
                                f"executed in one turn (limit {_budget})")
            # content, never error_chunk: an error object ends the session.
            return [emitter.content_chunk(
                "\n\n" + prompts.get("turn_call_budget_exhausted",
                                      count=stats.get("turn_tool_calls", 0),
                                      limit=_budget) + "\n")]
        if lodestar_state < int(LodestarState.READY) and isinstance(event.data, dict):
            emitter.defer(event.data.get("index", -1))
            return []
        # Same reasoning, different gate: the destructive guard also decides at
        # END, so anything it can block must not have streamed its arguments
        # first — a call dropped after partial emission reaches the client as
        # `JSON Parse error: Expected '}'`. Deferring is not the pre-emptive
        # name-based DROP this file forbids: nothing is decided here, the call
        # is only held so the decision at END can be acted on either way.
        #
        # Scoped to shell tools because those are what the guard blocks in
        # practice (destructive commands, a working directory that is not
        # there) and their arguments are small, so buffering costs nothing.
        # A `write` with an 800-line body is a different trade and is left
        # streaming; its own partial-emission window predates this and is
        # narrower, since only a path outside the workspace can close it.
        if (destructive_action != destructive.OFF
                and isinstance(event.data, dict)
                and (event.data.get("name") or "").lower() in SHELL_CANDIDATES):
            emitter.defer(event.data.get("index", -1))
            return []
        raw_call_name = (event.data.get("name") or "") if isinstance(event.data, dict) else ""
        raw_call_lower = raw_call_name.lower()
        # Reads join the searches: both loop guards decide at END, and a read
        # is the call a model repeats most. Its arguments are a path and two
        # numbers, so holding them costs nothing -- and a read that streamed
        # cannot be withdrawn (see emitter.has_streamed).
        if raw_call_lower in SEARCH_TOOLS or raw_call_lower in READ_TOOLS:
            emitter.defer(event.data.get("index", -1))
            return []
        if isinstance(event.data, dict) and should_buffer_tool(raw_call_name, tools):
            emitter.defer(event.data.get("index", -1))
            return []

    if event.type != EventType.TOOL_CALL_END:
        return emitter.emit_event(event, is_summary=is_summary)

    if is_summary:
        return []

    ctx = _CallContext(
        emitter=emitter, tools=tools, workspace_dir=workspace_dir,
        lodestar_state=lodestar_state, registered_names=registered_names,
        lodestar_todo=lodestar_todo, client_body=client_body, record=_record,
        max_synthesis=max_synthesis,
        # Calls fabricated earlier in this stream count too, not just the ones
        # already visible in history.
        synthesis_so_far=synthesis_so_far + (stats or {}).get("synthesized", 0),
        budget=budget,
    )
    d = event.data
    idx = d["index"]

    # A near-miss path is repaired before anything judges it, so every later
    # decision sees the file the model meant. Reads only — see path_repair.
    if not is_summary:
        _repair = path_repair.repair_params(
            emitter.resolve_name(d.get("name") or "") or (d.get("name") or ""),
            d.get("params", {}), workspace_dir)
        if _repair is not None:
            d["params"], _before, _after = _repair
            _log.warning("PATH REPAIR: %s — %r -> %r",
                         d.get("name"), _before, _after)
            _record(d.get("name") or "?", "repaired", "path_near_miss")

    # A subagent past budget + grace has its calls stopped. The notice the
    # normalizer appends asks first; this is what makes it a limit rather
    # than a suggestion, and tonight has shown repeatedly that the model
    # reads guidance and carries on regardless. The stage is computed once
    # per request, in the stream loop where the config lives — it cannot
    # change within a turn.
    if (subagent_hard_stop or cycle_hard_stop) and not is_summary:
        which = "SUBAGENT BUDGET" if subagent_hard_stop else "CYCLE BUDGET"
        _log.warning("%s: hard stop — '%s' not executed", which, d.get("name"))
        _record(d.get("name") or "?", "blocked",
                "subagent_budget" if subagent_hard_stop else "cycle_budget",
                emitted=False)
        emitter.suppress(idx)
        return []

    # A call the model has already run three times, with the same arguments
    # and the same result, is not going to say anything new the fourth time.
    # The client's own doom_loop detector sees this too, but all it can do is
    # ask the user — which puts a human in a loop the gateway can settle.
    if not is_summary and client_body:
        resolved_name = emitter.resolve_name(d.get("name") or "") or (d.get("name") or "")
        _call_key = (resolved_name.lower(), loop_breaker._canonical(d.get("params", {})))
        _intra_calls = stats.setdefault("current_turn_calls", []) if stats is not None else []
        # A call whose opening already reached the client can be completed and
        # nothing else: suppressing it here sends half a call, which the client
        # rejects as unparseable JSON and the model then re-issues -- the guard
        # becomes the loop. Such a call is let through and said so in the log;
        # the turn's call budget still bounds a model that keeps at it.
        _withdrawable = not emitter.has_streamed(idx)

        _is_loop, _repeats, _thresh, _had_not_found = loop_breaker.check_call_loop(
            resolved_name,
            d.get("params", {}), client_body.get("messages") or [])
        if _is_loop and not _withdrawable:
            _log.warning("LOOP BREAKER: '%s' repeated %d times, but it has already "
                         "streamed -- letting it through rather than breaking it",
                         d.get("name"), _repeats)
            _is_loop = False
        if _is_loop:
            _intra_calls.append(_call_key)
            _log.warning("LOOP BREAKER: '%s' already ran %d times with identical "
                         "arguments (thresh=%d, not_found=%s) — suppressing and telling the model",
                         d.get("name"), _repeats, _thresh, _had_not_found)
            _record(d.get("name") or "?", "blocked", f"loop:{_repeats}", emitted=False)
            emitter.suppress(idx)
            # Every repeated call in the turn is suppressed, but the model is
            # told ONCE. The first version spoke per call, and a turn with four
            # parallel repeats answered with the same paragraph four times over
            # — trading the model's loop for a loop in the output, which is
            # what the user saw in the subagent's reply.
            if stats is not None:
                stats["loop_message_emitted"] = 1
                stats["loop_tool_name"] = d.get("name") or "that"
                stats["loop_message_text"] = loop_breaker.loop_message(
                    d.get("name") or "that", _repeats, not_found=_had_not_found
                )
                if stats.get("loop_guard_chunk_sent"):
                    return []
                stats["loop_guard_chunk_sent"] = 1
                msg_text = stats["loop_message_text"]
            else:
                msg_text = loop_breaker.loop_message(
                    d.get("name") or "that", _repeats, not_found=_had_not_found
                )
            # content, never error_chunk: an error object ends the session.
            return [emitter.content_chunk("\n\n" + msg_text + "\n")]

        _intra_count = _intra_calls.count(_call_key)
        if _intra_count >= 1 and not _withdrawable:
            _log.warning(
                "INTRA-TURN DUPLICATE: '%s' repeated in the same turn, but it has "
                "already streamed -- letting it through rather than breaking it",
                d.get("name"))
            _intra_calls.append(_call_key)
        elif _intra_count >= 1:
            _intra_calls.append(_call_key)
            emitter.suppress(idx)
            if _intra_count >= 2:
                _log.warning(
                    "INTRA-TURN LOOP: '%s' called %d times in same turn — suppressing and stopping stream",
                    d.get("name"), _intra_count + 1)
                _record(d.get("name") or "?", "blocked", f"intra_turn_loop:{_intra_count + 1}", emitted=False)
                if stats is not None:
                    stats["loop_message_emitted"] = 1
                    stats["loop_tool_name"] = d.get("name") or "that"
                    stats["turn_loop_stop"] = 1
                    if not stats.get("loop_guard_chunk_sent"):
                        stats["loop_guard_chunk_sent"] = 1
                        msg_text = loop_breaker.loop_message(d.get("name") or "that", _intra_count + 1)
                        stats["loop_message_text"] = msg_text
                        return [emitter.content_chunk("\n\n" + msg_text + "\n")]
                return []
            else:
                _log.warning(
                    "INTRA-TURN DUPLICATE: '%s' called multiple times in same turn — suppressing duplicate",
                    d.get("name"))
                _record(d.get("name") or "?", "blocked", "intra_turn_duplicate", emitted=False)
                return []
        else:
            _intra_calls.append(_call_key)

    # Before anything protocol-related: this check applies in every state, and
    # in repos with no LODESTAR.md at all — which resolve to READY and would
    # otherwise never pass through a verdict.
    label = None
    if destructive_action != destructive.OFF and not is_summary:
        label = destructive.classify(
            emitter.resolve_name(d.get("name") or "") or (d.get("name") or ""),
            d.get("params", {}), workspace_dir,
        )
    if label:
        # An invented working directory is REPAIRED, not refused. Measured
        # 2026-08-31 22:26:21: refusing it emitted an SSE `error` object, and
        # OpenCode abandoned the whole request 14 ms later — no STREAM END for
        # that chat id, the agent loop dead. That is strictly worse than the
        # client's own permission prompt, which at least let the user decline
        # and carry on. The guidance text was never read by the model either:
        # an error object is not a turn, so only the human ever saw it.
        if destructive.is_unrunnable(label):
            _log.warning(
                "DESTRUCTIVE GUARD: repairing %s — %s (working directory "
                "dropped; the client runs in the workspace)",
                d.get("name"), label)
            _record(d.get("name") or "?", "repaired", f"destructive:{label}")
            d["params"] = destructive.strip_workdir(d.get("params", {}))
            label = None
    if label:
        if destructive_action == destructive.CONFIRM:
            _log.warning("DESTRUCTIVE GUARD: blocked %s — %s",
                         d.get("name"), label)
            _record(d.get("name") or "?", "blocked", f"destructive:{label}",
                    emitted=False)
            emitter.suppress(idx)
            # content, never error_chunk: an SSE error object has no `choices`
            # and clients treat it as a failed request, killing the session
            # instead of the call.
            return [emitter.content_chunk(
                "\n\n" + destructive.confirmation_message(label) + "\n")]
        _log.warning("DESTRUCTIVE GUARD: allowing %s — %s (action=warn)",
                     d.get("name"), label)
        _record(d.get("name") or "?", "destructive", f"destructive:{label}")

    # States 1-2: the protocol may answer on its own. None means "allowed,
    # carry on with normal handling".
    if lodestar_state < LodestarState.READY:
        gated = _gate_buffered_call(ctx, d)
        if gated is not None:
            return gated

    if emitter.is_buffered_task(idx):
        if (stats or {}).get("emitted_tasks", 0) >= 1:
            _log.warning(
                "LODESTAR ENFORCER: parallel subagent task suppressed (index=%s) — enforcing sequential single-task execution",
                idx,
            )
            ctx.record("task", "suppressed", "parallel_task_throttled", emitted=False)
            ctx.emitter.suppress(idx)
            return []
        if stats is not None:
            stats["emitted_tasks"] = stats.get("emitted_tasks", 0) + 1
        return _rewrite_task_call(ctx, d)

    if emitter.is_buffered_skill(idx):
        return _rewrite_skill_call(ctx, d)

    # Hallucinated calls suppressed back at TOOL_CALL_START.
    if emitter.is_suppressed(idx):
        return []

    # Deferred and allowed: `_emit_plain_call` only closes out arguments that
    # already streamed, and a deferred call has emitted no start chunk at all.
    if emitter.is_deferred(idx):
        return _emit_deferred_call(
            ctx, d, emitter.resolve_name(d.get("name") or "") or (d.get("name") or ""))

    return _emit_plain_call(ctx, d)


# ---------------------------------------------------------------------------
# TOOL_CALL_END handlers
# ---------------------------------------------------------------------------

def _gate_buffered_call(ctx: _CallContext, d: dict) -> list[bytes] | None:
    """Apply the LODESTAR decision to a completed call in states 1-2.

    Returns the chunks to emit, or None when the call is allowed through to
    normal handling. The decision itself lives in gateway.lodestar.policy so
    the non-streaming path reaches the same verdict; this function only turns
    it into SSE.
    """
    idx = d["index"]
    func_name = d.get("name") or ""
    raw_params = d.get("params", {})

    decision = decide_tool_call(
        func_name, raw_params, ctx.lodestar_state,
        tools=ctx.tools,
        registered_tools=ctx.registered_names,
        todo=ctx.lodestar_todo,
        workspace_dir=ctx.workspace_dir,
        client_body=ctx.client_body,
        # Aliases resolve BEFORE enforcement (read_file -> read): the raw alias
        # used to be checked against registered_tools and dropped as
        # "hallucinated" even though the emitter would have aliased it fine.
        resolve_name=ctx.emitter.resolve_name,
        max_synthesis=ctx.max_synthesis,
        synthesis_so_far=ctx.synthesis_so_far,
        rate_limited=(ctx.budget is not None
                      and ctx.budget.exhausted(ctx.workspace_dir)),
    )
    name_for_log = decision.canonical_name or func_name

    if decision.action == policy.DROP:
        if decision.reason == "hallucinated":
            _log.warning("LODESTAR ENFORCER: hallucinated tool '%s' suppressed in state %d",
                         func_name, ctx.lodestar_state)
        elif decision.reason == "synthesis_rate_limit":
            _log.error(
                "LODESTAR LIMIT: workspace over its synthesis rate — '%s' blocked, "
                "next required: %s", func_name, decision.next_text,
            )
        elif decision.reason == "synthesis_limit":
            _log.error(
                "LODESTAR LIMIT: refusing further synthesis in this cycle — "
                "'%s' blocked, next required: %s",
                func_name, decision.next_text,
            )
        elif decision.reason == "force_order":
            _log.info(
                "LODESTAR ENFORCER: force order (silent drop) — '%s' blocked, "
                "next required: %s", func_name, decision.next_text,
            )
        else:
            _log.info("LODESTAR ENFORCER: dropped tool call '%s' in state %d",
                      name_for_log, ctx.lodestar_state)
        ctx.record(name_for_log, "blocked", decision.reason, emitted=False)
        ctx.emitter.suppress(idx)
        return []  # silent: the user sees no error, the model retries

    if decision.action == policy.REMINDER:
        _log.info("LODESTAR ENFORCER: inject reminder for '%s' (file unchanged since last edit)",
                  name_for_log)
        ctx.record(name_for_log, "blocked", decision.reason, emitted=False)
        ctx.emitter.suppress(idx)
        # content, not error_chunk: an SSE error object has no `choices`, and a
        # client reads it as a failed request and abandons the agent loop —
        # measured 14 ms from emission to STREAM CANCELLED. A reminder that
        # ends the session is not a reminder.
        return [ctx.emitter.content_chunk(
            "\n\nNo-op: active-context.md has not changed since the last edit. "
            "Read the current content first if you need to update it.\n"
        )]

    if decision.action == policy.SYNTHESIZE and decision.synthesized is not None:
        tool_name, syn_args = decision.synthesized
        # NOTE: the original idx is deliberately NOT suppressed — the stream
        # loop keys finish_reason=tool_calls off this event's index, and a
        # synthetic call WAS emitted.
        syn_idx = 9000 + (idx if isinstance(idx, int) else 0)
        syn_call_id = f"syn_{uuid.uuid4().hex[:8]}"
        _log.info(
            "LODESTAR ENFORCER: auto-execute %s (synthetic call idx=%d id=%s) — '%s' blocked",
            decision.next_key, syn_idx, syn_call_id, func_name,
        )
        ctx.record(func_name, "blocked", "force_order", emitted=False)
        ctx.record(tool_name, "synthesized", decision.next_key, arguments=syn_args)
        if ctx.budget is not None:
            ctx.budget.record(ctx.workspace_dir)
        return [
            ctx.emitter.tool_call_start_chunk(syn_idx, syn_call_id, tool_name),
            ctx.emitter.tool_call_arg_chunk(syn_idx, syn_args),
        ]

    if ctx.emitter.is_deferred(idx):
        return _emit_deferred_call(ctx, d, decision.canonical_name or "")

    return None


def _emit_deferred_call(ctx: _CallContext, d: dict, canonical_name: str) -> list[bytes]:
    """Emit an allowed buffered call as one complete, validated argument chunk."""
    idx = d["index"]
    coerced = _coerce_params(canonical_name, d.get("params", {}), ctx.tools)
    missing = _missing_required_args(canonical_name, coerced, ctx.tools)
    args_str = _validate_tool_args_json(
        json.dumps({**coerced, **missing}, ensure_ascii=False), canonical_name, idx)
    if args_str is None:
        ctx.emitter.suppress(idx)
        return []
    if ctx.lodestar_todo is not None:
        # Keep the todo current within this turn, so a later call in the SAME
        # stream (an embed search right after the read) is not blocked against
        # stale history.
        ctx.lodestar_todo.mark_call(canonical_name, coerced)
    _log.info("DEFERRED EMIT: '%s' in state %d — buffered at START so the "
              "decision at END could go either way",
              canonical_name, ctx.lodestar_state)
    ctx.record(canonical_name, "ok")
    call_id = d.get("id") or f"call_{uuid.uuid4().hex[:10]}"
    return [
        ctx.emitter.tool_call_start_chunk(idx, call_id, canonical_name),
        ctx.emitter.tool_call_arg_chunk(idx, args_str),
    ]


def _rewrite_task_call(ctx: _CallContext, d: dict) -> list[bytes]:
    """Append the subagent directive to a delegated prompt.

    OpenCode task-subagent sessions carry no marker of their own, so an
    undetected subagent used to run the FULL protocol — including Phase D,
    which cleared active-context.md while the main session's task was still
    in progress (2026-07-09 incident).
    """
    idx = d["index"]
    target_tool = first_available(ctx.tools, ("subagent", "task")) or "task"
    coerced = _coerce_params(target_tool, d.get("params", {}), ctx.tools)

    target = next((c for c in ("prompt", "description", "task", "instructions")
                   if isinstance(coerced.get(c), str)), None)
    if target is None:
        str_params = [(k, v) for k, v in coerced.items() if isinstance(v, str)]
        if str_params:
            target = max(str_params, key=lambda kv: len(kv[1]))[0]
    if target:
        if _SUBAGENT_TASK_MARKER.strip() not in coerced[target]:
            coerced[target] = coerced[target].rstrip() + _SUBAGENT_TASK_MARKER

    # For OpenCode 2.0 `subagent`, ensure required fields: agent, description, prompt
    if target_tool == "subagent":
        if "agent" not in coerced:
            coerced["agent"] = coerced.get("subagent_type") or "general"
        if "prompt" not in coerced and target:
            coerced["prompt"] = coerced[target]
        if "description" not in coerced:
            coerced["description"] = "subagent task"

    missing = _missing_required_args(target_tool, coerced, ctx.tools)
    args_str = _validate_tool_args_json(
        json.dumps({**coerced, **missing}, ensure_ascii=False), target_tool, idx)
    if args_str is None:
        ctx.emitter.suppress(idx)
        return []
    _log.info("LODESTAR ENFORCER: %s call rewritten with subagent directive (param=%s)",
              target_tool, target)
    ctx.record(target_tool, "rewritten", "subagent_directive")
    call_id = d.get("id") or f"call_{uuid.uuid4().hex[:10]}"
    return [
        ctx.emitter.tool_call_start_chunk(idx, call_id, target_tool),
        ctx.emitter.tool_call_arg_chunk(idx, args_str),
    ]


def _rewrite_skill_call(ctx: _CallContext, d: dict) -> list[bytes]:
    """Turn skill(name) into a read of that skill's SKILL.md.

    The client has no `skill` tool; loading a skill means reading its file.
    """
    idx = d["index"]
    raw_params = d.get("params", {})
    skill_name = (raw_params.get("skill_name") or raw_params.get("name")
                  or raw_params.get("id") or raw_params.get("tool_call"))
    read_tool = first_available(ctx.tools, READ_CANDIDATES)
    if not (read_tool and skill_name):
        return []

    skill_path = resolve_skill_path(ctx.workspace_dir, skill_name)
    if skill_path is None:
        ctx.record("skill", "blocked", "skill_not_found", emitted=False)
        ctx.emitter.suppress(idx)
        return []

    param_name = _path_param_for(ctx.tools, read_tool)
    args_str = json.dumps({param_name: str(skill_path)}, ensure_ascii=False)
    _log.info("STREAM REWRITE SKILL CALL: skill(%s) -> %s(%s=%s)",
              skill_name, read_tool, param_name, skill_path)
    ctx.record("skill", "rewritten", "skill_to_read")
    call_id = d.get("id") or f"call_{uuid.uuid4().hex[:10]}"
    return [
        ctx.emitter.tool_call_start_chunk(idx, call_id, read_tool),
        ctx.emitter.tool_call_arg_chunk(idx, args_str),
    ]


def _emit_plain_call(ctx: _CallContext, d: dict) -> list[bytes]:
    """Close out a call whose parameters already streamed through untouched."""
    idx = d["index"]
    func_name = d.get("name") or ""
    ctx.record(func_name, "ok")

    coerced = _coerce_params(func_name, d.get("params", {}), ctx.tools)
    missing = _missing_required_args(func_name, coerced, ctx.tools)

    if not ctx.emitter.has_emitted_param(idx):
        if not missing:
            # Zero-param call: emit a complete empty object, not nothing.
            return [ctx.emitter.tool_call_arg_chunk(idx, "{}")]
        args_str = _validate_tool_args_json(
            json.dumps({**coerced, **missing}, ensure_ascii=False), func_name, idx)
        if args_str is None:
            return []  # unrepairable — drop silently
        return [ctx.emitter.tool_call_arg_chunk(idx, args_str)]

    # Parameters already streamed: append what the model omitted, then close.
    chunks = [
        ctx.emitter.tool_call_arg_chunk(
            idx, f',\n  "{key}": {json.dumps(val, ensure_ascii=False)}')
        for key, val in missing.items()
    ]
    chunks.append(ctx.emitter.tool_call_arg_chunk(idx, "\n}"))
    return chunks


def _coerce_params(func_name: str, raw_params: dict, tools: list[dict]) -> dict:
    """Cast every parameter to the type its schema declares."""
    if func_name == "question":
        raw_params = heal_question_arguments(raw_params)
    return {
        k: coerce_value(v, _get_prop_schema(func_name, k, tools))
        for k, v in raw_params.items()
    }



def _missing_required_args(func_name: str, present: dict, tools: list[dict]) -> dict:
    """Compute placeholders for required args the model omitted."""
    missing: dict = {}
    for t in tools:
        fn = t.get("function") or {}
        if fn.get("name") == func_name:
            schema = fn.get("parameters") or {}
            required = schema.get("required") or []
            properties = schema.get("properties") or {}
            for r in required:
                if r not in present:
                    if r in CRITICAL_FIELDS:
                        continue
                    missing[r] = placeholder_for(r, present, properties.get(r) or {})
            break
    return missing


def _get_prop_schema(func_name: str, param_name: str, tools: list[dict]) -> dict | None:
    """Look up the JSON Schema for a specific parameter."""
    for t in tools:
        fn = t.get("function") or {}
        if fn.get("name") == func_name:
            return (fn.get("parameters") or {}).get("properties", {}).get(param_name)
    return None


# ---------------------------------------------------------------------------
# Non-streaming path (unified FSM parsing)
# ---------------------------------------------------------------------------

def _guard_completion(text: str, *, model: str,
                      audit: AuditLogger | None = None,
                      request_id: str = "", t_start: float = 0.0,
                      queue_wait_ms: int | None = None) -> JSONResponse:
    """A guard's own words as a finished, ordinary-looking turn.

    The non-streaming counterpart of `emitter.content_chunk` + `finish("stop")`.
    A guard that has something to say to the MODEL must say it as content: an
    HTTP error, like an SSE error object, is not a turn, so the model never
    reads it and only the user sees a dead request.

    `finish_reason` is `stop` rather than anything sharper, because the client
    branches on it and there is no value meaning "the gateway intervened" --
    which is what the `[SYSTEM GUARD]` prefix in the text is for.
    """
    resp = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "model": model,
        "created": int(time.time()),
        "object": "chat.completion",
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    if audit:
        audit.log_response(
            request_id=request_id, finish_reason="stop", tokens_in=0,
            tokens_out=0,
            duration_ms=int((time.time() - t_start) * 1000) if t_start else 0,
            ttft_ms=None, queue_wait_ms=queue_wait_ms, tool_call_count=0)
    return JSONResponse(content=resp)


async def _non_stream_completion(
    client_body: dict,
    headers: dict,
    client_tools: list[dict],
    is_summary: bool,
    upstream: UpstreamClient,
    config: GatewayConfig,
    audit: AuditLogger | None = None,
    request_id: str = "",
    t_start: float = 0.0,
    semaphore: PrioritySemaphore | None = None,
    workspace_dir: str | None = None,
    budget: SynthesisBudget | None = None,
    spill_count: int = 0,
    compaction: bool = False,
    upstream_override: str | None = None,
    model_spec: "ModelSpec | None" = None,
    guard_streak: "runaway.GuardCutStreak | None" = None,
) -> JSONResponse:
    """Non-streaming chat completion — collects upstream chunks, then uses FSM."""
    sem_acquired = False
    queue_start = time.time()
    queue_wait_ms: int | None = None
    try:
        if semaphore:
            try:
                await asyncio.wait_for(
                    semaphore.acquire(request_priority(
                        has_tools=bool(client_tools), is_summary=is_summary)),
                    timeout=config.semaphore_timeout)
                sem_acquired = True
                queue_wait_ms = int((time.time() - queue_start) * 1000)
            except asyncio.TimeoutError:
                if audit:
                    audit.log_error(request_id, "upstream_busy", "concurrency slot timeout")
                return JSONResponse(
                    status_code=503,
                    content={"error": {"message": "upstream busy - try again later", "code": 503}},
                )

        # The card, AFTER the slot and before anything is sent. Both halves of
        # that order matter: holding the slot means no generation is in flight
        # on the engine about to be unloaded, and swapping before the request
        # is composed means the prompt is never built against an engine that
        # is going away.
        try:
            routed = await _hand_over_card(model_spec, config)
            if routed:
                upstream_override = routed
        except engines.EngineError as exc:
            _log.error("ENGINE SWAP FAILED id=%s: %s", request_id, exc)
            if audit:
                audit.log_error(request_id, "engine_swap_failed", str(exc))
            # As CONTENT, exactly as the streaming path does, and for the
            # reason this file states without qualifying it by transport: a
            # failed hand-over ends the TURN, not the session. This branch
            # returned HTTP 503 until 2026-09-13, so the same failure reached
            # OpenCode as a dead request on one path and as a readable
            # sentence on the other -- the policy was written once and
            # implemented once, in the path somebody happened to be looking
            # at. Seen the moment Gemma was asked for with the gateway still
            # holding the previous revision's paths.
            named = model_spec.id if model_spec else client_body.get("model", "")
            return _guard_completion(
                f"[SYSTEM GUARD] The engine for `{named}` could not be "
                f"started, so this turn did not reach a model.\n\n{exc}",
                model=named, audit=audit, request_id=request_id,
                t_start=t_start, queue_wait_ms=queue_wait_ms)

        use_chat = wants_chat_transport(model_spec)
        guard_cuts = guard_streak.depth(workspace_dir) if guard_streak else 0
        generate_body = _build_upstream_body(client_body, config, use_chat, guard_cuts)
        # Fix (2026-08-31): Extract target model name from generate_body so _parse_with_fsm
        # receives model_name without raising NameError on non-streaming /chat/completions calls.
        model = generate_body.get('model', '')
        start_in_thinking = False
        if (
            not use_chat
            and detect_model_family(model) == "qwen3"
            and not is_summary
        ):
            generate_body["prompt"] += "<think>\n"
            start_in_thinking = True
        server_body = generate_body.copy()
        # Asks about the address this request is GOING to, not the preset's
        # default. It read `upstream.resolve()` until 2026-09-12, so a request
        # routed elsewhere -- by escalation then, by the model registry now --
        # decided how to name its model from an upstream it was not being
        # sent to. The chat endpoint takes `model` as part of its contract; on
        # /completion the engine serves whatever it loaded and the field is
        # noise.
        path = "/v1/chat/completions" if "messages" in server_body else "/completion"
        if "messages" in server_body:
            server_body["model"] = model
        else:
            server_body.pop("model", None)

        # Same record as the streaming path. This path applies no prefill, so
        # the prompt is whatever normalization and pruning produced.
        if audit:
            audit.log_prompt(
                request_id=request_id,
                prompt=generate_body.get("prompt", ""),
                session_id=workspace_dir,
                model=model,
                lodestar_state=(
                    normalizer.determine_lodestar_state(
                        client_body.get("messages") or [], config)
                    if client_tools else int(LodestarState.READY)
                ),
                prefill=start_in_thinking,
                sampling=generate_body,
                spill_count=spill_count,
                compaction=compaction,
                dump_dir=(config.log_dir / "prompts") if config.audit_full_prompts else None,
            )

        resolved = upstream_override or upstream.resolve()
        # A chat body has `messages`, a completion body has `prompt`; the
        # endpoint follows from the BODY rather than from a flag somebody
        # has to remember to pass. `/completion` is where this gateway has
        # always gone, and the only reason the other exists is that it is
        # the only one that sees an image -- see build_chat_body.
        path = "/v1/chat/completions" if "messages" in server_body else "/completion"
        url = f"{resolved}{path}"

        try:
            stream_ctx = upstream.client.stream(
                "POST", url,
                content=json.dumps(server_body).encode("utf-8"),
                headers={**headers, "content-type": "application/json"},
            )
            upstream_resp = await stream_ctx.__aenter__()
        except (httpx.NetworkError, httpx.TimeoutException) as e:
            _log.warning("Non-stream connection failed to %s: %s. Re-resolving and retrying.", url, e)
            resolved = upstream_override or upstream.resolve(force=True)
            url = f"{resolved}{path}"
            stream_ctx = upstream.client.stream(
                "POST", url,
                content=json.dumps(server_body).encode("utf-8"),
                headers={**headers, "content-type": "application/json"},
            )
            upstream_resp = await stream_ctx.__aenter__()

        try:
            if upstream_resp.status_code != 200:
                err_content = await upstream_resp.aread()
                if audit:
                    audit.log_error(
                        request_id, "upstream_status",
                        f"HTTP {upstream_resp.status_code}: "
                        f"{err_content.decode('utf-8', errors='ignore').strip()[:500]}")
                return _wrap_error_to_json(upstream_resp.status_code, err_content)

            full_content = ""
            stopped_limit = False
            tokens_evaluated = 0
            tokens_predicted = 0
            # The FIFTH reader of an upstream stop chunk. It pulled the two
            # counts into locals rather than building a usage dict, which is
            # why it did not show up in the count of hand-written copies --
            # a different spelling of the same duplication, and the reason it
            # was the one path with no `cached_tokens`.
            final_usage: dict | None = None

            # The guards this path did not have. Measured 2026-09-12: seven
            # turns ran through here to the max_n_predict ceiling, 16,384
            # tokens and a median of fourteen minutes of the single slot each,
            # with no reasoning budget, no loop detector, no code detector and
            # no duration cap between them and the wall. The QUESTION is the
            # same call the streaming path makes; only the ACTION differs, and
            # it differs because there is no seam here to hide a retry behind.
            watch = runaway.RunawayWatch(
                config, is_summary,
            on_cut=(lambda g: guard_streak.record(workspace_dir, request_id, g))
            if guard_streak else None)
            guard_prompt: str | None = None
            guard_params: dict[str, Any] = {}
            gen_start = time.time()

            async for line in upstream_resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    continue
                try:
                    raw_chunk = json.loads(payload)
                except Exception:
                    continue
                # Asked OUTSIDE the decode guard above. That `except` swallows
                # everything, so a raise placed inside it would be caught and
                # this path would go straight back to answering nothing.
                detail = upstream_error_text(raw_chunk)
                if detail is not None:
                    _log.warning(
                        "UPSTREAM FAILED (non-stream): %s request_id=%s",
                        detail[:300], request_id)
                    if audit:
                        audit.log_error(
                            request_id, "upstream_reported_error", detail[:300])
                    guard_prompt = "upstream_reported_error"
                    guard_params = {"detail": detail}
                    break
                chunk = _as_completion_chunk(raw_chunk)

                if chunk.get("stop"):
                    stopped_limit = chunk.get("stopped_limit", False) or chunk.get("stop_type") == "limit"
                    final_usage = _merge_final_usage(final_usage, chunk)
                    tokens_evaluated = final_usage["prompt_tokens"]
                    tokens_predicted = final_usage["completion_tokens"]
                    content_piece = chunk.get("content", "")
                    if content_piece:
                        full_content += content_piece
                    break
                full_content += chunk.get("content", "")

                if _stream_time_exhausted(gen_start, config):
                    _log.warning(
                        "NON-STREAM: exceeded the %ds safety limit after %d "
                        "chars request_id=%s",
                        config.stream_max_duration, len(full_content), request_id)
                    if audit:
                        audit.log_error(
                            request_id, "stream_timeout_guard",
                            f"non-stream exceeded {config.stream_max_duration}s")
                    guard_prompt = "stream_timeout_guard"
                    stopped_limit = True
                    break

                reasoning_so_far, still_thinking = runaway.open_think_text(
                    full_content, start_in_thinking)
                content_so_far, calls_started = runaway.content_and_calls(
                    full_content, start_in_thinking)
                verdict = watch.check_stream(
                    reasoning_so_far, still_thinking,
                    content_so_far, calls_started)
                if verdict is not None:
                    _log.warning(
                        "NON-STREAM RUNAWAY (%s): %d chars of reasoning "
                        "request_id=%s", verdict.guard, verdict.chars, request_id)
                    if audit:
                        audit.log_error(
                            request_id, f"reasoning_{verdict.guard}_guard",
                            f"non-stream: {verdict.chars} chars of reasoning")
                    guard_prompt = _RUNAWAY_PROMPTS[verdict.guard]
                    stopped_limit = True
                    break
        finally:
            # Breaking out of aiter_lines runs this, which closes the response
            # -- and closing the socket is what actually stops the upstream.
            # A bare `break` without it would leave the generation running into
            # a socket nobody reads. See "Cancelling must close the socket".
            await stream_ctx.__aexit__(None, None, None)

        # --- Parse full response using FSM ---
        finish_reason = "length" if stopped_limit else "stop"
        in_triple, in_single = get_initial_backtick_state(client_body.get("messages") or [])
        content, reasoning, tool_calls = _parse_with_fsm(
            full_content, finish_reason, client_tools, is_summary,
            in_triple=in_triple, in_single=in_single, model_name=model,
            start_in_thinking=start_in_thinking,
            family=(model_spec.family if model_spec is not None else None),
        )

        # The guard's own words, as CONTENT. Never an error object: guidance
        # meant for the model must reach it as a turn, and an error object is
        # not one -- only the user would ever see it.
        if guard_prompt and not tool_calls:
            try:
                content = (content or "").rstrip()
                content = (f"{content}\n\n"
                           f"{prompts.get(guard_prompt, **guard_params)}\n").lstrip()
            except KeyError:
                _log.error("no prompt named %r", guard_prompt)

        if tool_calls and client_tools:
            registry = ToolSchemaRegistry(client_tools)
            state = normalizer.determine_lodestar_state(client_body.get("messages") or [], config)
            registry.process_tool_calls(tool_calls, allow_skills=True, workspace_dir=workspace_dir)

            tool_calls = _guard_destructive_calls(
                tool_calls, workspace_dir, config.destructive_action,
                audit=audit, request_id=request_id, session_id=workspace_dir,
                state=state,
            )

            # LODESTAR enforcement. Runs the same decision as the streaming
            # path (gateway.lodestar.policy), synthesis included — before
            # 2026-07-29 this half only dropped calls, so the same request
            # behaved differently with stream:false than with stream:true.
            if state < int(LodestarState.READY):
                tool_calls = _enforce_non_stream_calls(
                    tool_calls, state, client_body, client_tools,
                    frozenset(registry.tool_names), workspace_dir,
                    audit=audit, request_id=request_id, session_id=workspace_dir,
                    max_synthesis=config.max_synthesis_per_cycle,
                    budget=budget,
                )

            for i, c in enumerate(tool_calls):
                c.setdefault("index", i)

        # Auto-heal: if model exhausted its generation in <think> without emitting content or tools
        if (
            not is_summary
            and not tool_calls
            and (content is None or not str(content).strip())
            and reasoning
        ):
            content = _extract_conclusion_from_reasoning(reasoning)
            _log.info(
                "NON-STREAM: auto-healed empty content from reasoning tail (%d chars)",
                len(content),
            )

        msg: dict = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
            msg["reasoning"] = reasoning
        if tool_calls:
            msg["tool_calls"] = tool_calls

        resp_finish = "tool_calls" if tool_calls and not is_summary else finish_reason

        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "model": generate_body["model"],
            "created": int(time.time()),
            "object": "chat.completion",
            "choices": [{"index": 0, "message": msg, "finish_reason": resp_finish}],
            "usage": final_usage or {
                "prompt_tokens": tokens_evaluated,
                "completion_tokens": tokens_predicted,
                "total_tokens": tokens_evaluated + tokens_predicted,
            },
        }

        if audit:
            audit.log_response(
                request_id=request_id,
                finish_reason=resp_finish,
                tokens_in=tokens_evaluated,
                tokens_out=tokens_predicted,
                duration_ms=int((time.time() - t_start) * 1000) if t_start else 0,
                # No ttft on this path: the body arrives in one piece, so there
                # is no first token to time.
                ttft_ms=None,
                queue_wait_ms=queue_wait_ms,
                cached_tokens=cached_tokens_of(final_usage),
                tool_call_count=len(tool_calls),
            )

        return JSONResponse(content=resp)

    except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ReadError) as e:
        _log.error("TIMEOUT/READ ERROR error=%s request_id=%s", e, request_id)
        if audit:
            audit.log_error(request_id, "upstream_timeout", str(e))
        return JSONResponse(
            status_code=504,
            content={"error": {"message": f"proxy upstream timeout: {e}", "code": 504}},
        )
    except httpx.HTTPError as e:
        _log.error("UPSTREAM ERROR error=%s request_id=%s", e, request_id, exc_info=True)
        if audit:
            audit.log_error(request_id, "upstream_error", str(e))
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"proxy upstream error: {e}", "code": 502}},
        )
    except Exception as e:
        # Everything the upstream itself can raise is an httpx.HTTPError, and
        # it is caught above. What reaches HERE is by definition a fault of the
        # gateway — filing it under `upstream_error` is how 247 of the 253
        # events with that name (one NameError, one day) sent diagnosis to the
        # wrong machine. O-2026-02.
        _log.error("GATEWAY ERROR error=%s request_id=%s", e, request_id, exc_info=True)
        if audit:
            audit.log_error(request_id, "gateway_error", f"{type(e).__name__}: {e}")
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"gateway error: {type(e).__name__}: {e}",
                               "code": 502}},
        )
    finally:
        # `sem_acquired` is only ever True when `semaphore` is not None, but
        # that is an invariant of the code rather than of the types.
        if sem_acquired and semaphore is not None:
            semaphore.release()



def _guard_destructive_calls(
    tool_calls: list[dict],
    workspace_dir: str | None,
    action: str,
    audit: AuditLogger | None = None,
    request_id: str = "",
    session_id: str | None = None,
    state: int = 3,
) -> list[dict]:
    """Filter model-issued calls recognised as destructive.

    Kept apart from `_enforce_non_stream_calls`, which only runs while LODESTAR
    is gating (states 1-2). This has to apply in state 3 as well — that is
    where the model does its actual work, and it is the only state a repo
    without `.context/LODESTAR.md` is ever in.
    """
    for call in tool_calls:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        repair = path_repair.repair_params(
            fn.get("name") or "", args if isinstance(args, dict) else {},
            workspace_dir)
        if repair is None:
            continue
        fixed, before, after = repair
        _log.warning("PATH REPAIR (non-stream): %s — %r -> %r",
                     fn.get("name"), before, after)
        fn["arguments"] = json.dumps(fixed, ensure_ascii=False)
        if audit is not None:
            audit.log_tool_call(
                request_id=request_id, session_id=session_id,
                tool_name=fn.get("name") or "?", parse_status="repaired",
                blocked_reason="path_near_miss", emitted=True,
                lodestar_state=state)

    if action == destructive.OFF or not tool_calls:
        return tool_calls

    kept: list[dict] = []
    for call in tool_calls:
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, ValueError, TypeError):
            args = {}
        label = destructive.classify(name, args if isinstance(args, dict) else {},
                                     workspace_dir)
        if not label:
            kept.append(call)
            continue

        if destructive.is_unrunnable(label):
            _log.warning("DESTRUCTIVE GUARD (non-stream): repairing %s — %s",
                         name, label)
            fn["arguments"] = json.dumps(
                destructive.strip_workdir(args if isinstance(args, dict) else {}),
                ensure_ascii=False)
            if audit is not None:
                audit.log_tool_call(
                    request_id=request_id, session_id=session_id, tool_name=name,
                    parse_status="repaired", blocked_reason=f"destructive:{label}",
                    emitted=True, lodestar_state=state)
            kept.append(call)
            continue

        blocked = action == destructive.CONFIRM
        _log.warning("DESTRUCTIVE GUARD (non-stream): %s %s — %s",
                     "blocked" if blocked else "allowing", name, label)
        if audit is not None:
            audit.log_tool_call(
                request_id=request_id, session_id=session_id, tool_name=name,
                parse_status="blocked" if blocked else "destructive",
                blocked_reason=f"destructive:{label}", emitted=not blocked,
                lodestar_state=state,
            )
        if not blocked:
            kept.append(call)
    return kept


def _enforce_non_stream_calls(
    tool_calls: list[dict],
    state: int,
    client_body: dict,
    tools: list[dict],
    registered: frozenset[str],
    workspace_dir: str | None,
    audit: AuditLogger | None = None,
    request_id: str = "",
    session_id: str | None = None,
    max_synthesis: int | None = None,
    budget: SynthesisBudget | None = None,
) -> list[dict]:
    """Apply the LODESTAR decision to a non-streaming response.

    Same decisions as the streaming path, serialised as `tool_calls` entries
    instead of SSE chunks. Names arrived canonical here — the schema registry
    resolved them before enforcement — so no resolver is passed.
    """
    from gateway.lodestar import LodestarTodo
    from gateway.lodestar.context_injector import user_requested_skills

    msgs = client_body.get("messages") or []
    todo = LodestarTodo.from_messages(
        msgs, LodestarState(state), user_wants_skills=user_requested_skills(msgs),
        workspace_dir=workspace_dir,
    )

    def _record(name: str, status: str, reason: str | None, emitted: bool,
                arguments: str | None = None) -> None:
        if audit is not None:
            audit.log_tool_call(
                request_id=request_id, session_id=session_id, tool_name=name,
                parse_status=status, blocked_reason=reason, emitted=emitted,
                lodestar_state=state, arguments=arguments,
            )

    kept: list[dict] = []
    synthesized_here = 0
    emitted_tasks = 0
    for call in tool_calls:
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, ValueError, TypeError):
            args = {}
        if not isinstance(args, dict):
            args = {}

        decision = decide_tool_call(
            name, args, state,
            tools=tools,
            registered_tools=registered or None,
            todo=todo,
            workspace_dir=workspace_dir,
            client_body=client_body,
            max_synthesis=max_synthesis,
            synthesis_so_far=synthesis_count(msgs) + synthesized_here,
            rate_limited=(budget is not None and budget.exhausted(workspace_dir)),
        )

        if decision.action == policy.ALLOW:
            if name == "task":
                if emitted_tasks >= 1:
                    _log.warning(
                        "LODESTAR ENFORCER (non-stream): parallel subagent task dropped — enforcing sequential single-task execution"
                    )
                    _record(name, "blocked", "parallel_task_throttled", False)
                    continue
                emitted_tasks += 1
            kept.append(call)
            _record(name, "ok", None, True)
            continue

        if decision.action == policy.SYNTHESIZE and decision.synthesized is not None:
            syn_tool, syn_args = decision.synthesized
            _log.info(
                "LODESTAR ENFORCER (non-stream): auto-execute %s — '%s' blocked",
                decision.next_key, name,
            )
            _record(name, "blocked", "force_order", False)
            _record(syn_tool, "synthesized", decision.next_key, True, arguments=syn_args)
            synthesized_here += 1
            if budget is not None:
                budget.record(workspace_dir)
            kept.append({
                "id": f"syn_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": syn_tool, "arguments": syn_args},
            })
            continue

        _log.info("LODESTAR ENFORCER (non-stream): dropped '%s' in state %d (%s)",
                  name, state, decision.reason)
        _record(name, "blocked", decision.reason, False)

    return kept


def _parse_with_fsm(
    full_content: str,
    finish_reason: str,
    tools: list[dict],
    is_summary: bool,
    in_triple: bool = False,
    in_single: bool = False,
    model_name: str | None = None,
    start_in_thinking: bool = False,
    start_in_tool_call: bool = False,
    family: str | None = None,
) -> tuple[str | None, str | None, list[dict]]:
    """Parse full response content using the FSM.

    Returns (content, reasoning, tool_calls).
    Handles the truncated-thinking heuristic from v1.
    """
    events, fsm = collect_events(
        full_content, finish_reason, tools=tools,
        in_triple_backticks=in_triple, in_single_backtick=in_single,
        model_name=model_name,
        start_in_thinking=start_in_thinking,
        start_in_tool_call=start_in_tool_call,
        family=family,
    )

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []

    for event in events:
        if event.type == EventType.CONTENT:
            content_parts.append(event.data)
        elif event.type == EventType.REASONING:
            reasoning_parts.append(event.data)
        elif event.type == EventType.TOOL_CALL_END:
            if not is_summary:
                d = event.data
                raw_params = d.get("params", {})
                coerced: dict = {}
                for k, v in raw_params.items():
                    prop_schema = _get_prop_schema(d["name"], k, tools)
                    coerced[k] = coerce_value(v, prop_schema)
                tool_calls.append({
                    "id": d["id"],
                    "type": "function",
                    "function": {
                        "name": d["name"],
                        "arguments": json.dumps(coerced, ensure_ascii=False),
                    },
                })

    content = "".join(content_parts).strip() or None
    reasoning = "".join(reasoning_parts).strip() or None

    # Sufiksowa heurystyka regresyjna (Backward Suffix Heuristic) przy finish_reason == "length"
    if finish_reason == "length" and not tool_calls:
        # Przypadek A: Dane strukturalne wylądowały w reasoning (bo model nie domknął <think>)
        if reasoning and not content:
            r_strip = reasoning.strip()
            idx = -1
            # Szukamy najwcześniejszego wystąpienia znaczników strukturalnych od końca
            markers = ["```", "<tool_call", '{"', '["']
            for m in markers:
                pos = r_strip.rfind(m)
                if pos != -1 and (idx == -1 or pos < idx):
                    idx = pos
            
            if idx != -1:
                extracted = r_strip[idx:]
                new_reasoning = r_strip[:idx].strip()
                content = _repair_truncated_structure(extracted)
                reasoning = new_reasoning or None
                
        # Przypadek B: Dane wylądowały w content, ale zostały ucięte
        elif content and not reasoning:
            c_strip = content.strip()
            is_structured = any(c_strip.startswith(m) for m in ["```", "<", "{", "["])
            if is_structured:
                content = _repair_truncated_structure(content)
            else:
                # Klasyczna heurystyka truncated-thinking: ucięty zwykły tekst traktujemy jako reasoning
                reasoning = content
                content = None
        elif content:
            c_strip = content.strip()
            is_structured = any(c_strip.startswith(m) for m in ["```", "<", "{", "["])
            if is_structured:
                content = _repair_truncated_structure(content)

    return content, reasoning, tool_calls


# ---------------------------------------------------------------------------
# /v1/embeddings — embeddings endpoint with concurrency control
# ---------------------------------------------------------------------------

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """Embeddings endpoint with concurrency control (semaphore)."""
    config: GatewayConfig = request.app.state.config
    upstream: UpstreamClient = request.app.state.upstream
    audit: AuditLogger = request.app.state.audit
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    t_start = time.time()

    raw_body = await request.body()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }

    try:
        client_body = json.loads(raw_body or b"{}")
    except Exception:
        audit.log_error(request_id, "bad_json", "Invalid JSON in request body")
        return JSONResponse(status_code=400, content={"error": {"message": "bad JSON"}})

    # Log embedding request
    input_data = client_body.get("input")
    input_len = 0
    if isinstance(input_data, str):
        input_len = 1
    elif isinstance(input_data, list):
        input_len = len(input_data)

    _log.info("EMBEDDINGS START items=%d request_id=%s", input_len, request_id)

    sem_acquired = False
    try:
        # Concurrency control
        semaphore = request.app.state.concurrency_semaphore
        try:
            # BACKGROUND: nothing in the agentic loop reaches this endpoint
            # -- LODESTAR's memory search runs embed.py's own model on :18001 --
            # while a corpus walk fires thousands of these, which is exactly the
            # traffic that should yield to a turn somebody is watching.
            await asyncio.wait_for(semaphore.acquire(BACKGROUND),
                                   timeout=config.semaphore_timeout)
            sem_acquired = True
        except asyncio.TimeoutError:
            _log.error("EMBEDDINGS TIMEOUT waiting for semaphore request_id=%s", request_id)
            audit.log_error(request_id, "upstream_busy", "concurrency slot timeout")
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "upstream busy - try again later", "code": 503}},
            )

        # Forward the request to upstream /v1/embeddings
        if "model" in client_body:
            client_body["model"] = config.resolve_model(client_body["model"])

        # Forward to upstream
        try:
            r = await upstream.request(
                "POST", "/v1/embeddings",
                content=json.dumps(client_body).encode("utf-8"),
                headers={**headers, "content-type": "application/json"},
                timeout=httpx.Timeout(connect=30.0, read=90.0, write=30.0, pool=60.0)
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            _log.warning("Embeddings connection failed: %s. Retrying.", e)
            r = await upstream.request(
                "POST", "/v1/embeddings",
                content=json.dumps(client_body).encode("utf-8"),
                headers={**headers, "content-type": "application/json"},
                timeout=httpx.Timeout(connect=30.0, read=90.0, write=30.0, pool=60.0)
            )

        if r.status_code != 200:
            return _wrap_error_to_json(r.status_code, r.content)

        resp_json = r.json()
        _log.info("EMBEDDINGS SUCCESS request_id=%s duration_ms=%d", request_id, int((time.time() - t_start) * 1000))
        return JSONResponse(content=resp_json)

    except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ReadError) as e:
        _log.error("EMBEDDINGS TIMEOUT/READ ERROR error=%s request_id=%s", e, request_id)
        if audit:
            audit.log_error(request_id, "upstream_timeout", str(e))
        return JSONResponse(
            status_code=504,
            content={"error": {"message": f"proxy upstream timeout: {e}", "code": 504}},
        )
    except Exception as e:
        _log.error("EMBEDDINGS ERROR error=%s request_id=%s", e, request_id, exc_info=True)
        if audit:
            audit.log_error(request_id, "upstream_error", str(e))
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"proxy upstream error: {e}", "code": 502}},
        )
    finally:
        # `sem_acquired` is only ever True when `semaphore` is not None, but
        # that is an invariant of the code rather than of the types.
        if sem_acquired and semaphore is not None:
            semaphore.release()


# ---------------------------------------------------------------------------
# /v1/models — model listing with alias injection
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models(request: Request):
    """Proxy /v1/models with alias injection."""
    config: GatewayConfig = request.app.state.config
    upstream: UpstreamClient = request.app.state.upstream
    # Reported as unlogged by the v3 audit and again as O-2026-01: this route
    # could 502 and leave nothing at all behind. It has no request event of its
    # own, so the error carries its own id rather than orphaning one.
    audit: AuditLogger | None = getattr(request.app.state, "audit", None)
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    try:
        r = await upstream.request("GET", "/v1/models")
    except httpx.HTTPError as e:
        _log.error("MODELS UPSTREAM ERROR error=%s", e)
        if audit:
            audit.log_error(request_id, "upstream_error", str(e))
        return JSONResponse(status_code=502, content={"error": {"message": str(e)}})
    except Exception as e:
        _log.error("MODELS GATEWAY ERROR error=%s", e, exc_info=True)
        if audit:
            audit.log_error(request_id, "gateway_error", f"{type(e).__name__}: {e}")
        return JSONResponse(status_code=502, content={"error": {"message": str(e)}})

    if r.status_code != 200:
        if audit:
            audit.log_error(
                request_id, "upstream_status",
                f"HTTP {r.status_code}: "
                f"{r.content.decode('utf-8', errors='ignore').strip()[:500]}")
        return _wrap_error_to_json(r.status_code, r.content)

    try:
        models_resp = json.loads(r.content)

        aliases_to_inject = set()
        for alias, real_name in config.model_aliases.items():
            aliases_to_inject.add(alias)
            aliases_to_inject.add(real_name)

        # The registry, which is what a hot-swap rig actually serves. This
        # route knew only `model_aliases`, so on the eGPU preset two of four
        # models -- the vision one and Gemma -- were invisible to any client
        # that discovers models here. Invisible to OpenCode too, and harmless
        # there only because it reads its own config instead; the next client
        # would simply not see them. R-2026-05.
        #
        # A spec's own `n_ctx` wins over the upstream's reported window, for
        # the same reason `config.for_spec` prefers it: the upstream serves ONE
        # of these models at a time, so its `/props` describes whichever engine
        # currently holds the card and says nothing about the others.
        #
        # `vision` travels with the window, and for the same reason. A client
        # discovering models here cannot otherwise tell which of them will
        # read an image and which will answer about one confidently without
        # having seen it -- the failure that cost a day on 2026-09-13. The
        # gateway knows; saying nothing makes the client guess.
        #
        # `installed` travels with them for the same reason again. A preset may
        # declare a model whose weights are not on disk -- Gemma 4 is here
        # before anything was downloaded -- and asking for one is a 503 from
        # the engine supervisor. Two models can see; one of them cannot run.
        # A client picking "the first that declares vision" picks the wrong one
        # and reads the 503 as the gateway being broken.
        #
        # A FastStrata spec's `target` is the run config's FILE NAME
        # (`strata-qwen36-ud-q4_k_m.json`), which no client asks for, so only
        # its id is listed; a remote spec's target is the name that server
        # answers to, and is listed beside the id.
        spec_windows: dict[str, int] = {}
        spec_vision: dict[str, bool] = {}
        spec_installed: dict[str, bool] = {}
        home = config.polystrata_home()
        for spec in config.registry():
            names = [spec.id]
            if spec.target and spec.engine == ENGINE_REMOTE:
                names.append(spec.target)
            try:
                on_disk = spec.installed(home)
            except OSError:
                # Answering "no" about an unreadable path would tell a client
                # to avoid a model that may be perfectly fine; the gateway
                # says nothing rather than guessing.
                on_disk = True
            for name in names:
                aliases_to_inject.add(name)
                if spec.n_ctx:
                    spec_windows[name] = spec.n_ctx
                spec_vision[name] = spec.vision
                spec_installed[name] = on_disk

        existing_data = models_resp.get("data", [])
        real_meta = None
        context_window = None

        if existing_data:
            for m in existing_data:
                m_meta = m.get("meta")
                if m_meta and "n_ctx" in m_meta:
                    m["context_window"] = m_meta["n_ctx"]
                    if real_meta is None:
                        real_meta = m_meta
                        context_window = m_meta["n_ctx"]
                elif m.get("id") in spec_windows:
                    # A model the UPSTREAM already lists, which the registry
                    # also knows. Without this the same rig answered two ways:
                    # `qwen3.8-27b-vision` carried its window because the
                    # gateway injected it, while `qwen3.8-27b` carried none
                    # because the ExLlamaV3 server lists it itself and reports
                    # no `meta`. Measured against the live process on
                    # 2026-09-13; two of four registry models came back with
                    # `context_window: null`.
                    #
                    # The upstream's own `meta.n_ctx` still wins where it
                    # exists: that describes the engine actually loaded, and
                    # this branch only fills a gap rather than overriding an
                    # answer.
                    m["context_window"] = spec_windows[m["id"]]
                if m.get("id") in spec_vision:
                    m["vision"] = spec_vision[m["id"]]
                    m["installed"] = spec_installed[m["id"]]

        for model_id in sorted(aliases_to_inject):
            if any(m.get("id") == model_id for m in existing_data):
                continue
            alias_model = {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "proxy-alias",
            }
            if real_meta:
                alias_model["meta"] = real_meta
            window = spec_windows.get(model_id) or context_window
            if window:
                alias_model["context_window"] = window
            if model_id in spec_vision:
                alias_model["vision"] = spec_vision[model_id]
                alias_model["installed"] = spec_installed[model_id]
            existing_data.append(alias_model)
        models_resp["data"] = existing_data

        # Legacy 'models' list
        existing_models = models_resp.get("models", [])
        for model_id in sorted(aliases_to_inject):
            if any(m.get("name") == model_id for m in existing_models):
                continue
            existing_models.append({
                "name": model_id,
                "model": model_id,
                "modified_at": "",
                "size": "",
                "digest": "",
                "type": "model",
                "description": "(alias)",
                "tags": [""],
                "capabilities": ["completion"],
                "parameters": "",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "",
                    "families": [""],
                    "parameter_size": "",
                    "quantization_level": "",
                },
            })
        models_resp["models"] = existing_models

        return JSONResponse(content=models_resp)
    except Exception as e:
        _log.error("Failed to inject aliases into /v1/models: %s", e, exc_info=True)
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))


# ---------------------------------------------------------------------------
# Passthrough for other endpoints
# ---------------------------------------------------------------------------

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def passthrough(path: str, request: Request):
    """Forward an allowlisted path to upstream; 404 everything else.

    The route pattern still matches everything so that unknown paths get a
    clean 404 instead of falling through, but only the paths in
    `config.passthrough_allowlist` are actually proxied. Anything else would
    bypass the whole request pipeline (LODESTAR, FSM, tool validation), which
    is exactly what an attacker would reach for.
    """
    upstream: UpstreamClient = request.app.state.upstream
    config: GatewayConfig = request.app.state.config

    normalized = "/" + path.strip("/")
    if normalized not in config.passthrough_allowlist:
        _log.warning("PASSTHROUGH BLOCKED method=%s path=%s (not in allowlist)",
                     request.method, normalized)
        return JSONResponse(
            status_code=404,
            content={"error": {"message": f"path not routed: {normalized}", "code": 404}},
        )

    method = request.method
    raw_body = await request.body()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }

    try:
        r = await upstream.request(method, "/" + path, content=raw_body,
                                   headers=headers, params=request.query_params)
    except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ReadError) as e:
        _log.error("PASSTHROUGH TIMEOUT path=%s error=%s", path, e)
        return JSONResponse(
            status_code=504,
            content={"error": {"message": f"proxy upstream timeout: {e}", "code": 504}},
        )
    except httpx.HTTPError as e:
        _log.error("PASSTHROUGH UPSTREAM ERROR path=%s error=%s", path, e, exc_info=True)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"proxy upstream error: {e}", "code": 502}},
        )
    except Exception as e:
        # Same split as _non_stream_completion: what is left after every httpx
        # error is the gateway's own fault, and saying "proxy upstream error"
        # about it points the reader at the wrong machine (O-2026-02).
        _log.error("PASSTHROUGH GATEWAY ERROR path=%s error=%s", path, e, exc_info=True)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"gateway error: {type(e).__name__}: {e}",
                               "code": 502}},
        )

    if r.status_code >= 400:
        return _wrap_error_to_json(r.status_code, r.content)

    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/octet-stream"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_error_to_json(status_code: int, content: bytes) -> JSONResponse:
    err_msg = content.decode("utf-8", errors="ignore").strip()
    try:
        err_json = json.loads(err_msg)
        if not isinstance(err_json, dict) or "error" not in err_json:
            err_json = {"error": {"message": err_msg, "code": status_code}}
    except Exception:
        err_json = {"error": {"message": err_msg, "code": status_code}}
    return JSONResponse(status_code=status_code, content=err_json)


def is_summary_request(messages: list[dict]) -> bool:
    """True when the client is asking for a context compaction, not a task.

    A named function rather than an inline `any(...)` so the behaviour
    scenarios can assert it: the previous version of this check looked for
    "context summarization assistant" while OpenCode writes "agent", was wrong
    for the life of the file, and nothing noticed — the audit event recorded
    the flag eighty lines before it was computed, so every request in the log
    claimed False. An offline scenario over a recorded compaction is the
    cheapest thing that could have caught it (T-2026-01).
    """
    return summary_request_kind(messages) is not None


#: A compaction whose system prompt is the summarizer's own (OpenCode 1.x).
SUMMARY_LEGACY = "legacy"
#: A compaction asked for in the last user turn, on the agent's own system
#: prompt and tools (OpenCode 2.0) -- whose prefix must not be touched.
SUMMARY_ANCHORED = "anchored"


def summary_request_kind(messages: list[dict]) -> str | None:
    """Which kind of compaction this is, or None for ordinary traffic.

    The two kinds want different handling. A legacy compaction already has a
    prompt of its own, so summary mode may strip the tools from it. An
    anchored one kept the agent's prompt precisely so the engine reuses the
    cached prefix -- 13.9 s to the first token at 104k instead of minutes on
    this rig -- and stripping anything from it would throw that away.
    """
    if any(marker in _message_text(m)
           for m in messages
           if m.get("role") in ("system", "user")
           for marker in client_contract.summary_markers()):
        return SUMMARY_LEGACY
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last_user is not None:
        text = _message_text(last_user)
        if any(marker in text for marker in client_contract.anchored_summary_markers()):
            return SUMMARY_ANCHORED
    return None


def _append_to_last_user(messages: list[dict], text: str) -> bool:
    """Add `text` to the end of the last user message, whatever its shape."""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, list):
            m["content"] = [*content, {"type": "text", "text": text}]
        else:
            m["content"] = str(content or "") + text
        return True
    return False


def _message_text(msg: dict) -> str:
    """Lowercased text of a message, whatever shape the client used.

    Anything reading `content` BEFORE normalization has to survive a
    content-part list — OpenCode sends user messages that way — and normalizer
    only flattens them at step 2. A bare `.lower()` here raises AttributeError
    ahead of the audit event, so the request 500s and leaves no trace at all.
    """
    content = msg.get("content")
    if isinstance(content, str):
        return content.lower()
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
        return '\n\n'.join(parts).lower()
    return ""


def _messages_bytes(messages: list[dict]) -> int:
    """Total size of the message contents, for the spill audit event."""
    return sum(
        len(str(m.get("content") or "").encode("utf-8")) for m in messages
    )


def _args_from_chunks(chunks: list[bytes] | None) -> str | None:
    """Recover the arguments of a synthesized call from the SSE it produced.

    The stubborn-guide and Phase-D paths build their call straight into SSE
    chunks, so the arguments exist only on the wire. They are worth reading
    back — the embed search query in particular is derived from the user's
    text and is the one synthesized value that varies — and parsing our own
    emitter's output is cheaper than threading a second return value through
    the synthesis helpers and their tests.
    """
    if not chunks:
        return None
    parts: list[str] = []
    for raw in chunks:
        for line in raw.decode("utf-8", errors="replace").splitlines():
            if not line.startswith("data: "):
                continue
            body = line[6:].strip()
            if not body or body == "[DONE]":
                continue
            try:
                chunk = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                continue
            for choice in chunk.get("choices") or []:
                for tc in (choice.get("delta") or {}).get("tool_calls") or []:
                    args = (tc.get("function") or {}).get("arguments")
                    if args:
                        parts.append(str(args))
    return "".join(parts) or None


def _extract_workspace_dir(messages: list[dict]) -> str | None:
    return normalizer.workspace_from_messages(messages)
