"""Writing the gateway's model list into OpenCode's own configuration.

OpenCode does not discover models from `/v1/models` for a custom provider: the
list in its config IS the list, and every entry carries a context limit the
client uses to decide when to compact. So a rig whose models changed has two
places that must agree, and until now the second one was edited by hand -- with
the result this repository already documents for the FIRST one, that the two
drifted and the drift was silent.

WHAT MAKES THIS DELICATE, AND IT IS NOT THE JSON

The live config on this machine contains an API key for an unrelated MCP
server. A writer that regenerated the file would destroy it, and a writer that
regenerated "just the provider block" would destroy everything else. So this
module only ever MERGES into a parsed copy, touches exactly one key --
`provider.<id>.models` -- and keeps a `.bak` of what was there before.

**Comments are the part that cannot be merged.** `.jsonc` permits them, `json`
does not emit them, and a round trip therefore deletes every one. That is not
something to discover afterwards, so `plan_export` counts them and the caller
is expected to ask before `write_export` runs. Silently deleting somebody's
notes to save them one edit is a bad trade at any ratio.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from gateway.model_registry import ModelSpec

_log = logging.getLogger("gateway.opencode_export")

#: The provider key this gateway owns in OpenCode's config. Everything under
#: it is ours to rewrite; everything outside it is not. Its own name, not
#: Local-LLM-Gateway's `local-gateway`, so a PC with both keeps both lists.
DEFAULT_PROVIDER_ID = "faststrata"
DEFAULT_PROVIDER_NAME = "FastStrata"
DEFAULT_NPM = "@ai-sdk/openai-compatible"

#: 30 minutes, in the milliseconds OpenCode wants. A cold 13.6 GB model load
#: plus a long agentic turn is minutes, not seconds, and a client that gives up
#: at the default leaves the gateway generating into a socket nobody reads.
DEFAULT_TIMEOUT_MS = 1_800_000


@dataclass
class ExportPlan:
    """What an export would do, before it does it."""
    path: Path
    exists: bool
    comments: int
    provider_id: str
    models: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class ExportResult:
    path: Path
    backup: Path | None
    models: list[str]
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


# ---------------------------------------------------------------------------
# Finding the file
# ---------------------------------------------------------------------------

def candidate_paths(home: Path | None = None) -> list[Path]:
    """Every place OpenCode reads its global config from, best first.

    `~/.opencode/` is deliberately NOT here. OpenCode 1.x read it as a global
    config directory and the provider block lived there; 2.0 reads only
    `~/.config/opencode/` (`opencode debug config` lists the sources), so a
    provider declared in `~/.opencode/` simply vanished from the model list
    on upgrade. Offering it as a candidate would make "the file that declares
    the provider" pick the one file 2.0 never opens. `~/.config/opencode/` is
    read by 1.x as well, so nothing is lost on the older client.
    """
    base = home or Path(os.path.expanduser("~"))
    return [
        base / ".config" / "opencode" / "opencode.jsonc",
        base / ".config" / "opencode" / "opencode.json",
    ]


def pick_config_path(home: Path | None = None,
                     provider_id: str = DEFAULT_PROVIDER_ID) -> Path | None:
    """The config to write to: the one that already knows this provider.

    Falling back to "the first file that exists" would be wrong on a machine
    with several -- this one has three, and only one of them carries the
    provider block. Writing the models into a file OpenCode does read but
    which has no provider would produce a config that looks complete and
    serves nothing.
    """
    existing = [p for p in candidate_paths(home) if p.is_file()]
    for path in existing:
        data, _ = load_jsonc(path)
        if isinstance(data, dict) and provider_id in (data.get("provider") or {}):
            return path
    return existing[0] if existing else None


# ---------------------------------------------------------------------------
# Reading JSONC
# ---------------------------------------------------------------------------

_COMMENT_RE = re.compile(
    r'"(?:\\.|[^"\\])*"'      # a string, matched first so its contents survive
    r"|/\*.*?\*/"             # /* block */
    r"|//[^\n]*",             # // line
    re.DOTALL,
)


def strip_comments(text: str) -> tuple[str, int]:
    """(JSON without comments, how many were removed).

    The string alternative comes first in the pattern on purpose: without it a
    URL inside a value -- `"https://opencode.ai/config.json"` -- is a `//`
    comment and the rest of that line disappears. Which is exactly what the
    live config on this machine is full of.
    """
    removed = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal removed
        token = match.group(0)
        if token.startswith(('"', )):
            return token
        removed += 1
        return " " * len(token) if "\n" not in token else "\n" * token.count("\n")

    return _COMMENT_RE.sub(_replace, text), removed


def load_jsonc(path: Path) -> tuple[dict[str, Any] | None, int]:
    """(parsed config, comment count). (None, 0) when it cannot be read."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _log.warning("Cannot read %s: %s", path, exc)
        return None, 0
    cleaned, comments = strip_comments(text)
    # Trailing commas are legal in JSONC and fatal to json.loads.
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        _log.warning("Cannot parse %s: %s", path, exc)
        return None, comments
    return (data if isinstance(data, dict) else None), comments


# ---------------------------------------------------------------------------
# The merge
# ---------------------------------------------------------------------------

def models_block(specs: Iterable[ModelSpec]) -> dict[str, Any]:
    """The `models` map OpenCode wants, from the registry.

    `limit.context` is the model's OWN window rather than the gateway's prune
    limit, and the difference matters: the client uses this to decide when to
    compact, and the gateway prunes at `n_ctx - context_reserve` underneath it.
    Telling the client the smaller number would make it compact early and lose
    history the gateway was still able to carry.

    A model with no `n_ctx` is exported without a limit rather than with a
    guessed one. OpenCode then uses its own default, which is a visible
    behaviour; a number invented here would not be.

    `modalities` is what actually decides whether an image reaches the model,
    and `attachment` -- which this function wrote alone until 2026-09-14 -- is
    not. OpenCode derives `capabilities.input.image` from
    `modalities.input.includes("image")` and strips every image part of a
    message whose model lacks it, replacing the payload with the text
    `ERROR: Cannot read "<file>" (this model does not support image input).`
    So a vision model exported with `attachment: true` and no `modalities`
    was served images the CLIENT had already removed: the gateway never saw
    one (no MULTIMODAL line), llama-server never encoded one, and the model
    read a note saying it cannot see and reported exactly that. The failure
    reads as a broken vision pipeline and is a missing key in the client's
    config -- see AGENTS.md, "The Model List Is Two Places".

    It is declared for EVERY model, not only the ones that can see. The
    derivation is `modalities?.input?.includes(x) ?? <default>`, and `??`
    falls through on absence only: a declared list that omits "image" answers
    false, while no list at all leaves the question to a default this gateway
    does not control. Saying so for a text model is how it stays false on
    purpose rather than by luck.
    """
    out: dict[str, Any] = {}
    for spec in specs:
        entry: dict[str, Any] = {"name": spec.display_name}
        if spec.n_ctx:
            entry["limit"] = {"context": spec.n_ctx, "output": spec.max_output}
        modalities = ["text", "image"] if spec.vision else ["text"]
        entry["modalities"] = {"input": modalities, "output": ["text"]}
        # No `attachment`: OpenCode 2.0 drops it with "omitted unsupported
        # legacy setting" on every load, and `modalities` is what decides.
        out[spec.id] = entry
    return out


def merge_provider(config: dict[str, Any], specs: Iterable[ModelSpec],
                   base_url: str,
                   provider_id: str = DEFAULT_PROVIDER_ID) -> dict[str, Any]:
    """A copy of `config` with this provider's model list replaced. Pure.

    Replaced, not merged: a model removed from a preset has to disappear from
    the client too, or it stays selectable and fails at request time with a
    name the gateway no longer knows. Everything else in the provider block --
    and everything outside it -- is carried over untouched.
    """
    result = json.loads(json.dumps(config))  # deep copy through plain data
    providers = result.setdefault("provider", {})
    if not isinstance(providers, dict):
        providers = {}
        result["provider"] = providers

    provider = providers.get(provider_id)
    if not isinstance(provider, dict):
        provider = {"npm": DEFAULT_NPM, "name": DEFAULT_PROVIDER_NAME}
        providers[provider_id] = provider

    provider.setdefault("npm", DEFAULT_NPM)
    provider.setdefault("name", DEFAULT_PROVIDER_NAME)
    options = provider.setdefault("options", {})
    if not isinstance(options, dict):
        options = {}
        provider["options"] = options
    options["baseURL"] = base_url
    options.setdefault("timeout", DEFAULT_TIMEOUT_MS)

    provider["models"] = models_block(specs)
    return result


def base_url_for(port: int) -> str:
    return f"http://localhost:{port}/v1"


# ---------------------------------------------------------------------------
# Plan, then write
# ---------------------------------------------------------------------------

def plan_export(specs: Iterable[ModelSpec], path: Path | None = None,
                provider_id: str = DEFAULT_PROVIDER_ID,
                home: Path | None = None) -> ExportPlan:
    """What would change, so the caller can ask before anything is written."""
    specs = list(specs)
    target = path or pick_config_path(home, provider_id)
    if target is None:
        return ExportPlan(
            path=Path(""), exists=False, comments=0, provider_id=provider_id,
            error="No OpenCode configuration found. Looked in: "
                  + ", ".join(str(p) for p in candidate_paths(home)))

    if not target.is_file():
        return ExportPlan(path=target, exists=False, comments=0,
                          provider_id=provider_id,
                          models=[s.id for s in specs])

    data, comments = load_jsonc(target)
    if data is None:
        return ExportPlan(
            path=target, exists=True, comments=comments, provider_id=provider_id,
            error=f"{target} could not be parsed as JSON/JSONC. Nothing was "
                  "written — fix the file or point the export elsewhere.")

    current = ((data.get("provider") or {}).get(provider_id) or {}).get("models") or {}
    wanted = {s.id for s in specs}
    removed = sorted(k for k in current if k not in wanted)
    return ExportPlan(path=target, exists=True, comments=comments,
                      provider_id=provider_id,
                      models=[s.id for s in specs], removed=removed)


def write_export(specs: Iterable[ModelSpec], port: int,
                 path: Path | None = None,
                 provider_id: str = DEFAULT_PROVIDER_ID,
                 home: Path | None = None) -> ExportResult:
    """Write the model list into OpenCode's config, keeping a backup.

    Backup first, then a temp file and `os.replace`, which is the same shape
    `save_settings_to_file` uses and for the same reason: a crash halfway
    through must not leave the user with neither the old config nor a valid
    new one.
    """
    specs = list(specs)
    plan = plan_export(specs, path=path, provider_id=provider_id, home=home)
    if not plan.ok:
        return ExportResult(path=plan.path, backup=None, models=[],
                            error=plan.error)

    target = plan.path
    data: dict[str, Any] = {}
    backup: Path | None = None
    if target.is_file():
        loaded, _ = load_jsonc(target)
        data = loaded or {}
        # `.bak`, or `.bak-<date>-<time>` where a `.bak` is already there: a
        # backup somebody kept by hand (`opencode.json.bak` from months ago)
        # is theirs, and an export that replaced it would lose the one copy
        # of the config it was kept for.
        backup = target.with_suffix(target.suffix + ".bak")
        if backup.exists():
            backup = target.with_suffix(target.suffix + time.strftime(".bak-%Y%m%d-%H%M%S"))
        try:
            backup.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError as exc:
            return ExportResult(path=target, backup=None, models=[],
                                error=f"could not write the backup {backup}: {exc}")
    else:
        data = {"$schema": "https://opencode.ai/config.json"}

    merged = merge_provider(data, specs, base_url_for(port), provider_id)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(merged, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, target)
    except OSError as exc:
        return ExportResult(path=target, backup=backup, models=[],
                            error=f"could not write {target}: {exc}")

    _log.info("Exported %d model(s) to %s", len(specs), target)
    return ExportResult(path=target, backup=backup, models=[s.id for s in specs])


# ---------------------------------------------------------------------------
# From the command line: what FastStrata's setup runs
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """`python -m gateway.opencode_export [--write] [--drop-comments]`.

    The same export as the desktop app's button, for FastStrata's setup to run
    after it has written the models into settings.json. Prints one JSON object
    saying what would change (or what did), so the caller reads an answer
    rather than parsing prose.

    With no OpenCode config anywhere, the export CREATES
    `~/.config/opencode/opencode.json` -- a new install is the case this exists
    for. A `.jsonc` with comments is the one thing it will not do silently: it
    stops with exit code 3 and writes nothing, because a JSON round trip drops
    every comment (see the module docstring); `--drop-comments` is the caller
    having asked the user. Exit 0: done or nothing to do; 1: an error; 3: the
    comments.
    """
    import argparse

    from gateway.config import GatewayConfig

    ap = argparse.ArgumentParser(prog="python -m gateway.opencode_export",
                                 description="Write the gateway's models into OpenCode's config.")
    ap.add_argument("--write", action="store_true",
                    help="write the model list (default: only say what would change)")
    ap.add_argument("--drop-comments", action="store_true",
                    help="write a .jsonc even though its comments are lost (a .bak keeps them)")
    a = ap.parse_args(argv)

    config = GatewayConfig.from_env()
    specs = config.registry().specs
    path = pick_config_path() or candidate_paths()[-1]
    plan = plan_export(specs, path=path)
    summary: dict[str, Any] = {"path": str(plan.path), "exists": plan.exists,
                               "comments": plan.comments, "models": plan.models,
                               "removed": plan.removed, "written": False,
                               "backup": None, "error": plan.error}
    code = 0 if plan.ok else 1
    if a.write and plan.ok:
        if plan.comments and not a.drop_comments:
            summary["error"] = (f"{plan.path} has {plan.comments} comment(s) that writing "
                                "would delete; nothing was written")
            code = 3
        else:
            result = write_export(specs, config.port, path=plan.path)
            summary.update(written=result.ok, error=result.error,
                           backup=str(result.backup) if result.backup else None)
            code = 0 if result.ok else 1
    print(json.dumps(summary, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
