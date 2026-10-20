import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path

_log = logging.getLogger("gateway.gui_helpers")

# Top-level template files deployed alongside the .context tree
#: **What deploys is the PROTOCOL, and nothing else.**
#:
#: This used to carry `CLAUDE.md`, `ZANIM_GOTOWE.md`, `REGRESJE.md`,
#: `.gitattributes` and the two `normalize_git` scripts, and the `.context`
#: walk below copied that whole tree — including `audit/`, fourteen documents
#: of an audit playbook, and `rules/`, per-language editing conventions.
#:
#: Deploying those puts a body of somebody else's opinion into a stranger's
#: repository, in the one place guaranteed to be read: `.context/` is the tree
#: the model is told to work from. AGENTS.md is explicit that the audit set is
#: never injected and is started only when the USER asks for it — depositing it
#: unasked contradicts that at the other end. The git line-ending scripts are
#: housekeeping for THIS repository and have nothing to do with memory.
#:
#: `AGENTS.md` stays because it is the contract that tells an agent to follow
#: the protocol; without it the rest of the template is furniture.
#:
#: `CLAUDE.md` stays with it, and carries no content of its own — one
#: `@AGENTS.md` line. Claude Code reads `CLAUDE.md` and most other tools read
#: `AGENTS.md`, so a project that deploys only one of them is legible to one
#: family of tools and silent to the other. Two files with CONTENT would be two
#: versions of the truth; a forwarder is how you get compatibility without
#: that.
#:
#: The files that no longer deploy are still in `LODESTAR_template/` — this
#: stopped deploying them, it did not delete them, and the template's copies
#: differ from this repository's own.
_TEMPLATE_ROOT_FILES = ("AGENTS.md", "CLAUDE.md")

#: Paths inside the template's `.context/` that are this author's working
#: method rather than the memory protocol. Matched against the path relative
#: to `.context`, so a directory prunes everything under it.
_NOT_LODESTAR = ("audit", "TECHNICAL_DEBT_AUDIT.md")

#: Build litter. **The deploy copies from DISK, not from git**, so `.gitignore`
#: does nothing here -- and running the template's `embed.py` once leaves a
#: `__pycache__` that was then deployed into somebody's project: compiled
#: bytecode for whichever Python this machine happens to run, stale on arrival.
#: Seen on 2026-09-13, `embed.cpython-314.pyc` in a fresh deploy.
_LITTER_DIRS = ("__pycache__", ".pytest_cache", ".mypy_cache")
_LITTER_SUFFIXES = (".pyc", ".pyo")


def _is_not_lodestar(rel_path: Path) -> bool:
    """True for a template path that must not be deployed into a project."""
    parts = rel_path.parts
    if not parts:
        return False
    if parts[0] in _NOT_LODESTAR:
        return True
    if any(part in _LITTER_DIRS for part in parts):
        return True
    return rel_path.suffix in _LITTER_SUFFIXES


def deploy_lodestar_template(target_path_str: str) -> str:
    """Copy the LODESTAR_template files to the target workspace directory.

    Non-destructive: files that already exist in the target are NEVER
    overwritten. Re-deploying into a live project must not wipe its
    `active-context.md` or `semantic-memory/memory.db` (the previous merge
    implementation did exactly that).

    Returns:
        A message indicating success or details of the error.
    """
    target_path = Path(target_path_str).resolve()
    if not target_path.is_dir():
        return f"Target path is not a directory: {target_path_str}"

    # Source directory is at the root of local-llm-gateway. No fallback:
    # deploying the gateway's own .context (with its real memory) into a
    # user project would be far worse than an error message.
    gateway_root = Path(__file__).resolve().parent.parent
    template_dir = gateway_root / "LODESTAR_template"
    if not template_dir.is_dir():
        # A bare "not found" sends somebody looking for a bug when the answer
        # is that the supervisor needs no template: it reads three files in the
        # TARGET project, and the template only writes a first draft of them.
        return (
            "No LODESTAR template here, so there is nothing to copy.\n\n"
            "The supervisor does not need one: write .context/LODESTAR.md in "
            "your project and it takes over from there.\n"
            "README.md lists the three files it reads and what it does with "
            "each."
        )

    try:
        copied: list[str] = []
        skipped: list[str] = []

        def copy_missing(src_file: Path, dest_file: Path, label: str) -> None:
            if dest_file.exists():
                skipped.append(label)
                return
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest_file)
            copied.append(label)

        # 1. Copy the .context tree (skip-existing merge)
        src_context = template_dir / ".context"
        if src_context.is_dir():
            for root, _dirs, files in os.walk(src_context):
                rel_path = Path(root).relative_to(src_context)
                if _is_not_lodestar(rel_path):
                    continue
                for f in files:
                    if _is_not_lodestar(rel_path / f):
                        continue
                    copy_missing(Path(root) / f,
                                 target_path / ".context" / rel_path / f,
                                 str(Path(".context") / rel_path / f))

        # 2. Copy top-level template files
        for name in _TEMPLATE_ROOT_FILES:
            src = template_dir / name
            if src.is_file():
                copy_missing(src, target_path / name, name)

        if not copied and not skipped:
            return "No template files found to deploy."

        lines = [f"Successfully deployed template to {target_path}:"]
        lines += [f" + {f}" for f in copied]
        if skipped:
            lines.append(f"Kept {len(skipped)} existing file(s) untouched "
                         "(never overwritten):")
            lines += [f" = {f}" for f in skipped]
        if not copied:
            lines.insert(1, "(nothing new to copy - project already bootstrapped)")
        return "\n".join(lines)
    except Exception as e:
        _log.error("Failed to deploy Lodestar template to %s: %s", target_path, e)
        return f"Deployment failed: {e}"


LOG_COLOR_DEFAULT = "#e5e2e3"


def classify_log_line(text: str) -> tuple[str, bool]:
    """Return (display color, is_error_like) for a gateway log line.

    Error-like lines (ERROR/WARNING) additionally land in the dashboard's
    persistent error buffer so they survive the live view's line cap.
    """
    if "ERROR" in text:
        return "#ff1744", True     # Crimson Red
    if "WARNING" in text:
        return "#ffdb3c", True     # Accent Gold
    if "LODESTAR" in text:
        return "#cdbdff", False    # Purple/Indigo Accent
    if "INFO" in text:
        return "#00e676", False    # Neon Green
    return LOG_COLOR_DEFAULT, False


def get_startup_shortcut_path() -> Path:
    """Return the absolute path of the shortcut file in Windows Startup directory."""
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "faststrata-gateway.lnk"
    
    # Fallback/Linux/Mac
    return Path("~/.config/autostart/faststrata-gateway.desktop").expanduser()


def set_autostart(enabled: bool) -> bool:
    """Enable or disable autostart by managing the Startup folder shortcut."""
    shortcut_path = get_startup_shortcut_path()
    
    if not enabled:
        if shortcut_path.is_file():
            try:
                shortcut_path.unlink()
                _log.info("Removed startup shortcut: %s", shortcut_path)
                return True
            except Exception as e:
                _log.error("Failed to remove startup shortcut: %s", e)
                return False
        return True  # already disabled
        
    # Enable autostart: create shortcut
    if os.name != "nt":
        _log.warning("Autostart shortcut creation is only supported on Windows in this implementation.")
        return False
        
    try:
        shortcut_path.parent.mkdir(parents=True, exist_ok=True)
        
        # We need to target the pythonw runner or gui script
        pythonw_exe = Path(sys.executable).parent / "pythonw.exe"
        if not pythonw_exe.is_file():
            pythonw_exe = Path(sys.executable) # Fallback to python
            
        script_path = Path(__file__).resolve().parent / "gui.py"
        
        # Build powershell command to create shortcut
        cmd_parts = [
            f'$s = (New-Object -ComObject WScript.Shell).CreateShortcut("{shortcut_path}");',
            f'$s.TargetPath = "{pythonw_exe}";',
            '$s.Arguments = "-m gateway.gui";',
            f'$s.WorkingDirectory = "{script_path.parent.parent}";',
            '$s.WindowStyle = 7;', # Minimized/Hidden window style
            f'$s.IconLocation = "{pythonw_exe},0";',
            '$s.Save();'
        ]
        ps_cmd = " ".join(cmd_parts)
        
        subprocess.run(
            ["powershell", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            check=True
        )
        _log.info("Created startup shortcut: %s", shortcut_path)
        return True
    except Exception as e:
        _log.error("Failed to create startup shortcut: %s", e)
        return False


# ---------------------------------------------------------------------------
# Decisions taken out of the Qt widgets
#
# This project cannot test its GUI headlessly and says so. The consequence,
# found by the 2026-09-03 audits three times over, is that logic living inside
# a widget is logic nobody ever asked the hard question about: what does it do
# when the thing it reports on has failed? Everything below is that kind of
# decision, moved somewhere it can be asked.
# ---------------------------------------------------------------------------

def describe_save_outcome(
    settings_saved: bool,
    autostart_ok: bool,
    autostart_requested: bool,
) -> tuple[str, str]:
    """Return (level, message) for the Configuration tab's save dialog.

    `level` is "success" or "warning"; the caller picks the message box.

    Both inputs used to be discarded: `save_settings_to_file` returned None,
    `set_autostart` returned a bool nobody read, and the dialog said
    "Configuration saved and gateway restarted" either way. D-2026-01.

    `autostart_requested` matters because `set_autostart(False)` returning
    False is not a failure of anything the user asked for.
    """
    failures = []
    if not settings_saved:
        failures.append("settings.json could not be written")
    if autostart_requested and not autostart_ok:
        failures.append("the autostart shortcut could not be created")

    if not failures:
        return "success", "Configuration saved and gateway restarted."

    return "warning", (
        "The gateway was restarted, but "
        + " and ".join(failures)
        + ". It is running with the PREVIOUS settings — see gui.log."
    )


def describe_health_failure(exc: BaseException) -> dict:
    """Turn a failed /health poll into a payload that says what went wrong.

    Every failure used to become `{}`, so a stopped gateway, a busy port, a
    timeout and a malformed response lit the same red chip. Under S-2026-01 —
    two gateways racing for port 18000 — that was precisely the information
    whose absence made the diagnosis take a day: the light was red for a
    process that was running. O-2026-05.

    The shape keeps `upstream.status` != "ok" so the existing chip logic reads
    it as unhealthy without knowing about any of this.
    """
    import json as _json
    import urllib.error

    if isinstance(exc, urllib.error.URLError):
        reason = "unreachable"
    elif isinstance(exc, (_json.JSONDecodeError, ValueError)):
        reason = "bad_response"
    else:
        reason = "poll_failed"

    detail = str(exc) or type(exc).__name__
    if reason == "poll_failed":
        detail = f"{type(exc).__name__}: {detail}"

    return {"upstream": {"status": "error", "detail": detail},
            "reason": reason, "detail": detail}


def audit_log_path(config) -> Path:
    """Where the audit log lives, according to the configuration.

    The Sessions tab derived this from `gui.py`'s own location, so a
    configured `log_dir` showed "no audit log" for a log that existed.
    W-2026-01 — the surviving half of a finding whose main claim (that reading
    the log freezes the GUI) was measured away at 0.09 s.
    """
    return Path(config.log_dir) / "gateway_events.jsonl"


def describe_engine_memory(upstream: dict) -> str:
    """One tooltip line about the engine's recurrent checkpoint stash.

    Empty when the upstream does not report one -- llama-server never will, and
    a tooltip that says "unknown" is worse than no tooltip.

    This is the readout that decides `RECURRENT_CACHE_MB`. The ceiling is
    HOST RAM, which is what this machine runs out of before it runs out of
    VRAM, so the number to watch is how close `used` gets to `limit` during a
    long session: never close says the ceiling is more than this workload
    needs, and pinned at the limit says prefill is replaying recurrent layers
    it had already paid for.
    """
    used = (upstream or {}).get("recurrent_cache_used_mb")
    limit = (upstream or {}).get("recurrent_cache_limit_mb")
    if used is None or limit is None:
        return ""
    count = (upstream or {}).get("recurrent_checkpoints")
    held = f", {count} checkpoint(s)" if count is not None else ""
    line = (f"Recurrent state stash: {used:g} of {limit:g} MiB host RAM{held}. "
            f"It is a ceiling, not an allocation.")
    tier = (upstream or {}).get("kv_ram_tier_limit_mb")
    if tier:
        pinned = (upstream or {}).get("kv_ram_tier_pinned_mb")
        pages = (upstream or {}).get("kv_ram_tier_pages")
        restores = (upstream or {}).get("kv_ram_tier_restores")
        line += f"\nKV RAM tier: {tier:g} MiB"
        if pinned is not None:
            line += f", {pinned:g} MiB pinned"
        if pages is not None:
            line += f", {pages} page(s) held"
        if restores is not None:
            line += f", {restores} restored instead of prefilled"
    elif tier == 0:
        line += "\nKV RAM tier: off"
    mtp = (upstream or {}).get("mtp_draft_tokens")
    if mtp:
        line += f"\nMTP draft: {mtp} token(s)"
    ngram = (upstream or {}).get("ngram_match_min")
    if ngram is not None:
        line += ("\nN-gram draft: off" if not ngram
                 else f"\nN-gram draft: min match {ngram}")
    return line


def format_upstream_chip(upstream: dict) -> tuple[str, str, str]:
    """Return (label, color, tooltip) for the UPSTREAM status chip in the GUI.

    Handles 'ok', 'sleeping' (idle standby with free VRAM), and failure states.
    """
    status = (upstream or {}).get("status")
    if status == "ok":
        n_ctx = upstream.get("n_ctx")
        suffix = f" · ctx {n_ctx // 1000}k" if isinstance(n_ctx, int) else ""
        return f"UPSTREAM: OK{suffix}", "#00e676", describe_engine_memory(upstream)

    if status == "sleeping":
        vram = upstream.get("vram_free_gb")
        suffix = f" · {vram} GB free" if vram is not None else ""
        return (
            f"UPSTREAM: ASLEEP{suffix}",
            "#7c4dff",
            "The model is asleep and its VRAM is free. It wakes by itself in about 6 s on the next request.",
        )

    reason = str((upstream or {}).get("reason") or "")
    label = {
        "unreachable": "UNREACHABLE",
        "bad_response": "BAD REPLY",
        "poll_failed": "POLL FAILED",
    }.get(reason, "UNREACHABLE")
    detail = str((upstream or {}).get("detail") or "")
    return f"UPSTREAM: {label}", "#ff1744", detail
