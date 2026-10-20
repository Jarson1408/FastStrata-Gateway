"""
Recognise tool calls that can destroy work, before the client runs them.

`shell_safety.py` guards the commands the *gateway* fabricates. Nothing looks
at the commands the *model* emits: they go straight to the client, which runs
them with the user's own permissions. That was tolerable while the gateway ran
on the desk of the person who wrote it. It is not tolerable in an appliance
someone else switches on.

This is a classifier, not a sandbox, and the difference matters enough to say
plainly: the gateway does not execute tools, so it cannot confine anything. It
can only recognise a call and either say so in the log or refuse to pass it on.
Real confinement would mean running the tools here, which is a different
product (see the deepseek-harness `sandbox` package for what that actually
takes: bwrap, Landlock, Seatbelt, Windows restricted tokens).

Matching is deliberately conservative. A false positive in `warn` mode costs a
log line; a false negative costs a repository. Where the two conflict, the
patterns err towards noticing.
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

from gateway.lodestar.tool_names import (CMD_KEYS, PATH_KEYS, WORKDIR_KEYS,
                                         get_workdir)

__all__ = ["OFF", "WARN", "CONFIRM", "MISSING_WORKDIR", "classify",
           "confirmation_message", "is_unrunnable", "strip_workdir",
           "workdir_is_missing"]

_log = logging.getLogger("gateway.destructive")

# How long to wait for the filesystem to answer "does this directory exist".
# A dead mapped drive or an unreachable UNC share can make that question take
# many seconds, and this runs in the request path.
_STAT_TIMEOUT = 0.5

OFF = "off"
WARN = "warn"
CONFIRM = "confirm"

# Label prefix for the one finding here that is not about destruction at all.
MISSING_WORKDIR = "a working directory that does not exist"

# (compiled pattern, what to call it in the log and to the model)
_COMMAND_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\brm\s+(?:-\w*\s+)*-\w*[rR]\w*f|\brm\s+(?:-\w*\s+)*-\w*f\w*[rR]"),
     "recursive force delete (rm -rf)"),
    (re.compile(r"\bgit\s+reset\s+--hard\b"), "git reset --hard"),
    (re.compile(r"\bgit\s+clean\s+-\w*[dfx]"), "git clean"),
    (re.compile(r"\bgit\s+push\s+(?:[^\n]*\s)?(?:--force\b|--force-with-lease\b|-f\b)"),
     "force push"),
    (re.compile(r"\bgit\s+branch\s+(?:[^\n]*\s)?-D\b"), "force branch delete"),
    (re.compile(r"\bRemove-Item\b(?=[^\n]*-Recurse)(?=[^\n]*-Force)", re.IGNORECASE),
     "Remove-Item -Recurse -Force"),
    (re.compile(r"\b(?:rmdir|rd)\s+/s\b", re.IGNORECASE), "rmdir /s"),
    (re.compile(r"\bdel\s+(?:/\w+\s+)*/s\b", re.IGNORECASE), "del /s"),
    (re.compile(r"\bformat\s+[a-z]:", re.IGNORECASE), "disk format"),
    (re.compile(r"\bmkfs(?:\.\w+)?\b"), "filesystem creation"),
    (re.compile(r"\bdd\s+[^\n]*\bof=/dev/"), "raw device write"),
    (re.compile(r"\bDROP\s+(?:TABLE|DATABASE|SCHEMA)\b", re.IGNORECASE),
     "destructive SQL"),
    (re.compile(r"\bTRUNCATE\s+TABLE\b", re.IGNORECASE), "destructive SQL"),
    (re.compile(r"\bshutdown\b|\breboot\b", re.IGNORECASE), "host shutdown"),
)

# Tools whose path argument is a write target worth bounding to the workspace.
_WRITE_TOOLS = frozenset({
    "write", "write_file", "write_to_file", "edit", "apply_patch",
    "replace_file_content",
})


def _command_of(params: dict) -> str:
    for key in CMD_KEYS:
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _path_of(params: dict) -> str:
    for key in PATH_KEYS:
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _escapes_workspace(path: str, workspace_dir: str | None) -> bool:
    """True if `path` resolves outside the workspace root.

    Unresolvable input is treated as inside: a path the gateway cannot make
    sense of is not evidence of anything, and guessing would block ordinary
    edits on clients that pass relative paths.
    """
    if not workspace_dir or not path:
        return False
    try:
        root = Path(workspace_dir).resolve()
        target = Path(path)
        if not target.is_absolute():
            return False  # relative paths are resolved by the client, inside it
        return root not in target.resolve().parents and target.resolve() != root
    except (OSError, ValueError):
        return False


def _isdir_within(path: Path, timeout: float) -> bool | None:
    """Does `path` exist as a directory? None means the question timed out.

    `os.path.isdir` on a disconnected mapped drive or an unreachable UNC share
    blocks while Windows tries to reconnect — seconds, in the middle of a
    request. So the answer is fetched on a throwaway thread and abandoned if it
    does not come back. None is the honest result, and every caller treats it
    as permission: the gateway blocks only what it KNOWS is wrong.
    """
    result: list[bool] = []

    def probe() -> None:
        try:
            result.append(path.is_dir())
        except OSError:
            pass

    worker = threading.Thread(target=probe, daemon=True)
    worker.start()
    worker.join(timeout)
    if not result:
        _log.debug("workdir check timed out or errored for %s", path)
        return None
    return result[0]


def workdir_is_missing(params: dict, workspace_dir: str | None = None) -> str:
    """The working directory this call names, if it does not exist. Else "".

    Deliberately NOT a workspace check. A directory outside the workspace may
    be exactly what the user asked the model to look at, and the gateway has no
    way to tell that from a mistake — the client asks the human, which is the
    only party that knows. What the gateway can settle is a matter of fact: a
    working directory that is not on disk. Nothing legitimate produces one, and
    the call was going to fail anyway (Windows refuses to spawn a process in a
    missing directory: WinError 267), so catching it here costs nothing and
    turns an opaque failure into a sentence the model can act on.

    Returns "" whenever the answer is not a definite no — no workdir given, a
    relative one, a UNC path, or a filesystem that did not answer in time.
    """
    workdir = get_workdir(params)
    if not workdir:
        return ""

    # UNC paths are never probed: reaching a file server to answer a question
    # nobody asked is exactly the latency this guard must not introduce.
    if workdir.startswith("\\\\") or workdir.startswith("//"):
        return ""

    try:
        target = Path(workdir)
        if not target.is_absolute():
            # Relative to the workspace, which is where the client will run it.
            # Without a workspace there is nothing to resolve against, so the
            # gateway cannot know, and does not guess.
            if not workspace_dir:
                return ""
            target = Path(workspace_dir) / target
    except (OSError, ValueError):
        return ""

    exists = _isdir_within(target, _STAT_TIMEOUT)
    if exists is False:
        return workdir
    return ""


def classify(name: str, params: dict, workspace_dir: str | None = None) -> str | None:
    """Name the destructive property of this call, or None if it has none."""
    if not isinstance(params, dict):
        return None

    command = _command_of(params)
    if command:
        for pattern, label in _COMMAND_PATTERNS:
            if pattern.search(command):
                return label

    if name.lower() in _WRITE_TOOLS:
        path = _path_of(params)
        if _escapes_workspace(path, workspace_dir):
            return f"write outside the workspace ({path})"

    # Checked for any tool that names one, not just shells: whatever runs the
    # call, a working directory that is not there cannot be entered.
    missing = workdir_is_missing(params, workspace_dir)
    if missing:
        return f"{MISSING_WORKDIR} ({missing})"

    return None


def strip_workdir(params: dict) -> dict:
    """A copy of `params` with every working-directory argument removed.

    The repair for a working directory the model invented. Removing it is not
    a guess about intent: the client runs commands in the workspace by default,
    so a stripped call is the call the model meant, minus a detail it got
    wrong. The alternative — refusing the call — ends the session, which is a
    far larger harm than a corrected argument (see the note on error_chunk in
    app.py).

    Nothing is hidden by this. The repair is logged as a warning and recorded
    in the audit with its own reason, so `session_report.py` still shows a
    model that has lost track of where it is.
    """
    return {k: v for k, v in params.items() if k not in WORKDIR_KEYS}


def is_unrunnable(label: str | None) -> bool:
    """True for a label naming a call that cannot execute at all.

    Such a call is REPAIRED rather than blocked, in every mode but `off`, and
    that is the difference between this finding and the rest of the module.
    Every other pattern here describes a call that WILL work, which is exactly
    the problem, and only a human can say whether it was intended — so `warn`
    passes them on and `confirm` stops to ask. A working directory that is not
    on disk is neither: the OS refuses to spawn a process there (WinError
    267), so forwarding it is pointless, and refusing it outright proved worse
    than pointless. See strip_workdir.

    `off` still disables everything, because a switch that cannot be switched
    off is not a switch.
    """
    return label is not None and label.startswith(MISSING_WORKDIR)


def confirmation_message(label: str) -> str:
    """What the model is told when a call is held back for confirmation."""
    return (
        f"Blocked: this call was recognised as {label}. "
        "It was not executed. Explain to the user what you intend to do and "
        "why, and continue only after they agree."
    )
