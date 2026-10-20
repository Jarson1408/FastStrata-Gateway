"""Stop the model reissuing a call it has already run.

From a real session (2026-09-01, ACME_HW). The model ran this, three times in
a row, minutes apart:

    Get-Content "...\\spill\\afa746e3e117fc73.txt" | Select-String -Pattern "..."

Every one of them succeeded and returned the same 4,773 characters of real
content. Nothing failed, nothing was empty, and the answer was already sitting
in the conversation twice over. OpenCode's own `doom_loop` detector caught it —
but all a client can do is ask the user, which puts a human in the loop for a
problem the gateway can see and settle itself.

This is not a bug in the gateway, unlike most of what preceded it. It is the
model failing to notice it already has what it is asking for, and it is the
supervisor-level loop detection that both of the research reports put near the
top of their recommendations.

## What counts as a loop

Same tool, same canonical arguments, already EXECUTED (a result came back),
`threshold` times or more — and no write or edit in between. That last
condition is what keeps ordinary work out of it: re-running a test suite after
a fix is the same command twice on purpose, and the fix is the thing that makes
it a different question.

## What happens then

The call is suppressed and the model is told, in content. Not an error object —
that ends the client's session (see the note on error_chunk in app.py) — and
not a refusal to be argued with. It is told the exact number of times it has
run the command, that the output is above, and to do something else or say
what is blocking it.
"""
from __future__ import annotations

import json
import logging

from gateway import prompts
from gateway.lodestar.tool_names import EDIT_TOOLS, SHELL_CANDIDATES, parse_args

__all__ = [
    "DEFAULT_THRESHOLD",
    "FAST_THRESHOLD",
    "COMMAND_NOT_FOUND_PATTERNS",
    "count_identical_calls",
    "count_identical_calls_with_status",
    "check_call_loop",
    "looks_like_command_failure",
    "loop_message",
]

_log = logging.getLogger("gateway.loop_breaker")

# Three, not two, for routine commands (e.g. polling or retry).
DEFAULT_THRESHOLD = 3

# Two for unrecognized commands (CommandNotFoundException, missing executable, etc.).
# Repeating an invalid command that shell does not recognize is 100% futile.
FAST_THRESHOLD = 2

COMMAND_NOT_FOUND_PATTERNS = (
    "commandnotfoundexception",
    "is not recognized as the name of a cmdlet",
    "is not recognized as an internal or external command",
    "command not found",
    "not recognized as an operable program",
    "no such file or directory",
)

# How much of a tool result may carry the failure signal.
#
# A command that failed to start says so and stops; a command that ran prints
# its output. So the phrase near the TOP of a result is a status, and the same
# phrase two kilobytes in is content. 400 characters is several lines of shell
# error — enough for a PowerShell error block, which is verbose and spans the
# message plus a `+ CategoryInfo` line — and far short of any real output.
_NOT_FOUND_SCAN_CHARS = 400


def looks_like_command_failure(tool_name: str, result_text: str) -> bool:
    """Did this tool result come from a command the shell could not run?

    Two conditions, and the audit finding A-2026-02 is what both are for.
    Matching the patterns against the WHOLE result of ANY tool made a
    successful read look like a failure: `no such file or directory` is the
    content of every log, traceback and error-handling doc in the repository,
    so reading `gateway.log` twice dropped the loop threshold to
    FAST_THRESHOLD and the model was told a file it had just read did not
    exist. Measured by execution, not by reading:

        check_call_loop('read', {'filePath': '.../gateway.log'}, msgs)
        -> (True, 2, 2, True)      # both reads had SUCCEEDED

    - **Shell tools only.** For `read` or `glob` the phrase is data; the tool
      reports its own failure differently. Only for a shell call is the
      command's stderr the result.
    - **Near the start only.** `Get-Content x.log | Select-String error` is a
      shell call, succeeds, and returns thousands of characters that may well
      contain the phrase — and it is the exact call that produced the doom
      loop this module was written for. Position is what separates the two.
    """
    if (tool_name or "").lower() not in SHELL_CANDIDATES:
        return False
    head = (result_text or "")[:_NOT_FOUND_SCAN_CHARS].lower()
    return any(p in head for p in COMMAND_NOT_FOUND_PATTERNS)


def _canonical(params: object) -> str:
    """A stable key for a set of arguments, order-independent."""
    if isinstance(params, dict):
        try:
            return json.dumps(params, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return repr(sorted(params.items(), key=lambda kv: kv[0]))
    return repr(params)


def count_identical_calls_with_status(name: str, params: dict, messages: list[dict]) -> tuple[int, bool]:
    """How many times this exact call already ran, and whether any of those
    runs was a shell command the shell could not execute.

    Only calls with a matching tool RESULT are counted. A call the gateway
    suppressed, or one the user declined, never executed — holding those
    against the model would punish it for the gateway's own interventions.

    The second value is decided by `looks_like_command_failure`, which is
    narrower than "the result mentions an error"; see the note there.

    Returns (count, had_not_found).
    """
    if not name or not messages:
        return 0, False
    key = (name.lower(), _canonical(params))

    # Everything before the most recent write/edit is a different question.
    start = 0
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            if (fn.get("name") or "").lower() in EDIT_TOOLS:
                start = i + 1

    tool_results: dict[str, str] = {
        str(m["tool_call_id"]): str(m.get("content") or "")
        for m in messages[start:]
        if m.get("role") == "tool" and m.get("tool_call_id")
    }

    count = 0
    had_not_found = False
    for msg in messages[start:]:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            call_id = call.get("id")
            if call_id not in tool_results:
                continue
            fn = call.get("function") or {}
            if ((fn.get("name") or "").lower(), _canonical(parse_args(fn))) == key:
                count += 1
                if looks_like_command_failure(name, tool_results[call_id]):
                    had_not_found = True
    return count, had_not_found


def count_identical_calls(name: str, params: dict, messages: list[dict]) -> int:
    """How many times this exact call already ran, since the last edit."""
    count, _ = count_identical_calls_with_status(name, params, messages)
    return count


def check_call_loop(name: str, params: dict, messages: list[dict]) -> tuple[bool, int, int, bool]:
    """Check if a tool call is in a loop.

    Returns (is_loop, count, threshold, had_not_found).
    """
    count, had_not_found = count_identical_calls_with_status(name, params, messages)
    thresh = FAST_THRESHOLD if had_not_found else DEFAULT_THRESHOLD
    return count >= thresh, count, thresh, had_not_found


def loop_message(name: str, count: int, not_found: bool = False) -> str:
    """What the model is told. The count is the part that lands.

    Carries the same `[SYSTEM GUARD]` marker as the other guards, because this
    text can end up being the whole turn — when every call in it was
    suppressed, the model has nothing else to say, and for a subagent that
    turn becomes the answer handed back to its parent. The marker is what
    tells a reader it came from the gateway rather than from the model.

    The text lives in `gateway/prompts/`, not here (C-2026-03). A literal in
    `.py` cannot be tied to a revision: `prompts_version` in the audit log is
    what connects a change in model behaviour to the edit that caused it, and
    it does not move for a string in code.
    """
    return prompts.get(
        "loop_guard_not_found" if not_found else "loop_guard",
        tool=name, count=count,
    )
