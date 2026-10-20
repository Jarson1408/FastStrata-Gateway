"""
Everything the protocol reads from outside itself, in one place.

The state machine infers what happened from text it did not write:

  * error wording produced by the **client** ("Could not find oldString"),
  * headings and status words from the **workspace template**,
  * the output format of **embed.py**.

Those literals were scattered through state.py as inline tuples and regexes,
which made a real dependency look like an implementation detail. Changing
client, upgrading it, or translating the template silently changes protocol
behaviour — and in the worst direction: an unrecognised error message meant
`response_indicates_tool_error` returned False, the failed edit counted as a
committed Goal, and the task ran with no Goal at all. That is the 2026-07-09
13:24 incident, reachable again by nothing more than a client updating its
phrasing.

Two answers here, in order of preference:

1. **Stop asking the prose.** `write_landed_on_disk` checks whether the file
   actually contains what the call claimed to write. The gateway and the client
   share a filesystem — the enforcer already reads and writes these files — so
   this is ground truth, independent of any wording. Prose is the fallback for
   when the workspace is not reachable.

2. **Make the remaining literals configurable.** Every value below can be
   overridden from settings.json or the environment, so a localized template or
   a different client is a config change, not a patch.
"""
from __future__ import annotations

import functools
import os
import re
from pathlib import Path

__all__ = [
    "tool_error_markers",
    "active_context_heading",
    "search_result_re",
    "status_open_re",
    "status_closed_re",
    "new_content_keys",
    "summary_markers",
    "anchored_summary_markers",
    "stripped_media_re",
    "progress_log_heading",
    "progress_entry_re",
    "count_progress_entries",
    "is_known_error_marker",
    "write_landed_on_disk",
    "claimed_new_content",
]

# -- Defaults ----------------------------------------------------------------

# Failure wordings observed from OpenCode. An edit whose response matches did
# NOT change the file, so it must not advance the state machine.
_DEFAULT_ERROR_MARKERS = (
    "could not find oldstring",
    "oldstring not found",
    "must match exactly",
    "file has been modified since read",
    "error editing",
    "failed to write",
    "permission denied",
    "no such file",
)

# The client's own summarization system prompt. OpenCode says "You are a
# context summarization agent."; the gateway spent months looking for
# "assistant" and therefore never once entered summary mode — measured
# 2026-09-01 across every captured prompt dump: one match for "agent", zero
# for "assistant". Both wordings are kept, because the failure is silent by
# construction: a request that is not recognised looks exactly like ordinary
# traffic, so nothing reports the miss.
#
# Deliberately the FULL phrase and not the substring "context summarization":
# a user discussing the subject would otherwise have their tools stripped.
_DEFAULT_SUMMARY_MARKERS = (
    "context summarization agent",
    "context summarization assistant",
)

# OpenCode 2.0 compacts differently: it keeps the agent's own system prompt
# and tool list -- so the engine reuses the cached prefix of a 100k-token
# conversation -- and appends the request as the LAST user message, opening
# with one of these two sentences (read out of opencode-cli 2.0.32). The
# summarizer's own system prompt ("anchored context summarization assistant")
# never reaches the gateway on that path. Measured 2026-10-18: a compaction of
# 104 646 tokens was served as ordinary traffic, `is_summary` False.
#
# Matched in the last user message ONLY. After a compaction the summary is
# carried forward in the conversation, and a phrase found anywhere would make
# every later turn a "summary" with its tools switched off.
_DEFAULT_ANCHORED_SUMMARY_MARKERS = (
    "create a new anchored summary from the conversation history above",
    "update the anchored summary below using the conversation history above",
)

# What a client writes in place of an image it decided the model cannot read.
# OpenCode 1.18.30 emits
#   ERROR: Cannot read "<file>" (this model does not support image input).
# and the payload never leaves the client, so the gateway sees a turn that is
# indistinguishable from one that never carried a picture. Matching this is the
# only way the gateway can tell the two apart -- see "The Model List Is Two
# Places" in AGENTS.md for the day it cost.
#
# Only the PARENTHESISED phrase is matched. The filename between it and the
# verb varies without limit and is the client's, not ours; the phrase is one
# the client composed from a template and is what stays put across files.
_DEFAULT_STRIPPED_MEDIA_RE = (
    r"\(this model does not support (image|audio|video|pdf) input\)")

# The workspace template's opening heading. Deliberately strict: matching the
# mere substring "active-context.md" false-positives on the LODESTAR.md read,
# whose text mentions the file by name.
_DEFAULT_HEADING = "# active context"

# embed.py's own search output: "[lessons] sim=0.645 session=... : ...".
# This one is an internal contract — we print it — but it is still a string
# agreement between two programs, so it belongs here with the rest.
_DEFAULT_SEARCH_RE = r"\[\w[\w-]*\]\s+sim=\d"

# Status vocabulary of the template. Polish by default because the shipped
# template is Polish; a translated template overrides these.
_DEFAULT_STATUS_OPEN = r"\bw\s+toku\b"
_DEFAULT_STATUS_CLOSED = r"zako[nń]czono"

# Argument keys carrying the NEW text of a write/edit. oldString is excluded on
# purpose: an edit replacing "W toku" with "Zakończono" carries the old status
# there and would mask the closure.
_DEFAULT_NEW_CONTENT_KEYS = ("content", "newString", "new_string", "new_content",
                             "CodeContent", "text")

# The template's completed-task log. Its heading is the workspace's contract,
# not ours, so it is overridable like the rest of this module.
_DEFAULT_PROGRESS_HEADING = "## progress log"
_DEFAULT_PROGRESS_ENTRY = r"^\s*[-*]\s+\S"

# The template ships "- [List of finished tasks]" under the heading, and models
# routinely keep placeholder lines. They are not entries.
_PLACEHOLDER_RE = re.compile(r"^[-*]?\s*\[.*\]\s*$")

_ENV_PREFIX = "LOCAL_LLM_GATEWAY_"


# -- Overrides ---------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _settings() -> dict:
    """settings.json, read once. Never raises — a broken file means defaults."""
    try:
        from gateway.config_file import load_settings_from_file
        return load_settings_from_file() or {}
    except Exception:
        return {}


def _override_tuple(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(f"{_ENV_PREFIX}{key.upper()}")
    if raw is not None:
        items = [p.strip().lower() for p in raw.split(",") if p.strip()]
        return tuple(items) or default
    from_file = _settings().get(key)
    if isinstance(from_file, list) and from_file:
        return tuple(str(p).strip().lower() for p in from_file)
    return default


def _override_str(key: str, default: str) -> str:
    raw = os.environ.get(f"{_ENV_PREFIX}{key.upper()}")
    if raw:
        return raw
    from_file = _settings().get(key)
    return str(from_file) if isinstance(from_file, str) and from_file else default


def tool_error_markers() -> tuple[str, ...]:
    """Lowercased substrings that mark a failed edit in a tool response."""
    return _override_tuple("tool_error_markers", _DEFAULT_ERROR_MARKERS)


def active_context_heading() -> str:
    return _override_str("active_context_heading", _DEFAULT_HEADING).lower()


@functools.lru_cache(maxsize=8)
def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def search_result_re() -> re.Pattern[str]:
    return _compiled(_override_str("search_result_pattern", _DEFAULT_SEARCH_RE))


def status_open_re() -> re.Pattern[str]:
    return _compiled(_override_str("status_open_pattern", _DEFAULT_STATUS_OPEN))


def status_closed_re() -> re.Pattern[str]:
    return _compiled(_override_str("status_closed_pattern", _DEFAULT_STATUS_CLOSED))


def summary_markers() -> tuple[str, ...]:
    """Lowercased substrings that mark the client's own summarization request.

    Matching these turns off every agentic guard for that request: the model is
    being asked to summarize a conversation, not to act on one.
    """
    return _override_tuple("summary_markers", _DEFAULT_SUMMARY_MARKERS)


def anchored_summary_markers() -> tuple[str, ...]:
    """Lowercased openings of a compaction request sent as the last user turn.

    Unlike `summary_markers`, these mark a request whose system prompt and
    tools are the agent's own, kept on purpose so the cached prefix survives;
    summary mode must therefore leave that prefix exactly as it is.
    """
    return _override_tuple("anchored_summary_markers",
                           _DEFAULT_ANCHORED_SUMMARY_MARKERS)


def stripped_media_re() -> re.Pattern[str]:
    """Matches a client's stand-in for media it refused to send.

    The modality it captures is the client's own word for what it removed,
    which is what makes the log line say `image` rather than `media`.
    """
    return _compiled(_override_str("stripped_media_pattern",
                                   _DEFAULT_STRIPPED_MEDIA_RE))


def new_content_keys() -> tuple[str, ...]:
    return _DEFAULT_NEW_CONTENT_KEYS


def progress_log_heading() -> str:
    return _override_str("progress_log_heading", _DEFAULT_PROGRESS_HEADING).lower()


def progress_entry_re() -> re.Pattern[str]:
    return _compiled(_override_str("progress_entry_pattern", _DEFAULT_PROGRESS_ENTRY))


def count_progress_entries(workspace_dir: str | Path | None) -> int | None:
    """Finished-task entries under `## Progress Log` in active-context.md.

    None when there is no reachable workspace, no readable file, or no such
    section — the caller then says nothing rather than guessing.

    The count is deliberately a heuristic: it counts top-level bullets after
    the heading, so an entry whose body wraps onto a line that itself starts
    with "- " counts twice. It only decides whether to REMIND the model to
    compress the log, so counting slightly high shows the reminder slightly
    early, which is the harmless direction to be wrong in.
    """
    if not workspace_dir:
        return None
    path = Path(workspace_dir) / ".context" / "memory" / "active-context.md"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):
        return None

    heading = progress_log_heading()
    body: list[str] | None = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith(heading):
            body = lines[i + 1:]
            break
    if body is None:
        return None

    entry_re = progress_entry_re()
    return sum(1 for ln in body
               if entry_re.match(ln) and not _PLACEHOLDER_RE.match(ln.strip()))


def is_known_error_marker(content: str) -> bool:
    low = (content or "").lower()
    return any(m in low for m in tool_error_markers())


# -- Ground truth ------------------------------------------------------------

def claimed_new_content(args: dict) -> str:
    """The text a write/edit call says it is putting into the file."""
    parts = [str(args[k]) for k in new_content_keys() if isinstance(args.get(k), str)]
    return "\n".join(parts)


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def write_landed_on_disk(args: dict, workspace_dir: str | Path | None) -> bool | None:
    """Did the file actually get the text this call claimed to write?

    Returns True (the text is there), False (it is not) or None (cannot tell —
    no workspace, unreadable file, or the call carried no new text). None means
    the caller should fall back to reading the client's prose.

    This is what makes the protocol independent of client wording: instead of
    asking "does this error message match my English list?", it asks "does the
    file say what the call said it would say?".
    """
    if not workspace_dir:
        return None
    claimed = _normalized(claimed_new_content(args))
    if len(claimed) < 8:
        # Nothing distinctive enough to look for; a two-word edit would match
        # almost any document by accident.
        return None
    path = Path(workspace_dir) / ".context" / "memory" / "active-context.md"
    try:
        current = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    return claimed in _normalized(current)
