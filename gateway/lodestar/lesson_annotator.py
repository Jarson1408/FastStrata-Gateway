"""LODESTAR lesson annotator.

The semantic-memory `embed.py search` command returns lessons learned from
previous sessions. Those lessons are HINTS, not ground truth: a lesson that
says "I created script X" tells the model what was done in the past, not
what is true in the current workspace.

A previous failure mode (2026-07-07 Capacitor_charging session): the model
found a lesson saying "I created capacitor_charge.py" and treated it as
proof the task was already done, never writing the file.

This module annotates tool results that contain the `[lessons] sim=...`
marker emitted by `embed.py search` with a strong, model-visible warning
that the lessons are stale candidates and must be verified against the
current workspace before being trusted.
"""
from __future__ import annotations

from gateway import prompts

import re


# Marker emitted by embed.py's search command for each match line.
# Examples (with and without TTL fields added in v2.1):
#   [lessons] sim=0.645: Created capacitor script...
#   [lessons] sim=0.512 session=abc123 2d ago: ...
_LESSON_LINE_RE = re.compile(
    r"^\[(?P<category>[^\]]+)\]\s+sim=(?P<sim>[\d.]+)(?:\s+session=(?P<session>\S+))?"
    r"(?:\s+(?P<age>\S+\s+ago))?\s*:\s*(?P<text>.*)$",
    re.IGNORECASE,
)


# Prepended once to any tool result carrying at least one lesson line.
#
# The text lives in `gateway/prompts/lesson_annotation.md` (C-2026-03): it is
# read by the model, so a change to it changes behaviour, and only a prompt
# file moves `prompts_version` in the audit log. The trailing blank line stays
# HERE rather than in the file — `prompts._read` strips trailing newlines, so
# a separator stored in the file would silently vanish.
_SEPARATOR = "\n\n"


def _warning() -> str:
    return prompts.get("lesson_annotation") + _SEPARATOR


def contains_lesson_line(content: str) -> bool:
    """Return True if the content has at least one [lessons] (or any [cat] sim=...) line."""
    if not isinstance(content, str):
        return False
    for line in content.splitlines():
        if _LESSON_LINE_RE.match(line.strip()):
            return True
    return False


def annotate(content: str) -> str:
    """If `content` contains lesson lines, prepend the staleness warning.

    Idempotent: if the warning header is already present, returns content unchanged.
    Safe to call on any string (no-op for non-string inputs and content without
    lesson lines).
    """
    if not isinstance(content, str) or not content:
        return content
    if "[LODESTAR LESSON ANNOTATION]" in content:
        return content
    if not contains_lesson_line(content):
        return content
    return _warning() + content
