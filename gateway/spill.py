"""
Tool-result spill: keep the big output, stop paying for it every turn.

A single `pytest` run or a wide `grep` returns tens of thousands of tokens.
`prune_messages` cannot help with that: it decides whole conversational blocks,
so one oversized result either fits or evicts several older turns — including,
eventually, the user's original request. The model needs the first failure and
the summary line, not three thousand lines of traceback.

So oversized tool results are replaced with a bounded preview and a path. The
full text is written to a file the client's own `read`/`grep` tools can reach,
which means nothing is lost — it just stops being resident in every subsequent
prefill.

Two properties this module must keep, both load-bearing:

**The preview is a pure function of the content.** The store is
content-addressed and the preview is derived deterministically, so the same
tool result renders to the same bytes on every request. That is what keeps the
KV prefix stable — a preview that varied between turns would invalidate the
cache and cost a full prefill on a 35B model, which is a far worse trade than
the tokens it saved.

**LODESTAR's evidence is never spilled.** The state machine derives the
protocol phase from the *content* of tool responses: it looks for the
`# Active Context` heading and for `[lessons] sim=` search output. Replacing
either with a preview would mean the model read active-context and the gateway
did not notice — leaving it stuck in Phase A, synthesizing the same read
forever. Those responses are small, so exempting them costs nothing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path

from gateway.lodestar.state import (
    response_indicates_context_read,
    response_indicates_embed_search,
)

__all__ = ["SpillStore", "SPILL_MARKER", "apply_spill_policy", "is_spilled"]

_log = logging.getLogger("gateway.spill")

# Opening token of the replacement text. Also the guard against spilling a
# preview a second time — the client echoes history back on every turn.
SPILL_MARKER = "[GATEWAY SPILL]"

# How much of the result survives inline. The head carries the command, the
# first error and any heading a downstream check might look for; the tail
# carries the summary line, which is where a test run says what it did.
_HEAD_LINES = 15
_TAIL_LINES = 5
_MAX_LINE_CHARS = 300


_UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def is_spilled(content: str) -> bool:
    """True if this text is already a spill preview."""
    return SPILL_MARKER in content


def _session_folder(session_id: str | None) -> str:
    """A stable, filesystem-safe folder name for one workspace.

    Keeps the last path component readable — a directory full of bare hashes
    is useless when you are looking for one session by hand — and appends a
    digest of the full path so two projects with the same folder name never
    collide.
    """
    raw = session_id or "anonymous"
    tail = Path(raw.replace("\\", "/")).name or "root"
    safe = _UNSAFE_NAME_RE.sub("-", tail).strip("-") or "root"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:40]}-{digest}"


class SpillStore:
    """Content-addressed storage for oversized tool results.

    Writes are best-effort. If the disk is full or the path is unwritable the
    store says so by returning None and the caller leaves the original result
    in place: degrading to a large prompt is recoverable, losing the model's
    only view of a test failure is not.
    """

    def __init__(self, root: Path):
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, session_id: str | None, content: str) -> Path:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return self._root / _session_folder(session_id) / f"{digest}.txt"

    def store(self, session_id: str | None, content: str) -> Path | None:
        """Persist `content` and return its path, or None if it could not be."""
        target = self.path_for(session_id, content)
        expected = len(content.encode("utf-8"))
        if target.is_file():
            try:
                if target.stat().st_size == expected:
                    return target  # same bytes already on disk
            except OSError:
                return target
            # Wrong size for this content: a file written before the newline
            # fix, inflated by one \r per line. Rewriting is self-healing, and
            # a stat() is cheap enough to pay on every spill — the alternative
            # is that the corrupted copies live forever, because the name is
            # derived from the content and the old code never rewrote a name
            # it found.
            _log.info("SPILL: rewriting %s — %d bytes on disk, %d expected",
                      target.name, target.stat().st_size, expected)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # newline="" disables newline translation. Without it, Python's
            # text mode turns every \n into \r\n on Windows — so content that
            # already had \r\n reached disk as \r\r\n. Every spill file this
            # gateway wrote was corrupted that way, and it COMPOUNDED: the
            # model reads the file, the doubled CR comes back in the tool
            # result, that result is spilled again, and another CR is added.
            # Measured 2026-09-01: every file in a live spill folder carried
            # \r\r\r, and the previews had decayed to nothing because
            # splitlines() saw the padding as blank lines.
            with target.open("w", encoding="utf-8", newline="") as fh:
                fh.write(content)
        except OSError as e:
            _log.warning("Spill write failed for %s: %s", target, e)
            return None
        return target

    def prune(self, max_age_days: int) -> int:
        """Delete spill files older than the threshold. Returns how many went.

        `max_age_days <= 0` keeps everything, for anyone who wants this
        directory to be an archive rather than a buffer.

        Until 2026-09-03 there was no deletion of any kind: the directory held
        literal copies of files from every project the gateway had ever
        served, retention was infinite, and no document mentioned that it
        existed (G-2026-01). Called once at startup — see the note on
        `spill_retention_days`.

        Every failure is swallowed and logged. Cleaning is hygiene, not a
        condition of running: one locked file must never stop the gateway from
        starting.
        """
        if max_age_days <= 0 or not self._root.is_dir():
            return 0

        cutoff = time.time() - max_age_days * 86400
        removed = 0
        for path in self._root.rglob("*"):
            if not path.is_file():
                continue
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
                path.unlink()
                removed += 1
            except OSError as e:
                _log.warning("SPILL: could not remove %s: %s", path, e)

        # A project folder left with nothing in it is noise in a directory
        # people open by hand to find one session.
        for folder in self._root.iterdir():
            try:
                if folder.is_dir() and not any(folder.iterdir()):
                    folder.rmdir()
            except OSError:
                pass

        if removed:
            _log.info("SPILL: removed %d file(s) older than %d days from %s",
                      removed, max_age_days, self._root)
        return removed


def _truncate_line(line: str, max_chars: int = _MAX_LINE_CHARS) -> str:
    """Cap single line length to prevent minified blobs from bloating previews."""
    if len(line) <= max_chars:
        return line
    return line[:max_chars] + "..."


def _first_nonblank(lines: list[str], limit: int) -> list[str]:
    """The first `limit` lines that carry something, in order."""
    out = []
    for line in lines:
        if line.strip():
            out.append(_truncate_line(line))
            if len(out) == limit:
                break
    return out


def _last_nonblank(lines: list[str], limit: int) -> list[str]:
    """The last `limit` lines that carry something, in order."""
    out: list[str] = []
    for line in reversed(lines):
        if line.strip():
            out.append(_truncate_line(line))
            if len(out) == limit:
                break
    out.reverse()
    return out


def _preview(content: str, path: Path) -> str:
    """Head + tail of `content` with a locator between them.

    Blank lines do not count towards the head or the tail. A preview exists to
    tell the model what it is looking at, and padding tells it nothing — but
    the first version counted lines rather than content, so a result padded
    with blank lines produced a preview of nothing at all. Measured
    2026-09-01: 21 KB and 39 KB results reduced to previews of 0 KB, after
    which the model could only go and read the file, get another empty
    preview, and repeat until the client's loop detector stopped it.

    Still a pure function of the content, which is what keeps the KV prefix
    stable: the same result always yields the same preview.
    """
    lines = content.splitlines()
    head = _first_nonblank(lines, _HEAD_LINES)
    tail_pool = lines[len(head):] if len(head) < len(lines) else []
    tail = _last_nonblank(tail_pool, _TAIL_LINES) if len(lines) > _HEAD_LINES else []
    omitted = len(lines) - len(head) - len(tail)
    kb = len(content.encode("utf-8")) / 1024

    notice = (
        f"{SPILL_MARKER} {omitted} more lines ({kb:.0f} KB total) are not shown here.\n"
        f"Full output: {str(path).replace(chr(92), '/')}\n"
        f"Read or grep that file if you need the part that was cut."
    )
    parts = ["\n".join(head), notice]
    if tail:
        parts.append("\n".join(tail))
    return "\n\n".join(parts)


def _exempt(content: str, tool_name: str = "") -> bool:
    """Responses that must never be spilled: LODESTAR evidence and subagent task summaries."""
    if tool_name.lower() in ("task", "subagent", "agent"):
        return True
    return (
        response_indicates_context_read(content)
        or response_indicates_embed_search(content)
    )


def _reads_a_spill(arguments: object, root: Path) -> bool:
    """True when a tool call's arguments name a file in the spill store: the model is reading back a result the
    gateway spilled, as the preview tells it to. Spilling that read as well hands it the same preview again, and it
    reads the file again until a loop guard stops it -- seen 2026-10-21 on FastStrata: an 8 KB main.js read, spilled,
    its spill file read three times, each time spilled to the same preview, and the turn ended with the model never
    having seen the file. Matched on the store's folder in the arguments, slashes either way, any case."""
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False) if arguments else ""

    def norm(text: str) -> str:
        return text.replace("\\\\", "/").replace("\\", "/").lower()

    return bool(arguments) and norm(str(root)) in norm(arguments)


def apply_spill_policy(
    messages: list[dict],
    max_inline_bytes: int,
    store: SpillStore,
    session_id: str | None = None,
) -> tuple[list[dict], int]:
    """Replace oversized tool results with previews.

    Returns the messages and how many results were spilled. Messages are
    copied rather than mutated — the caller's history is shared with the state
    machine and the enforcer, and both must keep seeing what the client sent.

    `max_inline_bytes <= 0` disables the policy entirely.
    """
    if max_inline_bytes <= 0 or not messages:
        return messages, 0

    # Build lookup of tool_call_id -> function name (and arguments) from assistant messages
    call_names: dict[str, str] = {}
    call_args: dict[str, object] = {}
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                c_id = tc.get("id")
                fn_name = (tc.get("function") or {}).get("name")
                if c_id and fn_name:
                    call_names[c_id] = fn_name
                if c_id:
                    call_args[c_id] = (tc.get("function") or {}).get("arguments")

    out: list[dict] = []
    spilled = 0
    for msg in messages:
        content = msg.get("content")
        if (
            msg.get("role") != "tool"
            or not isinstance(content, str)
            or len(content.encode("utf-8")) <= max_inline_bytes
            or is_spilled(content)
        ):
            out.append(msg)
            continue

        tool_name = msg.get("name") or call_names.get(msg.get("tool_call_id", ""), "")
        if _exempt(content, tool_name) or _reads_a_spill(call_args.get(msg.get("tool_call_id", "")), store.root):
            out.append(msg)
            continue

        path = store.store(session_id, content)
        if path is None:
            out.append(msg)  # best-effort: an unwritable store changes nothing
            continue

        replacement = dict(msg)
        replacement["content"] = _preview(content, path)
        out.append(replacement)
        spilled += 1
        _log.info(
            "SPILL: tool result %d KB → %d KB preview (%s)",
            len(content.encode("utf-8")) // 1024,
            len(replacement["content"].encode("utf-8")) // 1024,
            path.name,
        )

    return out, spilled
