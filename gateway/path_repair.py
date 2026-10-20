"""Repair a file path the model almost got right.

From a real session (2026-08-31, ACME_HW). The model had already read this file
successfully twice in the same conversation, then asked for it again as:

    C:\\Users\\user\\Documents\\PRO JEKTY\\ACME_HW\\engram-out\\converted\\
    ACME-DOC-00 1_3 Design Document_e97f2897.md

Two spurious spaces — `PROJEKTY` -> `PRO JEKTY`, `ACME-DOC-0001_3` ->
`ACME-DOC-00 1_3`. Not ignorance: the correct path was in its context. It is
a decoding artefact. Reproducing a long literal verbatim under `temperature
0.3` slips eventually, and these paths are long and full of rare token
sequences.

The gateway can fix that, because it has something the model does not: the
actual filesystem. A corrupted path is a near-miss, and near-misses are
mechanically correctable.

## How

Segment by segment from the longest existing prefix, never a search of the
tree. For each segment that does not exist, list its parent and look for
candidates within a small edit distance. Accept only when there is EXACTLY
one; zero or several means the gateway does not know, and not knowing means
leaving the call alone.

Bounded by construction: depth x directory size, with a scan cap and a wall
clock, and it stops at the first ambiguity.

## What it must never do

**Never repair a path for a write or an edit.** A file that does not exist is
the normal case there — the model is creating it — and "repairing" a new
file's name to a similar existing one would overwrite the wrong file. This is
the sharpest edge in the module; `is_repairable_tool` is the only gate, and
READ_TOOLS is the only set that passes it.

**Never refuse.** Measured 2026-08-31 22:26:21, refusing a call cost the whole
session (see the note on error_chunk in app.py). A path that cannot be
repaired is passed through untouched: "no such file" from the client is a fine
outcome the model can act on, and an existing path outside the workspace stays
the user's decision, which the client already asks about.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from gateway.lodestar.tool_names import PATH_KEYS, READ_TOOLS

__all__ = ["repair_path", "repair_params", "repair_history",
           "is_repairable_tool"]

_log = logging.getLogger("gateway.path_repair")

# Whole-operation wall clock. A disconnected mapped drive turns a directory
# listing into seconds, and this runs in the request path.
_BUDGET_SECONDS = 1.0
# Directories larger than this are not searched: at that size a near-miss is
# far more likely to be ambiguous than correct.
_MAX_ENTRIES = 5000


def is_repairable_tool(name: str) -> bool:
    """Only reads. See the module docstring — this is the dangerous edge."""
    return name.lower() in READ_TOOLS


def _max_distance(segment: str) -> int:
    """How wrong a segment may be and still be recognisable.

    Scaled by length: one character in `PROJEKTY` is a big share of the name,
    one in a forty-character document title is not.
    """
    return max(1, min(2, len(segment) // 10))


def _distance_within(a: str, b: str, limit: int) -> int | None:
    """Levenshtein distance, or None once it is known to exceed `limit`.

    Bounded on purpose: the answer "far apart" is worth no cycles, and this
    runs against every entry of a directory.
    """
    if abs(len(a) - len(b)) > limit:
        return None
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return None
        prev = cur
    return prev[-1] if prev[-1] <= limit else None


def _unique_near_match(parent: Path, segment: str) -> str | None:
    """The one entry of `parent` that `segment` was probably meant to be.

    None when there is no candidate, or more than one. Ambiguity is not a
    close call to be broken by a heuristic: two plausible files mean the
    gateway does not know which the model wanted, and guessing would open a
    document nobody asked for.
    """
    try:
        entries = os.listdir(parent)
    except OSError:
        return None
    if len(entries) > _MAX_ENTRIES:
        _log.debug("path repair: %s has %d entries, not searching",
                   parent, len(entries))
        return None

    limit = _max_distance(segment)
    lowered = segment.lower()
    best: list[str] = []
    best_distance = limit + 1
    for entry in entries:
        # Case differences alone are not a corruption worth repairing: Windows
        # opens the file regardless, so a "repair" would only add noise.
        d = _distance_within(lowered, entry.lower(), limit)
        if d is None or d == 0:
            continue
        if d < best_distance:
            best, best_distance = [entry], d
        elif d == best_distance:
            best.append(entry)

    if len(best) != 1:
        if best:
            _log.debug("path repair: %d equally close candidates for %r in %s",
                       len(best), segment, parent)
        return None
    return best[0]


def repair_path(path: str, workspace_dir: str | None) -> str | None:
    """The real path this near-miss meant, or None to leave the call alone.

    None is returned far more often than a repair, and every one of those
    cases is deliberate: the path already exists, nothing plausible is near
    it, several things are equally near it, the filesystem was slow, or the
    result would land outside the workspace.
    """
    if not path or not workspace_dir:
        return None
    # UNC paths are never touched: a listing across an unreachable share is
    # exactly the latency this must not introduce.
    if path.startswith("\\\\") or path.startswith("//"):
        return None

    deadline = time.monotonic() + _BUDGET_SECONDS
    try:
        root = Path(workspace_dir)
        target = Path(path)
        if not target.is_absolute():
            target = root / target
        if target.exists():
            return None  # nothing to repair
    except (OSError, ValueError):
        return None

    parts = target.parts
    if not parts:
        return None

    current = Path(parts[0])
    repaired = False
    for segment in parts[1:]:
        if time.monotonic() > deadline:
            _log.debug("path repair: budget exhausted on %s", path)
            return None
        candidate = current / segment
        try:
            if candidate.exists():
                current = candidate
                continue
        except OSError:
            return None
        match = _unique_near_match(current, segment)
        if match is None:
            return None
        current = current / match
        repaired = True

    if not repaired:
        return None
    try:
        if not current.exists():
            return None
        # The gateway is only confident about the project it was given. A
        # near-miss that lands elsewhere is precisely where the user's
        # judgement belongs, and the client already asks for it.
        resolved, root_resolved = current.resolve(), root.resolve()
        if root_resolved != resolved and root_resolved not in resolved.parents:
            _log.debug("path repair: %s would land outside %s", current, root)
            return None
    except (OSError, ValueError):
        return None
    return str(current)


def _repair_with_timeout(path: str, workspace_dir: str | None) -> str | None:
    """`repair_path` on a throwaway thread, abandoned if the disk hangs."""
    result: list[str | None] = []

    def probe() -> None:
        try:
            result.append(repair_path(path, workspace_dir))
        except Exception as exc:      # a repair must never fail a request
            _log.debug("path repair raised %s", exc)

    worker = threading.Thread(target=probe, daemon=True)
    worker.start()
    worker.join(_BUDGET_SECONDS + 0.5)
    return result[0] if result else None


def repair_history(messages: list[dict], workspace_dir: str | None) -> tuple[list[dict], int]:
    """Repair corrupted paths in the assistant's OWN earlier tool calls.

    Repairing only the outgoing call is not enough, and the logs said so.
    A call the gateway could not repair still reaches the client, fails, and
    stays in the history — where it becomes an exemplar. Measured 2026-09-01:
    a prompt carried the model's own `PROJEKTOACME_HW` twice, and the next
    corruption was `PROJKTOACME_HW` — that same string minus one character.
    The model had stopped corrupting the real path and started corrupting its
    own last attempt, which is why the damage compounded instead of repeating.

    So the exemplars are cleaned before the model sees them again. This
    rewrites what the model is shown, not what the client did: the tool
    results are left exactly as they came back, because a result claiming a
    path that was never used would be a lie about what happened.

    The rewrite is stable — the same corrupted string always maps to the same
    real path — so the KV prefix pays once, at the turn the repair first
    applies, and is byte-identical on every turn after it.
    """
    if not workspace_dir or not messages:
        return messages, 0

    cache: dict[str, str | None] = {}
    out: list[dict] = []
    fixed = 0
    for msg in messages:
        calls = msg.get("tool_calls")
        if msg.get("role") != "assistant" or not isinstance(calls, list):
            out.append(msg)
            continue
        new_calls = []
        touched = False
        for call in calls:
            fn = call.get("function") or {}
            name = fn.get("name") or ""
            if not is_repairable_tool(name):
                new_calls.append(call)
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (json.JSONDecodeError, ValueError, TypeError):
                new_calls.append(call)
                continue
            if not isinstance(args, dict):
                new_calls.append(call)
                continue
            changed = False
            for key in PATH_KEYS:
                value = args.get(key)
                if not isinstance(value, str) or not value.strip():
                    continue
                value = value.strip()
                if value not in cache:
                    cache[value] = _repair_with_timeout(value, workspace_dir)
                repaired = cache[value]
                if repaired and repaired != value:
                    args[key] = repaired
                    changed = True
            if not changed:
                new_calls.append(call)
                continue
            new_call = {**call, "function": {
                **fn, "arguments": json.dumps(args, ensure_ascii=False)}}
            new_calls.append(new_call)
            touched = True
            fixed += 1
        out.append({**msg, "tool_calls": new_calls} if touched else msg)
    return out, fixed


def repair_params(name: str, params: dict,
                  workspace_dir: str | None) -> tuple[dict, str, str] | None:
    """Repaired params plus (before, after), or None if nothing was changed."""
    if not is_repairable_tool(name) or not isinstance(params, dict):
        return None
    for key in PATH_KEYS:
        value = params.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        fixed = _repair_with_timeout(value.strip(), workspace_dir)
        if fixed and fixed != value:
            return {**params, key: fixed}, value, fixed
    return None
