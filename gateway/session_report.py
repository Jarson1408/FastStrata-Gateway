"""
Reassemble sessions from the JSONL audit log.

`gateway_events.jsonl` is written one event at a time, interleaved across
requests, and answers questions about single events well. It answers nothing
about a *task*: what the model was shown at turn 12, whether the prompt had
changed since turn 11, which calls the gateway made instead of the model, when
the temperature moved and why.

This module is the missing direction. It reads the log, groups events by
request into turns and turns into sessions, and renders a timeline. It is pure
reading — no Qt, no network, no model — so the desktop app and the CLI can both
sit on top of it and stay thin.

The unit of a session is `session_id`, which is the workspace directory the
request came from. Events without one (title generators, graph builders, any
request the gateway could not attribute) collect under a single anonymous key
rather than being dropped, because a request the gateway cannot attribute is
exactly the kind it is worth being able to find later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

__all__ = [
    "ToolEvent",
    "Turn",
    "Session",
    "read_events",
    "build_sessions",
    "format_session",
    "format_session_list",
]

ANONYMOUS = "(bez workspace)"

# Rotation suffixes RotatingFileHandler produces, oldest first.
_ROTATED = (".3", ".2", ".1")


@dataclass
class ToolEvent:
    """One tool call as the gateway decided it."""

    name: str
    status: str                      # ok | blocked | synthesized | rewritten | failed
    reason: str | None = None
    emitted: bool = True
    arguments: str | None = None

    @property
    def is_synthetic(self) -> bool:
        return self.status == "synthesized"


@dataclass
class Turn:
    """One upstream call: what went in, what the gateway did, what came back."""

    request_id: str
    ts: float = 0.0
    model: str = ""
    lodestar_state: int | None = None

    # From the `prompt` event — what the model was actually shown.
    prompt_sha256: str | None = None
    prompt_chars: int = 0
    lodestar_chars: int = 0
    has_todo: bool = False
    prefill: bool = False
    temperature: float | None = None
    spill_count: int = 0
    compaction: bool = False

    # From the `response` event.
    finish_reason: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: int = 0

    tools: list[ToolEvent] = field(default_factory=list)
    prune: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def synthesized(self) -> int:
        return sum(1 for t in self.tools if t.is_synthetic)

    @property
    def blocked(self) -> int:
        return sum(1 for t in self.tools if t.status == "blocked")


@dataclass
class Session:
    """Every turn the gateway served for one workspace."""

    session_id: str
    turns: list[Turn] = field(default_factory=list)

    @property
    def started_at(self) -> float:
        return self.turns[0].ts if self.turns else 0.0

    @property
    def ended_at(self) -> float:
        return self.turns[-1].ts if self.turns else 0.0

    @property
    def tokens_in(self) -> int:
        return sum(t.tokens_in for t in self.turns)

    @property
    def tokens_out(self) -> int:
        return sum(t.tokens_out for t in self.turns)

    @property
    def synthesized(self) -> int:
        return sum(t.synthesized for t in self.turns)

    @property
    def errors(self) -> int:
        return sum(len(t.errors) for t in self.turns)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_events(path: str | Path, include_rotated: bool = True) -> Iterator[dict]:
    """Yield events from the audit log, oldest first.

    Malformed lines are skipped rather than raising: the log is append-only and
    written from a running server, so the last line can be a partial write, and
    a report that refuses to open is worse than one missing a line.
    """
    main = Path(path)
    files: list[Path] = []
    if include_rotated:
        files.extend(p for p in (main.with_name(main.name + s) for s in _ROTATED)
                     if p.is_file())
    if main.is_file():
        files.append(main)

    for f in files:
        try:
            with f.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if isinstance(event, dict):
                        yield event
        except OSError:
            continue


def build_sessions(events: Iterable[dict]) -> list[Session]:
    """Group events into sessions of turns, newest session last.

    A turn is keyed on `request_id` — the one field every event of a single
    upstream call shares. `session_id` is filled in from whichever event
    carries it: the `error` event has none, and a turn that only errored still
    belongs to its workspace.
    """
    turns: dict[str, Turn] = {}
    order: list[str] = []
    owner: dict[str, str] = {}

    for event in events:
        rid = str(event.get("request_id") or "")
        if not rid:
            continue
        turn = turns.get(rid)
        if turn is None:
            turn = Turn(request_id=rid, ts=float(event.get("ts") or 0.0))
            turns[rid] = turn
            order.append(rid)

        sid = event.get("session_id")
        if sid and rid not in owner:
            owner[rid] = str(sid)

        kind = event.get("event")
        if kind == "request":
            turn.ts = float(event.get("ts") or turn.ts)
            turn.model = str(event.get("model") or turn.model)
            if event.get("lodestar_state") is not None:
                turn.lodestar_state = int(event["lodestar_state"])
        elif kind == "prompt":
            turn.prompt_sha256 = event.get("prompt_sha256")
            turn.prompt_chars = int(event.get("prompt_chars") or 0)
            turn.lodestar_chars = int(event.get("lodestar_chars") or 0)
            turn.has_todo = bool(event.get("has_todo"))
            turn.prefill = bool(event.get("prefill"))
            turn.spill_count = int(event.get("spill_count") or 0)
            turn.compaction = bool(event.get("compaction"))
            if event.get("temperature") is not None:
                turn.temperature = float(event["temperature"])
            if event.get("lodestar_state") is not None:
                turn.lodestar_state = int(event["lodestar_state"])
            if event.get("model"):
                turn.model = str(event["model"])
        elif kind == "tool_call":
            turn.tools.append(ToolEvent(
                name=str(event.get("tool_name") or "?"),
                status=str(event.get("parse_status") or "?"),
                reason=event.get("blocked_reason"),
                emitted=bool(event.get("emitted", True)),
                arguments=event.get("arguments"),
            ))
        elif kind == "response":
            turn.finish_reason = event.get("finish_reason")
            turn.tokens_in = int(event.get("tokens_in") or 0)
            turn.tokens_out = int(event.get("tokens_out") or 0)
            turn.duration_ms = int(event.get("duration_ms") or 0)
        elif kind == "prune":
            turn.prune = {
                "pruned_count": int(event.get("pruned_count") or 0),
                "tokens_before": int(event.get("tokens_before") or 0),
                "tokens_after": int(event.get("tokens_after") or 0),
            }
        elif kind == "error":
            turn.errors.append(
                f"{event.get('error_type') or 'error'}: {event.get('message') or ''}".strip()
            )

    sessions: dict[str, Session] = {}
    seq: list[str] = []
    for rid in order:
        sid = owner.get(rid, ANONYMOUS)
        session = sessions.get(sid)
        if session is None:
            session = Session(session_id=sid)
            sessions[sid] = session
            seq.append(sid)
        session.turns.append(turns[rid])

    for session in sessions.values():
        session.turns.sort(key=lambda t: t.ts)
    return [sessions[s] for s in seq]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _clock(ts: float) -> str:
    if not ts:
        return "--:--:--"
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _stamp(ts: float) -> str:
    if not ts:
        return "?"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _thousands(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _short_args(args: str | None, limit: int = 96) -> str:
    if not args:
        return ""
    flat = " ".join(str(args).split())
    return flat if len(flat) <= limit else flat[:limit - 1] + "…"


def format_session(session: Session) -> str:
    """Render one session as a readable timeline.

    Deliberately plain text: it has to be as useful piped through `less` at
    two in the morning as it is in the dashboard.
    """
    head = [
        f"=== {session.session_id} ===",
        f"{len(session.turns)} tur · {_stamp(session.started_at)} → {_stamp(session.ended_at)}"
        f" · {_thousands(session.tokens_in)} in / {_thousands(session.tokens_out)} out"
        f" · synthesized: {session.synthesized}"
        + (f" · errors: {session.errors}" if session.errors else ""),
        "",
    ]

    lines: list[str] = []
    prev_sha: str | None = None
    for i, turn in enumerate(session.turns, start=1):
        bits = [f"[{i:>3}] {_clock(turn.ts)}  {turn.request_id}"]
        if turn.lodestar_state is not None:
            bits.append(f"state={turn.lodestar_state}")
        if turn.prompt_sha256:
            mark = "=" if prev_sha == turn.prompt_sha256 else " "
            bits.append(f"prompt={turn.prompt_sha256[:8]}{mark}")
            prev_sha = turn.prompt_sha256
        if turn.prompt_chars:
            detail = f"{_thousands(turn.prompt_chars)} zn"
            if turn.lodestar_chars:
                detail += f", LODESTAR {_thousands(turn.lodestar_chars)}"
            if turn.has_todo:
                detail += ", TODO"
            bits.append(f"({detail})")
        if turn.prefill:
            bits.append("prefill")
        if turn.temperature is not None:
            bits.append(f"temp={turn.temperature:.2f}")
        if turn.spill_count:
            bits.append(f"spill×{turn.spill_count}")
        if turn.compaction:
            bits.append("compaction")
        lines.append("  ".join(bits))

        if turn.prune:
            lines.append(
                f"      ⋯ prune: -{turn.prune['pruned_count']} messages "
                f"({_thousands(turn.prune['tokens_before'])} → "
                f"{_thousands(turn.prune['tokens_after'])} tok)"
            )

        for tool in turn.tools:
            label = tool.status.upper() if tool.is_synthetic else tool.status
            row = f"      → {tool.name:<14} {label}"
            if tool.reason:
                row += f" ({tool.reason})"
            if tool.arguments:
                row += f"  {_short_args(tool.arguments)}"
            lines.append(row)

        for err in turn.errors:
            lines.append(f"      ! {err}")

        if turn.finish_reason or turn.tokens_out:
            lines.append(
                f"      ← {turn.finish_reason or '?'}  "
                f"in={_thousands(turn.tokens_in)} out={_thousands(turn.tokens_out)}"
                f"  {turn.duration_ms / 1000:.1f}s"
            )
        lines.append("")

    return "\n".join(head + lines).rstrip() + "\n"


def format_session_list(sessions: list[Session]) -> str:
    """One line per session, newest last — the index for picking one to read."""
    if not sessions:
        return "Brak sesji w logu.\n"
    rows = []
    for s in sessions:
        rows.append(
            f"{_stamp(s.ended_at):<17} {len(s.turns):>4} tur  "
            f"{_thousands(s.tokens_in):>9} in  syn={s.synthesized:<3} "
            f"{'ERR ' if s.errors else '    '}{s.session_id}"
        )
    return "\n".join(rows) + "\n"
