"""
Summarise gateway_events.jsonl into per-session rows.

    python test_scripts/report_sessions.py
    python test_scripts/report_sessions.py --file path/to/gateway_events.jsonl
    python test_scripts/report_sessions.py --requests    # one row per request

The audit log records one JSON object per event with a request_id and (since
v4.4.0) a session_id naming the workspace. This turns that stream into the
view that actually answers the operational question: for each task, how many
turns did it take, how many tokens did it burn, and how often did the gateway
have to make a tool call because the model would not.

No dependencies beyond the standard library.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DEFAULT_LOG = Path(__file__).resolve().parent.parent / "gateway_events.jsonl"


def load_events(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"No audit log at {path}")
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn line from a rotated file
    return events


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "?"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _short_session(sid: str | None) -> str:
    if not sid:
        return "(no workspace)"
    return Path(sid).name or sid


def summarise(events: list[dict]) -> tuple[dict, dict]:
    """Return (per-session aggregate, per-request aggregate)."""
    requests: dict[str, dict] = defaultdict(lambda: {
        "session_id": None, "ts": None, "model": None, "stream": None,
        "tokens_in": 0, "tokens_out": 0, "duration_ms": 0,
        "tool_calls": 0, "synthesized": 0, "blocked": 0, "errors": 0,
        "finish_reason": None,
    })

    for ev in events:
        rid = ev.get("request_id")
        if not rid:
            continue
        r = requests[rid]
        kind = ev.get("event")
        if ev.get("session_id") and not r["session_id"]:
            r["session_id"] = ev["session_id"]

        if kind == "request":
            r["ts"] = ev.get("ts")
            r["model"] = ev.get("model")
            r["stream"] = ev.get("stream")
        elif kind == "response":
            r["tokens_in"] = ev.get("tokens_in", 0)
            r["tokens_out"] = ev.get("tokens_out", 0)
            r["duration_ms"] = ev.get("duration_ms", 0)
            r["tool_calls"] = ev.get("tool_call_count", 0)
            r["finish_reason"] = ev.get("finish_reason")
            r["synthesized"] = max(r["synthesized"], ev.get("synthesized_count", 0) or 0)
        elif kind == "tool_call":
            status = ev.get("parse_status")
            if status == "synthesized":
                r["synthesized"] += 1
            elif status == "blocked":
                r["blocked"] += 1
        elif kind == "error":
            r["errors"] += 1

    sessions: dict[str, dict] = defaultdict(lambda: {
        "requests": 0, "tokens_in": 0, "tokens_out": 0, "duration_ms": 0,
        "tool_calls": 0, "synthesized": 0, "blocked": 0, "errors": 0,
        "first_ts": None, "last_ts": None,
    })
    for r in requests.values():
        s = sessions[_short_session(r["session_id"])]
        s["requests"] += 1
        for k in ("tokens_in", "tokens_out", "duration_ms", "tool_calls",
                  "synthesized", "blocked", "errors"):
            s[k] += r[k]
        ts = r["ts"]
        if ts:
            s["first_ts"] = ts if s["first_ts"] is None else min(s["first_ts"], ts)
            s["last_ts"] = ts if s["last_ts"] is None else max(s["last_ts"], ts)

    return sessions, requests


def print_sessions(sessions: dict) -> None:
    if not sessions:
        print("No requests found in the audit log.")
        return
    name_w = max(len(n) for n in sessions) + 2
    print(f"{'workspace':<{name_w}}{'turns':>6}{'tok in':>10}{'tok out':>9}"
          f"{'calls':>7}{'synth':>7}{'blocked':>9}{'errors':>8}  last seen")
    print("-" * (name_w + 66))
    for name, s in sorted(sessions.items(), key=lambda kv: kv[1]["last_ts"] or 0, reverse=True):
        print(f"{name:<{name_w}}{s['requests']:>6}{s['tokens_in']:>10}"
              f"{s['tokens_out']:>9}{s['tool_calls']:>7}{s['synthesized']:>7}"
              f"{s['blocked']:>9}{s['errors']:>8}  {_fmt_ts(s['last_ts'])}")

    total_calls = sum(s["tool_calls"] for s in sessions.values())
    total_synth = sum(s["synthesized"] for s in sessions.values())
    print()
    print(f"{len(sessions)} workspace(s), "
          f"{sum(s['requests'] for s in sessions.values())} requests")
    if total_calls or total_synth:
        share = 100.0 * total_synth / max(total_calls + total_synth, 1)
        print(f"gateway synthesized {total_synth} of {total_calls + total_synth} "
              f"tool calls ({share:.1f}%) - lower is better")


def print_requests(requests: dict) -> None:
    print(f"{'request':<18}{'workspace':<22}{'tok in':>8}{'tok out':>9}"
          f"{'ms':>8}{'calls':>7}{'synth':>7}  finish")
    print("-" * 96)
    for rid, r in sorted(requests.items(), key=lambda kv: kv[1]["ts"] or 0):
        print(f"{rid:<18}{_short_session(r['session_id'])[:20]:<22}"
              f"{r['tokens_in']:>8}{r['tokens_out']:>9}{r['duration_ms']:>8}"
              f"{r['tool_calls']:>7}{r['synthesized']:>7}  {r['finish_reason'] or '?'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--file", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--requests", action="store_true", help="one row per request")
    args = ap.parse_args()

    sessions, requests = summarise(load_events(args.file))
    if args.requests:
        print_requests(requests)
    else:
        print_sessions(sessions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
