"""
Replay a gateway session from the JSONL audit log.

    python test_scripts/replay_session.py                 # list sessions
    python test_scripts/replay_session.py --last          # newest session
    python test_scripts/replay_session.py -s Local-LLM    # match by substring

Read-only: it opens the log and nothing else, so it is safe to run against a
gateway that is mid-request.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway.session_report import (  # noqa: E402
    build_sessions, format_session, format_session_list, read_events,
)

DEFAULT_LOG = Path(__file__).resolve().parent.parent / "gateway_events.jsonl"


def _normalize(path_like: str) -> str:
    """Fold case and path separators so `-s audyt/2.6` matches a Windows id."""
    return path_like.replace("\\", "/").lower()


def main() -> int:
    # The timeline uses arrows and separators that cp1250 — still the default
    # console encoding on Polish Windows — cannot encode, and a diagnostic tool
    # that dies on its own output is worse than no tool.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG,
                        help="path to gateway_events.jsonl")
    parser.add_argument("-s", "--session", default=None,
                        help="substring of the session id (workspace path)")
    parser.add_argument("--last", action="store_true",
                        help="replay the most recent session")
    parser.add_argument("--no-rotated", action="store_true",
                        help="read only the current log file, not .1/.2/.3")
    args = parser.parse_args()

    if not args.log.is_file():
        print(f"Nie znaleziono logu: {args.log}", file=sys.stderr)
        return 1

    sessions = build_sessions(
        read_events(args.log, include_rotated=not args.no_rotated))
    if not sessions:
        print("Brak zdarzeń w logu.", file=sys.stderr)
        return 1

    if args.last:
        chosen = [max(sessions, key=lambda s: s.ended_at)]
    elif args.session:
        needle = _normalize(args.session)
        chosen = [s for s in sessions if needle in _normalize(s.session_id)]
        if not chosen:
            print(f"Żadna sesja nie pasuje do {args.session!r}.", file=sys.stderr)
            return 1
    else:
        sys.stdout.write(format_session_list(sessions))
        return 0

    for session in chosen:
        sys.stdout.write(format_session(session))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
