"""
Where the wait actually goes: queue, prompt processing, or generation.

    python test_scripts/check_prefill_health.py [--last N] [--events PATH]

`duration_ms` adds three unrelated things together, which is why "the gateway
feels slow" was argued rather than measured for so long. The audit records the
split, and this reads it back:

    queue_wait_ms   waiting for the single upstream slot — another request's
                    cost, not this one's
    ttft_ms         upstream request sent -> first token back. This is prompt
                    processing.
    duration_ms     everything from the moment the slot was acquired, so it
                    does NOT include queue_wait_ms; duration - ttft is
                    generation

## The denominator, and why it changed on 2026-09-10

This used to divide TTFT by the WHOLE prompt and call the result "ms per 1k
tokens". That number is not a prefill rate and it is actively misleading,
because with a warm KV prefix the upstream processes only the tokens that are
NEW since the last turn. Measured on this rig:

    prompt   new tokens   ttft     implied rate over the whole prompt
    45,220      +14,975   18.6s    411 ms/1k   "slow"
    47,145       +1,925    3.0s     65 ms/1k   "fast"
    58,122      +10,977   17.4s    300 ms/1k   "slow"
    58,773         +651    1.8s     30 ms/1k   "fast"

Read per whole prompt, that alternation looks like a prefix cache being hit
unevenly — which is exactly what it was read as, in writing, before anyone
divided by the right thing. Per NEW token those four rows are 632-806 tok/s,
matching the 806 tok/s recorded in AGENTS.md. Nothing was wrong. The metric
was.

So the diagnostic column is now `ms/1k new` and its companion `tok/s`. A
missed prefix shows up as a turn whose NEW tokens are far more than the
conversation actually grew — which is what `new` being close to `tok_in`
means, and why both are printed.

The rate is NOT expected to be constant, and saying so was this file's second
wrong claim in one day. Every new token attends over everything before it, so
the cost per new token rises with total context whatever the cache does.
Measured here, prefix warm throughout: 806 tok/s at 45k, 632 at 58k, 602 at
85k, 459 at 107k, ~330 at 140k — a bit better than linear in context. Judge a
row against that curve.

Two kinds of row cannot carry the measurement, and they are labelled rather
than silently averaged in:

    engine  the upstream reported `cached_tokens` and `new` is MEASURED
            rather than inferred — ExLlamaV3 counts the KV pages the prompt
            actually hit. Trust these rows over any other basis, and note
            that they are the only ones that mean anything on the first turn
            of a session or after a compaction
    cold    first turn seen for that session — every token is new by
            definition, so the rate is a cold-prefill rate, not a regression
    rewrit  tokens_in went DOWN, so the client compacted or the gateway
            pruned; the prompt was rewritten rather than appended to, and a
            delta is meaningless

`gap` is the wall-clock time since the previous turn of the same session. A
large gap next to a slow first token usually means the upstream had gone to
sleep and the wake is inside `ttft_ms`, not that anything was re-processed.

Reads only; needs no running gateway.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVENTS = ROOT / "gateway_events.jsonl"

# Below this, a delta is too small for the division to say anything. Measured
# on a real log: a 414-token append reported 1 tok/s, because its TTFT was
# dominated by something that had nothing to do with those 414 tokens. 500 of
# them cost ~1.7 s at the slowest rate seen here, which is well clear of noise.
MIN_NEW_TOKENS = 500


def load(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue  # a truncated tail line is normal on a live log
    return rows


def turns(rows: list[dict]) -> list[dict]:
    """One record per request_id, with whatever each event contributed."""
    merged: dict[str, dict] = {}
    for r in rows:
        rid = r.get("request_id")
        if not rid:
            continue
        merged.setdefault(rid, {"request_id": rid})
        for key in ("ts", "lodestar_state", "tokens_in", "duration_ms", "ttft_ms",
                    "queue_wait_ms", "finish_reason", "prompt_lodestar_chars",
                    "session_id", "cached_tokens"):
            if r.get(key) is not None:
                merged[rid][key] = r[key]
        if r.get("event") == "response":
            merged[rid]["answered"] = True
    return [t for t in merged.values() if t.get("answered")]


def attribute_new_tokens(ordered: list[dict]) -> None:
    """Annotate each turn with how much of its prompt the upstream had to read.

    Mutates in place, adding `new_tokens`, `gap_s` and `basis`. Grouped by
    session, because two projects served in the same minute are two
    conversations and the delta between them is not a delta of anything.
    """
    last: dict[str, dict] = {}
    for t in ordered:
        session = str(t.get("session_id") or "?")
        prev = last.get(session)
        tin = t.get("tokens_in") or 0
        t["gap_s"] = (None if prev is None
                      else (t.get("ts", 0) or 0) - (prev.get("ts", 0) or 0))

        reported = t.get("cached_tokens")
        if reported is not None and tin:
            # MEASURED rather than inferred. ExLlamaV3 counts the KV pages the
            # prompt actually hit and reports them at eos; everything below
            # this branch is arithmetic over two consecutive prompts, which is
            # the best that can be done for an upstream that will not say --
            # llama.cpp's `tokens_evaluated` is the whole prompt whether it
            # read it or reused it.
            #
            # It also rescues the two rows the delta cannot describe: the
            # first turn of a session, where inference has to assume every
            # token is new, and a turn after a compaction, where the delta is
            # negative and means nothing while the engine still knows exactly
            # how much it re-read.
            t["basis"] = "engine"
            t["new_tokens"] = max(tin - int(reported), 0)
        elif prev is None:
            t["basis"] = "cold"
            t["new_tokens"] = tin
        else:
            grew = tin - (prev.get("tokens_in") or 0)
            if grew < 0:
                t["basis"] = "rewrit"
                t["new_tokens"] = None
            else:
                t["basis"] = "append"
                t["new_tokens"] = grew
        last[session] = t


def rate_ms_per_1k(turn: dict) -> float | None:
    """TTFT per 1k NEW tokens, or None when the row cannot carry it."""
    ttft, new = turn.get("ttft_ms"), turn.get("new_tokens")
    if ttft is None or not new or new < MIN_NEW_TOKENS:
        return None
    return ttft / (new / 1000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=25)
    ap.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    args = ap.parse_args()

    if not args.events.is_file():
        print(f"no audit log at {args.events}")
        return 1

    ordered = sorted(turns(load(args.events)), key=lambda t: t.get("ts", 0))
    if not ordered:
        print("no completed responses in the audit log")
        return 1

    attribute_new_tokens(ordered)

    measured = [t for t in ordered if t.get("ttft_ms") is not None]
    print(f"{len(ordered)} responses, {len(measured)} with a TTFT measurement.")
    if not measured:
        print("\nNo ttft_ms anywhere: this log predates the instrumentation, or\n"
              "every request took the non-streaming path. Run a streaming\n"
              "request and look again.")
        return 0

    print(f"\n{'time':>8} {'st':>2} {'tok_in':>8} {'new':>8} {'gap':>6} "
          f"{'queue':>7} {'ttft':>7} {'gen':>7} {'ms/1k new':>10} {'tok/s':>7}")
    print("-" * 84)
    for t in ordered[-args.last:]:
        when = datetime.datetime.fromtimestamp(t.get("ts", 0)).strftime("%H:%M:%S")
        ttft, qw = t.get("ttft_ms"), t.get("queue_wait_ms")
        dur, tin = t.get("duration_ms") or 0, t.get("tokens_in") or 0
        # NOT `- qw`: app.py resets stream_start after the slot is acquired
        # ("Generation starts now"), so duration_ms already excludes the queue
        # wait and subtracting it again charged the queue twice. It showed as
        # negative generation time, which is the kind of number a reader edits
        # around instead of trusting.
        gen = dur - (ttft or 0)

        new = t.get("new_tokens")
        new_col = f"{new:,}" if new is not None else t["basis"]
        gap = t.get("gap_s")
        gap_col = "-" if gap is None else f"{gap:.0f}s"
        r = rate_ms_per_1k(t)
        rate_col = f"{r:.0f}" if r is not None else "-"
        tps_col = f"{1000 / r * 1000:,.0f}" if r else "-"

        print(f"{when:>8} {str(t.get('lodestar_state', '-')):>2} {tin:>8,} "
              f"{new_col:>8} {gap_col:>6} "
              f"{('-' if qw is None else f'{qw/1000:.1f}s'):>7} "
              f"{('-' if ttft is None else f'{ttft/1000:.1f}s'):>7} "
              f"{gen/1000:>6.1f}s {rate_col:>10} {tps_col:>7}")

    rates = [(r, t) for t in measured if (r := rate_ms_per_1k(t)) is not None]
    skipped = len(measured) - len(rates)
    if not rates:
        print(f"\nNo row carried a usable delta ({skipped} skipped as cold, "
              f"rewritten, or under {MIN_NEW_TOKENS} new tokens). Run a few "
              f"turns of one conversation and look again.")
        return 0

    # Sort on the rate alone. A bare sort() falls through to comparing the turn
    # dicts whenever two rates tie, and dicts are not orderable — so the whole
    # script died with a TypeError on any log holding two turns with the same
    # tokens_in and ttft_ms, which is what re-sending an unchanged prompt
    # produces. The failure needed real traffic to appear.
    rates.sort(key=lambda pair: pair[0])
    median, _ = rates[len(rates) // 2]
    worst, worst_turn = rates[-1]

    print(f"\nmedian prefill: {median:.0f} ms per 1k new tokens "
          f"({1000 / median * 1000:,.0f} tok/s), over {len(rates)} turns "
          f"({skipped} skipped: cold, rewritten, or tiny)")
    print(f"worst: {worst:.0f} ms/1k ({1000 / worst * 1000:,.0f} tok/s) on "
          f"{worst_turn['new_tokens']:,} new tokens of a "
          f"{worst_turn.get('tokens_in', 0):,}-token prompt")
    print("\nWhat a MISSED prefix looks like: `new` close to `tok_in` on a row\n"
          "whose basis is `append`. That is the whole prompt being re-read.\n"
          "\nWhat is NOT a fault: the rate falling as the conversation grows.\n"
          "Every new token attends over everything before it, so the cost per\n"
          "new token rises with total context by construction. Measured on this\n"
          "rig with the prefix warm throughout: 806 tok/s at 45k, 632 at 58k,\n"
          "602 at 85k, 459 at 107k, ~330 at 140k. Compare a row against that\n"
          "curve, not against a constant.\n"
          "\nA slow first token after a large `gap` is usually the upstream\n"
          "waking up, and the wake is inside ttft_ms. Check the eGPU server log\n"
          "for a load in that window before blaming the cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
