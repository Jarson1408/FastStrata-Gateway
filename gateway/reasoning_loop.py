"""Spot a degenerate <think> block by how little of it is new.

`reasoning_budget_chars` catches a runaway reasoning block by its SIZE, which
means letting the whole budget run out first. This catches the tail of one by
its shape, which is visible immediately.

## Density, not count

The first version of this module counted repeats: three identical lines and it
cut the turn. Measured against the real 24,002-character block from
2026-09-01, that fired at character 4,500 — while the model was in the middle
of genuine engineering analysis, having merely restated its intention three
times over several thousand characters of new content. It would have cut off
the work.

What separates restating an intention from looping is whether anything NEW
appears between the repeats. Measured across that same block, in windows of
2,000 characters:

    up to 18,000   0.95 - 1.00 distinct lines   substantive work
    up to 22,000   0.67                          degenerating
    up to 24,000   0.30                          loop

Substantive reasoning never dropped below 0.90 anywhere in nineteen thousand
characters; the loop reached 0.30. There is no overlap, so the threshold sits
in a wide gap rather than on a knife edge.

## Why here and not in the sampler

DRY exists for this and was in the coding presets for one day. It suppresses
repeated SEQUENCES — which is also what a file path is, repeated verbatim from
earlier context — so it corrupted every long path the model touched. See the
note in config.py. Reading whole lines cannot confuse a sentence with an
identifier, and nothing here changes what the model is allowed to say next,
only when the turn is cut.

## What this does not solve

In the block it was written for, the loop began at character 19,735 — so even a
perfect detector saves the last fifth of that turn. The other four fifths were
the model doing the work and putting it in the wrong place: eleven thousand
characters of finished Python written inside `<think>` and then discarded,
because the turn ended without a tool call. That is the announce-then-stop
habit at its most expensive, and it is not this module's problem to fix.
"""
from __future__ import annotations

import re
from collections import Counter

__all__ = ["DEFAULT_MAX_DISTINCT_RATIO", "MIN_LINE_CHARS", "MIN_LINES",
           "WINDOW_CHARS", "CHECK_EVERY_CHARS", "find_loop"]

# Above this share of distinct lines the model is still saying new things.
# Chosen from the measurement above: work never fell below 0.90, the loop
# reached 0.30, so anything in between separates them — 0.6 is the middle of
# the gap rather than the edge of either side.
DEFAULT_MAX_DISTINCT_RATIO = 0.6

# Lines shorter than this are connectives and bullets, and repeat innocently.
MIN_LINE_CHARS = 20

# Too few lines to judge: a short window of two repeated sentences says
# nothing about whether the model is stuck.
MIN_LINES = 10

# Recent tail only. A conclusion restated at the end of a long block is a
# summary; the same handful of sentences cycling within a window this size is
# a loop. Window and interval were tuned against BOTH real blocks at once —
# the 24,002-character calculator block whose loop starts at 82%, and the
# 4,009-character unicode-fix block whose loop starts at 30%. Every setting
# tried was a true positive; these are simply the earliest. Going below 1,500
# was rejected: a shorter window often holds fewer than MIN_LINES qualifying
# lines, and a guard that goes quiet is worse than one that is late.
WINDOW_CHARS = 1500

# How much new reasoning has to arrive before looking again — a handful of
# scans per turn rather than one per streamed chunk.
CHECK_EVERY_CHARS = 500

_TAG_RE = re.compile(r"<[^>]{1,20}>")
_SEGMENT_RE = re.compile(r"[\r\n]+|(?<=[.!?])\s+")


def _normalise(line: str) -> str:
    """Tags and spacing removed, case folded: the same sentence twice is the
    same sentence however it was written."""
    return " ".join(_TAG_RE.sub(" ", line).split()).lower()


def find_loop(reasoning: str,
              max_distinct_ratio: float = DEFAULT_MAX_DISTINCT_RATIO,
              min_chars: int = MIN_LINE_CHARS,
              min_lines: int = MIN_LINES,
              window: int = WINDOW_CHARS) -> tuple[str, int] | None:
    """The line the model is cycling on and how often, or None.

    None is the answer for everything except a tail that has stopped saying
    anything new: too little text to judge, mostly distinct lines, or a
    threshold that disables the check.
    """
    if not reasoning or max_distinct_ratio <= 0:
        return None
    tail = reasoning[-window:] if window > 0 else reasoning

    counts: Counter[str] = Counter()
    originals: dict[str, str] = {}
    for raw in _SEGMENT_RE.split(tail):
        norm = _normalise(raw)
        if len(norm) < min_chars:
            continue
        counts[norm] += 1
        # The CLEANED line: reasoning_text carries the literal `<think>`
        # opener, and quoting that back at the model would be reporting the
        # gateway's own markup as its words.
        originals.setdefault(norm, _TAG_RE.sub("", raw).strip())

    total = sum(counts.values())
    if total < min_lines:
        return None
    if len(counts) / total > max_distinct_ratio:
        return None

    norm, n = counts.most_common(1)[0]
    if n < 2:
        return None
    return originals[norm], n
