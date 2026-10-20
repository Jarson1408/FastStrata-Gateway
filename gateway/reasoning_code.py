"""Catch the model writing the FILE inside its reasoning instead of emitting it.

The most expensive failure seen so far, and it does not look like a failure
while it happens. Measured 2026-09-01, ACME_HW: in one 24,002-character
`<think>` block the model planned the work, then wrote **eleven thousand
characters of finished Python** — a complete calculator, with a docstring,
argument parsing, seven calculation functions and formatted output — and then
ended the turn without ever emitting a `write` call. All of it was discarded.
It was not confused about what to do. It did the job and put it in the place
the client throws away.

That is the announce-then-stop habit at its most costly. The existing guards
cannot see it: it is not a loop (every line is different), it is not idle, and
by the time `reasoning_budget_chars` notices the size, the work is already
done and about to be lost.

## Telling authored code from a sketched formula

Reasoning legitimately contains formulas and fragments — the same block has
`Rg = (Vcc - Vbe_sat) / Ipeak` sitting in a paragraph of analysis, and cutting
that would be cutting real thinking. What separates them is not the presence of
code but its RUN: source code arrives as consecutive lines, prose does not.
Measured across that block:

    engineering analysis      longest run of code-like lines:   0 - 2
    the calculator itself     longest run:                     40 - 234

No overlap, and the gap is two orders of magnitude wide, so the threshold does
not have to be delicate. Blank lines do not break a run — source has them —
but a line of prose does.

## What happens then

The turn is cut and the model is told to emit the call with the content it was
about to write. It is cut EARLY, a few hundred characters into the code, so
little work is thrown away: on the measured block a 25-line threshold fires
around character 9,500, saving roughly sixty per cent of the turn.

The gateway does NOT try to salvage the code and synthesize the write itself.
It has the text, but not the path, not the certainty that the code is complete,
and no way to be sure it would not overwrite something. Fabricating a file
write out of a reasoning block is the one mistake here that could destroy work
rather than waste time.
"""
from __future__ import annotations

import re

__all__ = ["DEFAULT_MIN_LINES", "CHECK_EVERY_CHARS", "find_authored_code"]

# Consecutive code-like lines before the turn is cut. The measured gap is
# 2 versus 40, so this sits nowhere near either edge — and a snippet the model
# is merely quoting back from a file it read is usually shorter than this.
DEFAULT_MIN_LINES = 25

# How much new reasoning has to arrive before looking again.
CHECK_EVERY_CHARS = 1000

# A line that belongs to source rather than to prose. Deliberately syntactic:
# indentation, block keywords, decorators, fences, closers.
_CODE_RE = re.compile(
    r"^(?:\s{4,}\S"
    r"|(?:def|class|import|from|return|if|for|while|try|except|with|elif|else)\b"
    r"|#!/"
    r"|['\"]{3}"
    r"|@\w+"
    r"|[)\]}])"
)


def find_authored_code(reasoning: str,
                       min_lines: int = DEFAULT_MIN_LINES) -> tuple[int, str] | None:
    """(length of the longest code run, its first line) once it is long enough.

    None while the reasoning is prose, formulas, or a fragment short enough to
    be a quotation rather than a file being authored.
    """
    if not reasoning or min_lines <= 0:
        return None

    best = current = 0
    best_first = current_first = ""
    for raw in reasoning.splitlines():
        if not raw.strip():
            continue                     # source has blank lines; prose breaks a run
        if _CODE_RE.match(raw):
            if current == 0:
                current_first = raw.strip()
            current += 1
            if current > best:
                best, best_first = current, current_first
        else:
            current = 0

    if best >= min_lines:
        return best, best_first[:200]
    return None
