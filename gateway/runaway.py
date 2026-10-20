"""One question, asked by every path that reads a generation.

Three guards decide that a turn has stopped being worth its slot: source code
written inside `<think>`, a tail that has stopped saying anything new, and a
plain size budget. Their DETECTORS already live in `reasoning_code.py` and
`reasoning_loop.py` and are shared. What was not shared is the POLICY around
them -- the config knobs, the `not is_summary` and `in_thinking` conjunctions,
and the "check every N characters" throttles -- and policy that is written out
by hand in each caller is policy that will be missing from one.

It was missing from one. Measured 2026-09-12: `_non_stream_completion` had
**none** of the four guards, not the reasoning budget, not the loop detector,
not the code detector and not `stream_max_duration`. Seven turns went through
it to the `max_n_predict` ceiling, 16,384 tokens each, a median of fourteen
minutes of the single upstream slot apiece -- about an hour and a half of it
with nothing watching. That is audit finding W-2026-03 in the one path nobody
had looked at, and it is the third time this project has paid for a guard
written out twice.

So the QUESTION lives here and the ACTION stays with the caller, which is the
same split `_upstream_events` uses: the streaming path can hand a runaway to
the silent retry, while a non-streaming one has no seam to hide a retry behind
and simply ends the attempt. What must not differ is whether the question gets
asked.

**`in_thinking` is load-bearing in all of them.** Reasoning that has closed
its block, or that has begun emitting a tool call, is doing work; cutting
mid-`<tool_call>` truncates the arguments and hands the client a JSON parse
error. A caller that cannot answer that question honestly must pass False.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from gateway import reasoning_code, reasoning_loop

if TYPE_CHECKING:
    from gateway.config import GatewayConfig

#: Ordered as they fire. Code first, because it is the costliest failure and
#: the only one where the model is not stuck -- it is doing the work in the
#: place the client discards, and cutting early is what saves it. The budget
#: last, because it has to let the whole allowance run out before it can know.
CODE = "code"
LOOP = "loop"
BUDGET = "budget"

#: The fourth, and the only one that watches the half of a turn the other
#: three cannot see. See `RunawayWatch.check_content`.
CONTENT_LOOP = "content_loop"

#: The fifth: repeated tool calls suppressed by loop_breaker.
TOOL_LOOP = "tool_loop"


@dataclass(frozen=True)
class Verdict:
    """Which guard tripped, its evidence, and how much reasoning there was."""
    guard: str
    detail: Any
    chars: int


#: Which cuts say "this model is stuck in a way a different temperature could
#: shake it out of". CODE is deliberately absent: it means the model is doing
#: the work in the place the client discards, which is not a loop and which a
#: different temperature does not address -- it would only degrade the code
#: that is about to be written somewhere useful.
COUNTS_TOWARD_STREAK = (LOOP, BUDGET, CONTENT_LOOP, TOOL_LOOP)


class GuardCutStreak:
    """How many turns in a row a runaway guard has ended, per session.

    **Why this exists.** Adaptive sampling answers a loop by moving the
    temperature, and it learns about loops from `count_retry_depth`, which
    asks "same tool, same resource, failed again". A turn the gateway CUT has
    no tool call at all, so that count returns 0 on it -- the chain breaks on
    `if not last_call["failed"]`. A model looping inside `<think>` was
    therefore cut three turns running at the same temperature, and the
    escalation written precisely for that case could never see it. The two
    loop-detection systems were blind to each other.

    **In-process state, not derived from the conversation.** The gateway owns
    the fact that it cut a turn. Re-deriving it from the `[SYSTEM GUARD]` text
    in the history would make that marker forgeable -- and this project has
    deliberately left the prose markers un-neutralized on the grounds that they
    mislead the MODEL and nothing else. A gateway DECISION made from them stops
    being harmless. Same reasoning, and the same shape, as `SynthesisBudget`.

    Consecutive is the whole property, so `begin()` closes the books on the
    previous turn: a session whose last request was not cut starts again at
    zero. That is one call site rather than a `clear()` somebody has to
    remember on every path by which a turn can end.
    """

    #: Sessions tracked at once. A streak is a few hundred bytes and matters
    #: for minutes; this is about not growing without bound, not capacity.
    MAX_SESSIONS = 64

    def __init__(self) -> None:
        self._streak: dict[str, int] = {}
        self._last_request: dict[str, str] = {}
        self._cut_requests: dict[str, str] = {}

    def begin(self, session: str | None, request_id: str) -> int:
        """The streak this request inherits, having closed the previous turn.

        Returns 0 for an unknown session, which is the answer for almost every
        request ever served.
        """
        if not session:
            return 0
        previous = self._last_request.get(session)
        if previous is not None and self._cut_requests.get(session) != previous:
            self._streak[session] = 0
        self._last_request[session] = request_id
        self._trim()
        return self._streak.get(session, 0)

    def record(self, session: str | None, request_id: str, guard: str) -> None:
        """One turn was ended by `guard`. Idempotent per request.

        Idempotent because a watch is per ATTEMPT and a silent retry makes a
        second attempt inside one client request -- while what is counted here
        is TURNS. Two cuts in one turn are one cut.
        """
        if not session or guard not in COUNTS_TOWARD_STREAK:
            return
        if self._cut_requests.get(session) == request_id:
            return
        self._cut_requests[session] = request_id
        self._streak[session] = self._streak.get(session, 0) + 1
        self._trim()

    def depth(self, session: str | None) -> int:
        return self._streak.get(session or "", 0)

    def _trim(self) -> None:
        while len(self._last_request) > self.MAX_SESSIONS:
            oldest = next(iter(self._last_request))
            self._last_request.pop(oldest, None)
            self._streak.pop(oldest, None)
            self._cut_requests.pop(oldest, None)


class RunawayWatch:
    """The three reasoning guards, with their throttles, for one generation.

    One instance per attempt. `reset()` after a silent retry: the new
    generation must be judged on its own text, exactly as `TurnState.PERSIST`
    already prescribes for the counters that survive.

    `on_cut` is called once, with the guard's name, the first time this watch
    trips. It is how a cut reaches `GuardCutStreak` from the ONE place every
    verdict is produced: five readers call `check()`, and an increment written
    out at each of them is an increment that would be present at four. It is
    optional so the many tests that build a watch directly keep working, and
    `TestEveryWatchReportsItsCuts` reads app.py as SOURCE so a third
    construction site cannot quietly stop reporting.
    """

    def __init__(self, config: GatewayConfig, is_summary: bool,
                 on_cut: Callable[[str], None] | None = None) -> None:
        self._config = config
        self._is_summary = is_summary
        self._on_cut = on_cut
        self._code_checked_at = 0
        self._loop_checked_at = 0
        self._content_checked_at = 0
        self._reported = False

    def reset(self) -> None:
        # `_reported` deliberately survives: it is per TURN, and a silent retry
        # is a second attempt at the same turn rather than a new one.
        self._code_checked_at = 0
        self._loop_checked_at = 0
        self._content_checked_at = 0

    def check(self, reasoning: str, in_thinking: bool) -> Verdict | None:
        """The first guard to trip on this text, or None.

        Cheap enough to call on every chunk: the two detectors run at most
        once per CHECK_EVERY_CHARS of new reasoning, and the budget is a
        length comparison.
        """
        config = self._config
        if self._is_summary or not in_thinking:
            # A compaction's guard message would BECOME the compacted context
            # the client stores, which is worse than a long <think>. See
            # "A Flag That Cannot Be True Is Not Evidence".
            return None

        chars = len(reasoning)

        if (config.reasoning_code_lines
                and chars - self._code_checked_at >= reasoning_code.CHECK_EVERY_CHARS):
            self._code_checked_at = chars
            hit = reasoning_code.find_authored_code(
                reasoning, config.reasoning_code_lines)
            if hit is not None:
                return self._trip(Verdict(CODE, hit, chars))

        if (config.reasoning_loop_max_distinct
                and chars - self._loop_checked_at >= reasoning_loop.CHECK_EVERY_CHARS):
            self._loop_checked_at = chars
            hit_loop = reasoning_loop.find_loop(
                reasoning, config.reasoning_loop_max_distinct)
            if hit_loop is not None:
                return self._trip(Verdict(LOOP, hit_loop, chars))

        if config.reasoning_budget_chars and chars > config.reasoning_budget_chars:
            return self._trip(Verdict(BUDGET, chars, chars))

        return None

    def check_stream(self, reasoning: str, in_thinking: bool,
                     content: str, calls_started: bool) -> Verdict | None:
        """Both questions, so a reader cannot ask half of one.

        Every reader of a generation calls THIS. `check` and `check_content`
        stay public because they are separately testable, but a call site that
        asks only the reasoning half is a call site that goes blind the moment
        the turn closes its `<think>` -- which is what happened on 2026-09-14
        and cost a 22-call loop. `TestEveryReaderAsksBothHalves` counts the
        call sites in SOURCE for the same reason
        `TestEveryReaderGetsTheEmbeddings` does.
        """
        verdict = self.check(reasoning, in_thinking)
        if verdict is not None:
            return verdict
        return self.check_content(content, calls_started)

    def check_content(self, content: str, calls_started: bool) -> Verdict | None:
        """A loop in what the model is SAYING, or None.

        **Why this is a separate question from `check`.** Every guard in
        `check` is conjunctive on `in_thinking`, and that conjunction is
        correct for all three of them: reasoning that has closed its block is
        doing work. But it also means those guards stop at exactly the moment
        the turn starts producing the answer -- and every continuation path in
        `app.py` builds its parser with `start_in_thinking=False`, so on a
        continuation `check` is asked on every tick and returns None on every
        tick, by construction.

        Measured 2026-09-14: a turn hit the answer rescue, the rescue handed
        off to the content continuation, and the continuation then wrote
        *"Now let me apply all the edits to the report."* six times over while
        emitting **22 tool calls** in 170 seconds. `RunawayWatch` was watching
        it the whole way and could not fire. OpenCode's own `doom_loop`
        stopped it instead -- which is the outcome this project calls the
        worse one, because a client can only ask the USER.

        **The gate here is a tool call, not the think block.** Cutting a turn
        whose call is already on the wire truncates its arguments and hands
        the client a JSON parse error, which is the one trade this project
        never makes. A turn that has started calling tools is doing work and
        is left alone; what this watches is a turn that is only talking.
        """
        config = self._config
        if self._is_summary or calls_started:
            return None
        if not config.content_loop_max_distinct:
            return None

        chars = len(content)
        if chars - self._content_checked_at < reasoning_loop.CHECK_EVERY_CHARS:
            return None
        self._content_checked_at = chars

        hit = reasoning_loop.find_loop(content, config.content_loop_max_distinct)
        if hit is None:
            return None
        return self._trip(Verdict(CONTENT_LOOP, hit, chars))

    def _trip(self, verdict: Verdict) -> Verdict:
        """Report the cut once, then hand the verdict back unchanged."""
        if self._on_cut is not None and not self._reported:
            self._reported = True
            self._on_cut(verdict.guard)
        return verdict


def content_and_calls(text: str, started_in_think: bool) -> tuple[str, bool]:
    """(what the model has SAID, whether a tool call has begun) for raw text.

    The companion to `open_think_text`, for the non-streaming path, which has
    no FSM to ask. Content is everything after the reasoning block closed;
    while the block is still open there is no content yet.

    The tool-call question is answered from the opener appearing in the text
    at all, which is deliberately pessimistic: a reader that cannot be sure
    must say a call HAS started, because the cost of being wrong that way is a
    guard that stays quiet, and the cost of being wrong the other way is a
    truncated tool call reaching the client as a parse error.
    """
    if started_in_think:
        _head, closer, rest = text.partition("</think>")
        content = rest if closer else ""
    else:
        opener = text.find("<think>")
        if opener < 0:
            content = text
        else:
            _head, closer, rest = text[opener + len("<think>"):].partition("</think>")
            content = rest if closer else ""
    return content, ("<tool_call>" in text)


def open_think_text(text: str, started_in_think: bool) -> tuple[str, bool]:
    """(reasoning so far, is the block still open) for a raw completion string.

    The streaming path asks the FSM; a path that only has the accumulated text
    has to work it out, and the answer is simple because the prompt prefilled
    the opener: everything up to `</think>` is reasoning, and the block is open
    until that closer arrives.

    Returns ("", False) when no block was ever opened, which is what makes a
    caller that cannot answer honestly pass False by construction rather than
    by remembering to.
    """
    if started_in_think:
        head, closer, _rest = text.partition("</think>")
        return head, not closer
    opener = text.find("<think>")
    if opener < 0:
        return "", False
    head, closer, _rest = text[opener + len("<think>"):].partition("</think>")
    return head, not closer
