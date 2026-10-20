"""Who gets the single upstream slot first.

This rig serves one request at a time, and `asyncio.Semaphore` is FIFO -- so a
title generator that arrived a moment earlier holds the slot while the agentic
turn a human is watching waits behind it.

Measured 2026-09-12 from `gateway_events.jsonl`:

    requests with tools .......  2 069   (27%)
    requests without .......... 5 588   (73%)   titles, summaries, graph builds

    queue wait   p50      0 ms
                 p90      0 ms
                 p99    140 s
                 max   2 612 s   (43.5 minutes)

So the queue is empty almost always and catastrophic when it is not: 54
requests waited over five seconds, 23 over thirty, 13 over two minutes. Those
are the ones worth reordering, and the median request cannot tell the
difference either way.

`PrioritySemaphore` is `asyncio.Semaphore` with one extra argument, so the call
sites keep their `asyncio.wait_for(sem.acquire(), timeout=...)` shape and the
existing timeout stays the backstop. Within a priority it is still FIFO: equal
requests are served in the order they arrived, because a scheduler that
reorders equals gives nobody a guarantee.

**Starvation is bounded by the timeout, not by the scheduler.** A low-priority
request can in principle be passed over indefinitely, and it is not, for a
reason specific to this workload rather than a clever rule: agentic turns are
SERIAL -- one session, one turn at a time -- so the slot frees between them.
`semaphore_timeout` is what turns the theoretical case into an error instead of
a hang. If this ever runs several agentic sessions at once, that assumption is
the thing to revisit, and aging the queue is the fix.
"""
from __future__ import annotations

import asyncio
import heapq
import itertools

#: A human is watching a loop: an agentic turn, or a compaction the client is
#: blocked on mid-session.
INTERACTIVE = 0

#: Nobody is blocked: title generation, graph building, a background summary.
BACKGROUND = 10


class PrioritySemaphore:
    """A counting semaphore that wakes the most important waiter first.

    Lower `priority` wins. `acquire()` with no argument behaves exactly like
    `asyncio.Semaphore.acquire()` at INTERACTIVE, so a caller that does not
    care is never demoted by accident.
    """

    def __init__(self, value: int = 1) -> None:
        if value < 0:
            raise ValueError("semaphore initial value must be >= 0")
        self._value = value
        self._waiters: list[tuple[int, int, asyncio.Future[None]]] = []
        self._counter = itertools.count()

    # -- introspection, for /health and the tests ---------------------------
    @property
    def value(self) -> int:
        return self._value

    def waiting(self) -> int:
        return sum(1 for _p, _s, f in self._waiters if not f.done())

    def waiting_at(self, priority: int) -> int:
        return sum(1 for p, _s, f in self._waiters
                   if p == priority and not f.done())

    # -- the semaphore ------------------------------------------------------
    async def acquire(self, priority: int = INTERACTIVE) -> bool:
        # Taking a free slot only when nobody is queued is what keeps a stream
        # of arrivals from stepping over waiters indefinitely. The two states
        # coexist only momentarily, because release() drains the queue.
        if self._value > 0 and not self._waiters:
            self._value -= 1
            return True

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        entry = (priority, next(self._counter), fut)
        heapq.heappush(self._waiters, entry)

        try:
            await fut
        except asyncio.CancelledError:
            # Two different situations, and telling them apart is the whole
            # correctness of this class. Either the wait was cancelled before
            # anything was handed over -- drop out of the queue -- or the slot
            # had ALREADY been granted in the same tick, in which case it
            # belongs to somebody else now and must go back. A cancelled
            # waiter that swallows its grant leaks the only slot there is, and
            # every later request times out against a semaphore that will
            # never be released.
            if fut.done() and not fut.cancelled():
                self._value += 1
                self._wake()
            else:
                self._discard(entry)
            raise
        return True

    def release(self) -> None:
        self._value += 1
        self._wake()

    def _wake(self) -> None:
        while self._value > 0 and self._waiters:
            _priority, _seq, fut = heapq.heappop(self._waiters)
            if fut.done():
                # Cancelled while queued. Its slot was never taken, so simply
                # move on to the next one.
                continue
            self._value -= 1
            fut.set_result(None)

    def _discard(self, entry: tuple[int, int, asyncio.Future[None]]) -> None:
        try:
            self._waiters.remove(entry)
        except ValueError:
            return
        heapq.heapify(self._waiters)


def request_priority(*, has_tools: bool, is_summary: bool) -> int:
    """Where a request belongs in the queue.

    A compaction counts as INTERACTIVE even though it registers no tools: the
    client runs it in the middle of a session and will not continue until it
    comes back, so treating it as background would put the user behind a title
    generator at exactly the moment they are already waiting.
    """
    return INTERACTIVE if (has_tools or is_summary) else BACKGROUND
