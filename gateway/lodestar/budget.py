"""
A synthesis budget the client cannot refill.

`max_synthesis_per_cycle` counts synthesized calls by scanning the history for
tool_call ids starting `syn_`. That works only while the history is a faithful
record — and it is not. OpenCode compacts long conversations, replacing earlier
turns with a summary, and the synthesized calls go with them. The count returns
to zero, the budget silently refills, and a model that never converges can be
carried by the gateway indefinitely: five synthesized calls, compaction, five
more, forever. The limit reads as a hard stop and is really a speed bump.

No history-derived key fixes this, because compaction rewrites the history the
key would be derived from. So the backstop is keyed on the one thing the client
cannot rewrite: wall-clock time. This tracks, per workspace, when the gateway
acted on the model's behalf, and refuses once that exceeds a rate.

The guarantee changes shape deliberately. Per-cycle counting answers "has this
task had too much help?" and stays as the primary, precise signal. This answers
"has this workspace had too much help lately?", which is a weaker statement
about any single task but a stronger one about the harm actually worth
preventing — a loop holding the single upstream slot for hours.

Held in memory, in app.state. A gateway restart clears it; that is an
operator-initiated event, not something the model can cause, and the per-cycle
count still covers whatever history survives.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque

__all__ = ["SynthesisBudget"]

# Workspaces tracked at once. Bounded so a client sending varied paths cannot
# grow this without limit; the least recently used entry is dropped.
_MAX_TRACKED_WORKSPACES = 64

_ANONYMOUS = "(no workspace)"


class SynthesisBudget:
    """Rolling per-workspace count of calls the gateway made for the model."""

    def __init__(self, max_per_window: int = 20, window_seconds: float = 3600.0):
        self._max = max_per_window
        self._window = window_seconds
        self._events: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(workspace: str | None) -> str:
        return workspace or _ANONYMOUS

    def _prune(self, key: str, now: float) -> deque[float]:
        stamps = self._events.get(key)
        if stamps is None:
            stamps = deque()
            self._events[key] = stamps
        cutoff = now - self._window
        while stamps and stamps[0] < cutoff:
            stamps.popleft()
        self._events.move_to_end(key)
        while len(self._events) > _MAX_TRACKED_WORKSPACES:
            self._events.popitem(last=False)
        return stamps

    def record(self, workspace: str | None) -> int:
        """Note one synthesized call. Returns the count now inside the window."""
        now = time.time()
        with self._lock:
            stamps = self._prune(self._key(workspace), now)
            stamps.append(now)
            return len(stamps)

    def used(self, workspace: str | None) -> int:
        with self._lock:
            return len(self._prune(self._key(workspace), time.time()))

    def exhausted(self, workspace: str | None) -> bool:
        return self.used(workspace) >= self._max

    def remaining(self, workspace: str | None) -> int:
        return max(0, self._max - self.used(workspace))

    def reset(self, workspace: str | None = None) -> None:
        """Clear one workspace, or all of them. For tests and operator use."""
        with self._lock:
            if workspace is None:
                self._events.clear()
            else:
                self._events.pop(self._key(workspace), None)

    @property
    def max_per_window(self) -> int:
        return self._max

    @property
    def window_seconds(self) -> float:
        return self._window
