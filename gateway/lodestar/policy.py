"""
One decision per tool call, shared by both request paths.

The streaming and non-streaming paths used to enforce LODESTAR separately, and
they did not agree: streaming blocked a call and then performed the pending
protocol step on the model's behalf, while non-streaming only dropped the call
and moved on. The same request answered differently depending on whether the
client asked for `stream: true`, and only the streaming half had meaningful
test coverage.

So the *decision* lives here, on plain data, and each path only serialises it —
SSE chunks on one side, `tool_calls` JSON on the other.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from gateway.lodestar.enforcer import EnforceResult, enforce_tool_call
from gateway.lodestar.idempotency import (
    IdempotencyResult, check_active_context_idempotency,
)
from gateway.lodestar.state import LodestarState
from gateway.lodestar.synthesis import synthesize_required_step
from gateway.lodestar.tool_names import ACTIVE_CONTEXT, EDIT_TOOLS, get_path

if TYPE_CHECKING:
    from gateway.lodestar.todo import LodestarTodo

__all__ = ["ToolDecision", "decide_tool_call", "idempotency_for"]

_log = logging.getLogger("gateway.lodestar.policy")

# What the gateway does with a call. `reason` is what lands in the audit log.
ALLOW = "allow"
DROP = "drop"
REMINDER = "reminder"
SYNTHESIZE = "synthesize"


@dataclass(frozen=True)
class ToolDecision:
    """What should happen to one completed tool call."""

    action: str
    canonical_name: str | None = None
    reason: str | None = None
    next_key: str | None = None
    next_text: str = ""
    # (tool_name, args_json) when action is SYNTHESIZE.
    synthesized: tuple[str, str] | None = None

    @property
    def blocks_original(self) -> bool:
        """True when the model's own call must not reach the client."""
        return self.action != ALLOW


def idempotency_for(canonical_name: str, params: dict) -> IdempotencyResult | None:
    """IdempotencyResult for an active-context edit, or None.

    Lets the enforcer answer REMINDER instead of allowing a write that would
    change nothing.
    """
    if canonical_name.lower() not in EDIT_TOOLS:
        return None
    path = get_path(params)
    if ACTIVE_CONTEXT not in path:
        return None
    ctx_path = Path(path)
    if not ctx_path.is_file():
        return None
    try:
        return check_active_context_idempotency(ctx_path, last_known_hash=None)
    except (OSError, ValueError):
        return None


def decide_tool_call(
    name: str,
    params: dict,
    state: LodestarState | int,
    *,
    tools: list[dict],
    registered_tools: frozenset[str] | set[str] | None = None,
    todo: "LodestarTodo | None" = None,
    workspace_dir: str | None = None,
    client_body: dict | None = None,
    resolve_name: Callable[[str], str | None] | None = None,
    max_synthesis: int | None = None,
    synthesis_so_far: int = 0,
    rate_limited: bool = False,
) -> ToolDecision:
    """Decide the fate of one completed tool call.

    `resolve_name` maps a model-emitted name onto one the client declared
    (read_file -> read), returning None for a name the client never declared.
    The streaming path passes the emitter's resolver; the non-streaming path
    has already canonicalised names through the schema registry and passes
    nothing.
    """
    state = LodestarState(int(state))
    canonical = resolve_name(name) if resolve_name is not None else name

    if canonical is None:
        # A name the client never declared. Models invent these — the
        # brainstorming skill's `question` tool in the 2026-07-07 session.
        return ToolDecision(DROP, None, reason="hallucinated")

    if state >= LodestarState.READY:
        return ToolDecision(ALLOW, canonical)

    verdict = enforce_tool_call(
        canonical, params, state,
        idempotency_result=idempotency_for(canonical, params),
        registered_tools=frozenset(registered_tools) if registered_tools else None,
        todo=todo,
    )

    if verdict == EnforceResult.ALLOW:
        return ToolDecision(ALLOW, canonical)

    if verdict == EnforceResult.DROP:
        return ToolDecision(DROP, canonical, reason="state_policy")

    if verdict == EnforceResult.INJECT_REMINDER:
        return ToolDecision(REMINDER, canonical, reason="idempotent_edit")

    # INJECT_TODO: the model reached past a pending protocol step. Do that step
    # for it rather than dead-ending the session.
    next_key, next_text = None, ""
    if todo is not None:
        req = todo.next_required()
        if req is not None:
            next_key, next_text = req

    if rate_limited:
        # The per-cycle count above is read from the history the client sends,
        # and compaction erases synthesized calls along with the turns they sat
        # in. This backstop is keyed on wall-clock time, which the client cannot
        # rewind.
        _log.error(
            "LODESTAR LIMIT: workspace is over its synthesis rate — refusing to "
            "synthesize %s for '%s'", next_key, canonical,
        )
        return ToolDecision(DROP, canonical, reason="synthesis_rate_limit",
                            next_key=next_key, next_text=next_text)

    if max_synthesis is not None and synthesis_so_far >= max_synthesis:
        # The gateway has already done this work several times over and the
        # model still is not taking the step. Doing it again would just burn
        # the single upstream slot; the injected directive tells the model to
        # act or explain itself.
        _log.error(
            "LODESTAR LIMIT: %d synthesized calls already in this cycle "
            "(max %d) — refusing to synthesize %s for '%s'",
            synthesis_so_far, max_synthesis, next_key, canonical,
        )
        return ToolDecision(DROP, canonical, reason="synthesis_limit",
                            next_key=next_key, next_text=next_text)

    synthesized = synthesize_required_step(
        next_key,
        tools=tools,
        workspace_dir=workspace_dir,
        client_body=client_body,
        todo=todo,
        model_command=str(params.get("command") or ""),
    )
    if synthesized is not None:
        return ToolDecision(SYNTHESIZE, canonical, reason="force_order",
                            next_key=next_key, next_text=next_text,
                            synthesized=synthesized)

    return ToolDecision(DROP, canonical, reason="force_order",
                        next_key=next_key, next_text=next_text)
