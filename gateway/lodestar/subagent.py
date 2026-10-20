"""LODESTAR subagent detection - header, gateway marker, then anchored opener.

A delegated subagent must skip the protocol entirely. When one went undetected
it ran the full sequence including Phase D, clearing active-context.md while
the main session's task was still in progress (2026-07-09 incident), so the
detection has to stay capable.

Three signals, narrowing as they get weaker:

1. `X-Lodestar-Subagent: 1` — explicit, set by the SDD orchestrator.
2. SUBAGENT_MARKER anywhere in a system/user message — the gateway appends this
   to every `task` prompt it rewrites, so it is unambiguous and self-produced.
3. A legacy opener at the START of the conversation's first system/user
   message — for subagents spawned by something other than this gateway
   (superpowers opens delegated prompts with "You are implementing Task N of
   ...").

Signal 3 used to be a substring match anywhere in any system or user message,
against a list that included the bare filename "sdd_helper". That made an
ordinary sentence a silent kill switch for the whole protocol: asking about
"sprawdz plik sdd_helper.py" — a file in this very repository — turned off
Phase A, the memory search, the Goal commit and the closing lesson, with
nothing in the log to say why. Anchoring the match to the opening of the first
message keeps real delegated prompts detected while a mid-conversation mention
can no longer trigger it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Request

# The tag that opens gateway/prompts/subagent_directive.md. Defined here rather
# than imported from synthesis.py, which imports this module.
SUBAGENT_MARKER = "[LODESTAR SUBAGENT DIRECTIVE]"

# Role declarations that open a delegated prompt. Matched as a prefix only.
_LEGACY_OPENERS = (
    "you are a subagent",
    "you are a sub-agent",
    "you are an implementer",
    "you are implementing task",
)


def _has_gateway_marker(messages: list[dict]) -> bool:
    """True if any message carries the directive the gateway appends to tasks."""
    for m in messages:
        if m.get("role") not in ("system", "user"):
            continue
        if SUBAGENT_MARKER in str(m.get("content") or ""):
            return True
    return False


def _opens_with_role_declaration(messages: list[dict]) -> bool:
    """True if the conversation OPENS with a delegated-prompt role declaration.

    Only the first system/user message counts. A delegated session begins with
    its directive; anything later is the user talking about subagents, not a
    session being one.
    """
    for m in messages:
        role = m.get("role")
        if role not in ("system", "user"):
            continue
        text = str(m.get("content") or "").lstrip().lower()
        return text.startswith(_LEGACY_OPENERS)
    return False


def is_subagent_request(request: "Request | None", messages: list[dict]) -> bool:
    if request is not None:
        header = request.headers.get("x-lodestar-subagent", "")
        if header and header.strip().lower() not in ("0", "false", ""):
            return True
    return _has_gateway_marker(messages) or _opens_with_role_declaration(messages)


# ---------------------------------------------------------------------------
# Turn budget
# ---------------------------------------------------------------------------
#
# A subagent skips the whole protocol, and every limit the gateway had was
# derived from LODESTAR cycle state — so nothing bounded one at all. Measured
# 2026-09-01: a subagent ran 51 tool calls over 2 h 52 m, on a rig with a
# single upstream slot, for a task whose whole output is a paragraph of
# findings. (`max_cycle_turns` would not have helped even if it applied: it
# only writes a log line.)
#
# The budget is counted here rather than in the state machine because it needs
# nothing from the protocol. A subagent's own conversation is the whole
# measure.

# Turns of exploration before the model is told to report. Generous for a
# search task: the runaway above was double this before anyone noticed.
DEFAULT_TURN_BUDGET = 25

# How far past the budget the model may go while still being asked nicely.
# After this its tool calls stop being executed, because a subagent that has
# ignored ten warnings is not going to stop on the eleventh.
BUDGET_GRACE = 10

SOFT = "soft"
HARD = "hard"


def turn_count(messages: list[dict]) -> int:
    """Assistant turns this subagent has taken.

    The agent loop lives in the client, so there is no iteration counter to
    read — but the history the client sends carries one turn per assistant
    message, which is the same number by another name.
    """
    return sum(1 for m in messages or [] if m.get("role") == "assistant")


def budget_state(messages: list[dict], budget: int = DEFAULT_TURN_BUDGET,
                 grace: int = BUDGET_GRACE) -> tuple[str | None, int]:
    """(None | SOFT | HARD, turns taken).

    SOFT asks for the report and leaves the tools working, because a subagent
    one turn over budget may be about to finish anyway. HARD stops executing
    its calls, which is the only thing that actually bounds a model that has
    read the warning and carried on.
    """
    turns = turn_count(messages)
    if budget <= 0:
        return None, turns
    if turns >= budget + max(0, grace):
        return HARD, turns
    if turns >= budget:
        return SOFT, turns
    return None, turns
