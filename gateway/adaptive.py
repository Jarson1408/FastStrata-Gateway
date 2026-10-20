"""
Adaptive sampling — dynamically adjust sampling parameters based on conversation patterns.

Analyzes the incoming message history to detect retry loops, edit failures,
and context pressure, then adjusts temperature and other parameters to
improve model accuracy.

Design:
  - Pure function: analyze(messages, base_params) → adjusted_params
  - No state between requests (each request gets the full conversation)
  - Adjustments are conservative (small deltas, never below 0.0)
  - All adjustments are logged for debugging
"""
from __future__ import annotations

import logging
import re
from typing import Any

from gateway.lodestar.tool_names import (
    READ_TOOLS, SEARCH_TOOLS, get_resource, parse_args,
)

_log = logging.getLogger("gateway.adaptive")

# Tools whose "not found" / "no match" is an ANSWER, not a failure.
#
# This used to be a hand-written set, spelled out TWICE in this file, and it
# named `grep_search`, `read_file`, `list_dir`, `view_file_outline` --
# Windsurf's vocabulary. The client this gateway actually serves calls them
# `grep`, `read`, `glob`, `list`, so the exemption covered NONE of the tools in
# use and a search that legitimately found nothing was counted as an error.
# Measured 2026-09-12: three empty greps took the next generation's
# temperature from 0.3 to 0.15, 227 times over five weeks.
#
# One definition, in gateway/lodestar/tool_names.py, which had the right
# answer the whole time. A local copy is how this went wrong; do not make
# another one.
_NOT_A_FAILURE = READ_TOOLS | SEARCH_TOOLS

# ---------------------------------------------------------------------------
# Pattern detectors
# ---------------------------------------------------------------------------

# Common error patterns in tool results indicating edit failures
_EDIT_FAIL_PATTERNS = re.compile(
    r"no match|not found|does not match|didn['\u2019]t match|"
    r"could not find|cannot find|string not found|"
    r"oldString.*not found|old_string.*not found|"
    r"no such file|failed to apply|edit failed|"
    r"ENOENT|FileNotFoundError|"
    r"JSON parsing failed|JSON Parse error|Unterminated string|invalid input",
    re.IGNORECASE,
)

# Patterns indicating tool call was rejected (by client)
_TOOL_REJECT_PATTERNS = re.compile(
    r"invalid.*tool|unavailable tool|unknown tool|"
    r"tool.*not found|tool.*not available",
    re.IGNORECASE,
)


def _count_recent_tool_failures(messages: list[dict], window: int = 10) -> int:
    """Count tool result messages containing error patterns in the last N messages."""
    failures = 0
    recent = messages[-window:] if len(messages) > window else messages

    for m in recent:
        if m.get("role") != "tool":
            continue
        content = m.get("content") or ""
        tool_name = m.get("name", "")
        
        is_rejected = bool(_TOOL_REJECT_PATTERNS.search(content))
        is_edit_failed = False
        if tool_name not in _NOT_A_FAILURE:
            if tool_name in ("bash", "run_command") and "loading weights" in content.lower():
                pass
            else:
                is_edit_failed = bool(_EDIT_FAIL_PATTERNS.search(content))
            
        if is_rejected or is_edit_failed:
            failures += 1
    return failures


def _count_recent_tool_calls(messages: list[dict], window: int = 10) -> int:
    """Count assistant messages with tool_calls in the last N messages."""
    count = 0
    recent = messages[-window:] if len(messages) > window else messages
    for m in recent:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            count += 1
    return count


def _detect_retry_loop(messages: list[dict], window: int = 8) -> bool:
    """Detect if the model is retrying the same tool call repeatedly."""
    return count_retry_depth(messages, window=window) >= 3


def _estimate_context_pressure(messages: list[dict]) -> float:
    """Estimate context pressure (0.0 = light, 1.0 = heavy).

    Based on message count and total content length.
    """
    total_len = sum(len(m.get("content") or "") for m in messages)
    msg_count = len(messages)

    # Heuristic: ~4 chars per token, heavy starts at ~50k tokens
    token_estimate = total_len / 4
    pressure = min(1.0, token_estimate / 100_000)

    # Also factor in message count (many short messages = complex conversation)
    msg_pressure = min(1.0, msg_count / 100)

    return max(pressure, msg_pressure)


# ---------------------------------------------------------------------------
# Tool call status analysis
# ---------------------------------------------------------------------------

def _analyze_tool_sequence(messages: list[dict]) -> list[dict]:
    """Analyze the sequence of tool calls and their corresponding status.

    Aligns assistant tool calls with tool responses (by tool_call_id or sequence order)
    to check if a tool call failed and extract target resources (like file paths).
    """
    tool_responses = {}  # tool_call_id -> content
    tool_order = []  # list of tool content for order-based fallback

    for m in messages:
        if m.get("role") == "tool":
            content = m.get("content") or ""
            t_id = m.get("tool_call_id")
            if t_id:
                tool_responses[t_id] = content
            tool_order.append(content)

    sequence = []
    tool_idx = 0
    assistant_msg_count = 0

    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            assistant_msg_count += 1
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                args = fn.get("arguments", "")
                t_id = tc.get("id")

                response_content = None
                if t_id and t_id in tool_responses:
                    response_content = tool_responses[t_id]
                elif tool_idx < len(tool_order):
                    response_content = tool_order[tool_idx]
                    tool_idx += 1

                failed = False
                if response_content is not None:
                    is_rejected = bool(_TOOL_REJECT_PATTERNS.search(response_content))
                    is_edit_failed = False
                    if name not in _NOT_A_FAILURE:
                        is_edit_failed = bool(_EDIT_FAIL_PATTERNS.search(response_content))
                    failed = is_rejected or is_edit_failed

                # What this call is working on. The key list used to be
                # written out here and did not contain `filePath`, which is
                # the ONLY key this client sends -- so the resource was always
                # None, `same_resource` fell back to comparing whole argument
                # strings, and a retry with a corrected `oldString` looked like
                # a different call. That is why the escalating-temperature
                # branch never fired once in five weeks: it could not.
                file_path = get_resource(parse_args({"arguments": args}))

                sequence.append({
                    "name": name,
                    "args": args,
                    "failed": failed,
                    "file": file_path,
                    "message_idx": assistant_msg_count
                })
    return sequence


# ---------------------------------------------------------------------------
# Main adaptive function
# ---------------------------------------------------------------------------

def count_retry_depth(messages: list[dict], window: int = 20) -> int:
    """Count how many consecutive turns the same failed tool/resource was retried at the tail.

    Returns 0 if no retry loop, otherwise the depth (2, 3, 4, 5, ...).
    """
    sequence = _analyze_tool_sequence(messages)
    if len(sequence) > window:
        sequence = sequence[-window:]

    if len(sequence) < 2:
        return 0

    last_call = sequence[-1]
    
    # If the last call did not fail, we are not in a retry loop
    if not last_call["failed"]:
        return 0

    # Count distinct assistant turns (message_idx) in the retry chain
    failed_turns = {last_call["message_idx"]}

    for i in range(len(sequence) - 2, -1, -1):
        prev_call = sequence[i]

        is_same_tool = prev_call["name"] == last_call["name"]
        was_failed = prev_call["failed"]

        same_resource = False
        if prev_call["file"] is not None and last_call["file"] is not None:
            same_resource = (prev_call["file"] == last_call["file"])
        else:
            norm_prev = re.sub(r"\s+", "", prev_call["args"])
            norm_last = re.sub(r"\s+", "", last_call["args"])
            same_resource = (norm_prev == norm_last)

        if is_same_tool and was_failed and same_resource:
            failed_turns.add(prev_call["message_idx"])
        else:
            break

    depth = len(failed_turns)
    return depth if depth >= 2 else 0



def effective_loop_depth(retry_depth: int, guard_cuts: int) -> int:
    """One depth from the two ways this gateway can tell the model is stuck.

    `retry_depth` counts a tool failing on the same resource turn after turn.
    `guard_cuts` counts turns a runaway guard ENDED -- which produce no tool
    call at all, so `count_retry_depth` returns 0 on every one of them and the
    escalation could never see the loops it was written for.

    **A guard cut never maps to depth 2, and that is the point of this
    function rather than a `max()` at the call site.** Depth 2 is the typo
    hypothesis: the model nearly had the argument right, so remove the
    sampling noise and let it try again. A reasoning loop has no argument to
    get right, and greedy decoding is exactly what makes a stuck model repeat
    itself VERBATIM -- which is what `reasoning_loop` measures, the share of
    distinct lines in the tail. Answering a cut with temperature 0.0 would arm
    the guard that just fired.

    So cuts enter at the exploration end: two in a row is depth 3 (temp 0.6),
    three is 4 (0.7), four or more is 5 (0.8, the cap).

    **One cut is not a streak.** The turn that prompted all this was a
    seven-image oscilloscope analysis cut once, legitimately mid-work; raising
    the temperature of careful long reasoning because it was long is the wrong
    lesson to draw from a single cut.

    The larger of the two wins, so a session that is both retrying a failing
    edit and being cut gets the stronger answer -- which is the exploration
    one, for the reason above.
    """
    from_cuts = guard_cuts + 1 if guard_cuts >= 2 else 0
    return max(retry_depth, from_cuts)


# Patterns indicating loop_breaker intervened and stopped a repeating tool call
_LOOP_GUARD_PATTERNS = re.compile(
    r"\[SYSTEM GUARD\].*(?:call was stopped|already run|already ran|loop_guard)",
    re.IGNORECASE | re.DOTALL,
)
_LOOP_GUARD_COUNT_RE = re.compile(
    r"already run\s+(\d+)\s+times|already ran\s+(\d+)\s+times",
    re.IGNORECASE,
)


def count_identical_loop_depth(messages: list[dict], window: int = 6) -> int:
    """Detect if recent messages contain a loop_breaker guard intervention.

    When loop_breaker suppresses a repeated tool call, the model was told in
    a [SYSTEM GUARD] message that the call was stopped. Adaptive sampling
    must raise temperature (depth >= 3 -> temp 0.6+) to force exploration,
    rather than lowering temperature via TOOL_HEAVY to 0.15 which forces
    deterministic repetition of the exact same blocked tool call.
    """
    if not messages:
        return 0

    recent_msgs = messages[-window:] if len(messages) > window else messages
    for m in reversed(recent_msgs):
        if m.get("role") == "assistant":
            content = m.get("content") or ""
            if isinstance(content, str) and _LOOP_GUARD_PATTERNS.search(content):
                match = _LOOP_GUARD_COUNT_RE.search(content)
                if match:
                    c_str = match.group(1) or match.group(2)
                    if c_str:
                        return max(3, int(c_str))
                return 3

    return 0


def should_escalate(messages: list[dict], threshold: int) -> int:
    """Retry depth if it has reached the escalation threshold, else 0.

    By the time this fires, `compute_adaptive_params` has already tried both
    of its answers to a loop: greedy at depth 2 to eliminate typos, then rising
    temperature at 3+ to force exploration. Neither helps when the model simply
    cannot do the step, and each attempt costs the single upstream slot. A
    threshold above the exploratory range means escalation is the answer only
    after sampling has demonstrably failed.
    """
    if threshold <= 0:
        return 0
    depth = max(count_retry_depth(messages), count_identical_loop_depth(messages))
    return depth if depth >= threshold else 0


def compute_adaptive_params(
    messages: list[dict],
    base_params: dict[str, Any],
    n_ctx: int | None = None,
    guard_cuts: int = 0,
    temperature_floor: float = 0.0,
) -> dict[str, Any]:
    """Analyze conversation and adjust sampling parameters.

    Returns a new dict with adjusted parameters. Never mutates base_params.

    Adjustment strategy:
      - Recent edit failures → lower temperature (more precise text reproduction)
      - Retry loop detected (depth == 2) → GREEDY mode (temp = 0.0) to eliminate typos
      - Retry loop detected (depth >= 3) → INCREASE temperature to force exploration
      - Turns cut by a runaway guard (>= 2 in a row) → the same exploration
        branch, and never the greedy one (see `effective_loop_depth`)
      - High context pressure → increase min_p (filter weak tokens)
      - Many tool calls → slightly lower temperature (precision mode)

    `temperature_floor` (> 0) bounds every LOWERING rule above, the greedy step
    included, for a model whose loops get worse as temperature falls -- a
    thinking model. Measured on Qwen3.6, 2026-10-16/17: TOOL_HEAVY and
    TOOL_FAILURES took a 0.6 model to 0.55 and 0.5 on 90 turns, and with the
    earlier 0.3 base to 0.15-0.25, while the reasoning-loop guard was cutting
    the same model's repeated sentences. The raising branch is untouched.
    """
    params = dict(base_params)
    base_temp = params.get("temperature", 0.3)
    adjustments: list[str] = []

    # 1. Retry loop detection (highest priority)
    retry_depth = effective_loop_depth(count_retry_depth(messages), guard_cuts)
    identical_depth = count_identical_loop_depth(messages)
    if identical_depth >= 3:
        retry_depth = max(retry_depth, identical_depth)
    floor = max(0.0, float(temperature_floor or 0.0))
    if retry_depth == 2 and floor > 0.0:
        # The typo hypothesis does not get its greedy retry on a model with a
        # floor: greedy decoding is what makes a thinking model repeat itself
        # verbatim, which is what the reasoning-loop guard then cuts.
        adjustments.append(
            f"RETRY_LOOP(depth={retry_depth}): greedy retry skipped "
            f"(temperature floor {floor})"
        )
    elif retry_depth == 2:
        params["temperature"] = 0.0
        adjustments.append(
            f"RETRY_LOOP(depth={retry_depth}): temp {base_temp}→0.0 (greedy retry to fix typos)"
        )
    elif retry_depth >= 3:
        # Escalating temperature: deeper loop → higher temperature
        # depth 3 → max(base, 0.6), depth 4 → +0.1, depth 5+ → capped at max(base, 0.8)
        max_cap = max(base_temp, 0.8)
        new_temp = round(min(max_cap, max(base_temp, 0.6) + (retry_depth - 3) * 0.1), 2)
        params["temperature"] = new_temp
        # Also increase top_p slightly for more diversity
        params["top_p"] = min(1.0, params.get("top_p", 0.95) + 0.04)
        adjustments.append(
            f"RETRY_LOOP(depth={retry_depth}"
            + (f", {guard_cuts} guard cut(s)" if guard_cuts >= 2 else "")
            + f"): temp {base_temp}→{new_temp} (↑ exploration to break loop)"
        )

    else:
        # 2. Recent tool failures (not a loop yet — reduce temp for precision)
        failures = _count_recent_tool_failures(messages, window=10)
        if failures >= 1:
            new_temp = (max(0.1, base_temp - 0.15) if failures >= 3
                        else max(0.15, base_temp - 0.1))
            new_temp = max(floor, new_temp)
            if new_temp < base_temp:
                params["temperature"] = new_temp
                adjustments.append(f"TOOL_FAILURES({failures}): temp {base_temp}→{new_temp}")

        # 3. Tool-heavy conversation
        tool_calls = _count_recent_tool_calls(messages, window=10)
        if tool_calls >= 5:
            current_temp = params.get("temperature", base_temp)
            new_temp = max(floor, 0.15, current_temp - 0.05)
            if new_temp < current_temp:
                params["temperature"] = new_temp
                adjustments.append(f"TOOL_HEAVY({tool_calls}): temp→{new_temp}")

        # 4. Context pressure
        pressure = _estimate_context_pressure(messages)
        if pressure > 0.7:
            current_min_p = params.get("min_p", 0.05)
            limit = n_ctx if n_ctx else 120_000
            scale = 1.0
            if limit > 100_000:
                scale = 0.2  # very gentle increase for large context models
            new_min_p = min(0.15, current_min_p + pressure * 0.05 * scale)
            if new_min_p > current_min_p:
                params["min_p"] = round(new_min_p, 3)
                adjustments.append(f"CTX_PRESSURE({pressure:.2f}): min_p→{new_min_p:.3f}")

    # Log adjustments
    if adjustments:
        _log.info("ADAPTIVE SAMPLING: %s", " | ".join(adjustments))
    else:
        _log.debug("ADAPTIVE SAMPLING: no adjustments (base temp=%.2f)", base_temp)

    return params

