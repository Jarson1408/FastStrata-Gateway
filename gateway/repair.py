"""
Repairing what the model got structurally wrong.

Local models routinely emit tool-call arguments with an unbalanced brace or an
unescaped quote, and stop mid-structure when they hit the token limit. The
OpenCode client answers a malformed argument string with "JSON Parse error:
Expected '}'" and the chat breaks in front of the user — the 2026-07-07
Capacitor session hit that five times in a row on one bash command.

Nothing here is LODESTAR-specific; it is the structural safety net that runs
under every response, streaming or not.
"""
from __future__ import annotations

import json
import logging

_log = logging.getLogger("gateway.repair")

def validate_tool_args_json(args_str: str, func_name: str, idx: int) -> str | None:
    """v2.4.3: validate the emitted tool-call args as JSON.

    The model sometimes produces tool calls where the args string is
    missing a closing brace or has unescaped quotes. The OpenCode
    client then errors with "JSON Parse error: Expected '}'" and the
    user sees a broken chat.

    This helper delegates to `gateway.tool_schema.repair_json` which
    already implements the full repair pipeline (strict json.loads →
    json_repair.repair_json → legacy regex fallback). If the args are
    already valid, returns them unchanged. If repair succeeds, returns
    the repaired JSON. If repair fails, returns None so the caller
    can drop the call silently (no error visible to user).

    The 2026-07-07 Capacitor session showed this happening repeatedly
    when the model tried to call bash with an embed.py search command
    that contained nested backslash-escaped paths; the OpenCode client
    parser was rejecting the args with "Expected '}'" 5+ times in a
    row, breaking the chat.

    Note: this is the only place outside tool_schema.py that needs
    JSON repair, because tool_schema.py only repairs the args of
    skill calls (which it rewrites to read). Non-skill tool calls
    (bash, edit, etc.) go through the FSM and need this extra hook.
    """
    from gateway.tool_schema import repair_json
    try:
        repaired = repair_json(args_str)
    except Exception as e:
        _log.warning(
            "LODESTAR ENFORCER: tool '%s' idx=%d repair_json failed: %s; "
            "dropping. args=%r",
            func_name, idx, e, args_str[:200],
        )
        return None
    if not isinstance(repaired, dict):
        # repair_json may return {"items": [...]} for non-dict results;
        # for tool calls we need a dict, so drop.
        _log.warning(
            "LODESTAR ENFORCER: tool '%s' idx=%d repair_json returned non-dict %s; "
            "dropping. args=%r",
            func_name, idx, type(repaired).__name__, args_str[:200],
        )
        return None
    repaired_str = json.dumps(repaired, ensure_ascii=False)
    if repaired_str != args_str:
        _log.info(
            "LODESTAR ENFORCER: tool '%s' idx=%d had malformed JSON args; "
            "repair_json fixed it. Before=%r After=%r",
            func_name, idx, args_str[:200], repaired_str[:200],
        )
    return repaired_str


def repair_truncated_structure(text: str) -> str:
    """Attempt to syntactically close a truncated JSON or XML structure."""
    text = text.strip()
    # 1. MD code blocks
    if text.startswith("```"):
        if not text.endswith("```") and len(text) > 3:
            return text + "\n```"
        return text

    # 2. JSON repair
    if text.startswith("{") or text.startswith("["):
        braces = 0
        brackets = 0
        in_string = False
        escape = False
        
        for char in text:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    braces += 1
                elif char == '}':
                    braces -= 1
                elif char == '[':
                    brackets += 1
                elif char == ']':
                    brackets -= 1
                    
        repaired = text
        if in_string:
            repaired += '"'
        if brackets > 0:
            repaired += "]" * brackets
        if braces > 0:
            repaired += "}" * braces
            
        try:
            json.loads(repaired)
            return repaired
        except Exception:
            return repaired
            
    # 3. XML repair
    if text.startswith("<"):
        if "<tool_call>" in text and "</tool_call>" not in text:
            return text + "\n</tool_call>"
            
    return text


def check_text_backtick_state(text: str) -> tuple[bool, bool]:
    """Scan string content to check if it ends with open backticks.

    Recognizes:
    - Code fences (```) opening/closing code blocks.
    - Inline code spans (`...`) toggling inline state.
    - Markdown table cells with inline backticks (| ... ` ```dataview ` ... |) do
      not open code fences.
    - Blank lines (\\n\\n) reset inline code state (CommonMark §6.1).
    - Closed tags (</tool_call>, </function>) indicate closed block context.
    """
    stripped = text.strip()
    if stripped.endswith("</tool_call>") or stripped.endswith("</function>"):
        return False, False

    in_triple = False
    in_single = False
    i = 0
    n = len(text)

    last_was_newline = True

    while i < n:
        if text[i:i+3] == "```":
            line_end = text.find("\n", i)
            rest_of_line = text[i+3:line_end] if line_end != -1 else text[i+3:]
            current_line_start = text.rfind("\n", 0, i)
            line_prefix = text[current_line_start+1:i] if current_line_start != -1 else text[:i]

            # If inside a markdown table cell (surrounded by | on current line), it is inline
            is_inline_table = ("|" in line_prefix and "|" in rest_of_line)

            if is_inline_table:
                if not in_triple:
                    in_single = not in_single
                i += 3
            else:
                in_triple = not in_triple
                if in_triple:
                    in_single = False
                i += 3
        elif text[i] == "`":
            if not in_triple:
                in_single = not in_single
            i += 1
        elif text[i] == "\n":
            if last_was_newline and not in_triple:
                # Blank line resets inline code state (CommonMark §6.1)
                in_single = False
            last_was_newline = True
            i += 1
        elif text[i] in (" ", "\t", "\r"):
            i += 1
        else:
            last_was_newline = False
            i += 1

    return in_triple, in_single


def get_initial_backtick_state(messages: list[dict]) -> tuple[bool, bool]:
    """Scan history to check if the generation is continuing an open backtick state.

    Backtick states (code fences and inline code spans) NEVER cross ChatML message
    turn boundaries. If the last message is not an assistant message (e.g. user or
    tool), this is a brand new assistant turn and starts with closed backticks (False, False).
    Only when the client sends an assistant message as the last message (prefill / continuation)
    does the new generation continue that assistant message's backtick state.
    """
    if not messages:
        return False, False

    last_msg = messages[-1]
    if last_msg.get("role") != "assistant":
        return False, False

    last_assistant_content = last_msg.get("content") or ""
    if not last_assistant_content:
        return False, False

    return check_text_backtick_state(last_assistant_content)



def extract_conclusion_from_reasoning(reasoning: str) -> str:
    """Extract a substantive summary/conclusion when the model exhausted its generation in <think>.

    When reasoning models (Qwen 3.6, DeepSeek R1) output several thousand tokens
    inside <think> and hit EOS immediately after </think> without emitting content
    or tool calls, this extracts the concluding thoughts so the client does not
    receive an empty assistant turn.
    """
    text = reasoning.strip()
    if not text:
        return "Task analysis completed."

    # Look for paragraphs
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        return text[-800:]

    # Check the last 4 paragraphs for conclusion keywords
    markers = ("in summary", "to summarize", "podsumowując", "wniosek", "conclusion", "therefore", "thus", "to fix", "overall", "we should", "powinniśmy")
    for i in range(len(paras) - 1, max(-1, len(paras) - 4), -1):
        p_lower = paras[i].lower()
        if any(m in p_lower for m in markers):
            start_idx = max(0, i - 1) if (i == len(paras) - 1 and len(paras) >= 2) else i
            res = "\n\n".join(paras[start_idx:])
            return res if len(res) <= 1200 else res[-1200:]

    conclusion = "\n\n".join(paras[-2:]) if len(paras) >= 2 else paras[-1]
    if len(conclusion) > 1200:
        conclusion = conclusion[-1200:]
    return conclusion


ASSISTANT_TURN_MARKER = "<|im_start|>assistant"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
TOOL_CALL_OPEN, TOOL_CALL_CLOSE = "<tool_call>", "</tool_call>"


def prepare_for_think_injection(base_prompt: str) -> tuple[str, bool]:
    """Make a prompt safe to append a `<think>` block to.

    Returns the (possibly trimmed) prompt and whether its `<think>` block is
    still OPEN, so the caller knows to close one rather than open one.

    Both continuation builders got this wrong in the same way (A-2026-03).
    They asked `"<think>" in tail`, which is also true of a block that has
    already CLOSED — and in LODESTAR states 1-2 the prefill ends the prompt
    with a closed think block followed by an OPEN `<tool_call>`. So they
    appended a second, unmatched `</think>` inside that open tool call, while
    the retry parser was built with `start_in_tool_call=False`: the model went
    on emitting `<function=...>`, the FSM read it as ordinary text, and raw XML
    reached the user with no tool call emitted at all.

    A dangling `<tool_call>` opener is DROPPED, and that is the point rather
    than a repair. Both callers run because the turn is being abandoned — a
    suppressed repeat, or a reasoning loop — so the call the prefill was
    forcing is precisely what must not happen now. Dropping it also makes the
    prompt agree with the parser the caller builds.
    """
    prompt = base_prompt.rstrip()
    tail = prompt.split(ASSISTANT_TURN_MARKER)[-1]

    open_call = tail.rfind(TOOL_CALL_OPEN)
    if open_call != -1 and TOOL_CALL_CLOSE not in tail[open_call:]:
        prompt = prompt[:len(prompt) - (len(tail) - open_call)].rstrip()
        tail = prompt.split(ASSISTANT_TURN_MARKER)[-1]

    return prompt, tail.count(THINK_OPEN) > tail.count(THINK_CLOSE)


def build_continuation_prompt(
    base_prompt: str,
    reasoning_text: str,
    max_tokens: int | None = None,
) -> str:
    """Build a prompt that closes the assistant's <think> block and prompts action.

    If reasoning_text fits within max_tokens, it is preserved verbatim so that the
    upstream server reuses the warm KV cache (0 ms prefill).
    If reasoning_text is too long to fit safely, its conclusion is extracted.
    """
    clean_reasoning = reasoning_text.strip()
    if not clean_reasoning:
        clean_reasoning = "I will stop deliberating and take the next concrete step now."
    else:
        # Check if keeping the whole reasoning would overflow max_tokens.
        # Heuristic: ~3.3 chars per token.
        if max_tokens and (len(base_prompt) + len(clean_reasoning)) // 3 > max_tokens:
            clean_reasoning = extract_conclusion_from_reasoning(clean_reasoning)
        clean_reasoning += "\nI will now take the next concrete step and provide the answer or call the required tool."

    p, think_open = prepare_for_think_injection(base_prompt)
    if think_open:
        return f"{p}\n{clean_reasoning}\n{THINK_CLOSE}\n\n"
    if not p.endswith("\n"):
        p += "\n"
    return f"{p}{THINK_OPEN}\n{clean_reasoning}\n{THINK_CLOSE}\n\n"


def build_content_continuation_prompt(
    messages: list[dict],
    content_text: str,
    reasoning_text: str = "",
    max_tokens: int | None = None,
    nudge: str | None = None,
) -> str:
    """Build a prompt to seamlessly continue generating content after truncation.

    Compacts message history to the system message(s) and the last user request to
    free up KV-cache headroom, retains the reasoning conclusion (if any) closed
    inside <think>, and appends the content emitted so far verbatim.
    Crucially, the assistant turn remains unclosed (no <|im_end|>) so the model
    resumes generation directly from the last emitted character.
    """
    from gateway.normalizer import format_chatml

    system_msgs = [m for m in messages if m.get("role") == "system"]
    last_user = None
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m
            break

    compact_msgs = []
    compact_msgs.extend(system_msgs)
    if last_user:
        compact_msgs.append(last_user)
    elif messages and messages[-1].get("role") != "system":
        compact_msgs.append(messages[-1])

    prefix = format_chatml(compact_msgs)

    reasoning_block = ""
    if reasoning_text and reasoning_text.strip():
        conclusion = extract_conclusion_from_reasoning(reasoning_text)
        if conclusion:
            reasoning_block = f"<think>\n{conclusion}\n</think>\n"

    content = content_text
    if max_tokens:
        # Heuristic: ~3 chars per token
        chars_used = len(prefix) + len(reasoning_block)
        remaining_chars = max(3000, (max_tokens * 3) - chars_used)
        if len(content) > remaining_chars:
            content = content[-remaining_chars:]

    if nudge:
        # The rescue case: the block above is CLOSED, so the model physically
        # cannot go back to deliberating, and the nudge is the last thing it
        # reads. Text asking a model to stop thinking is what it has already
        # been ignoring; taking the block away is the part that works, and
        # this is the same trick reasoning_retry_prefill uses.
        return f"{prefix}{reasoning_block}{content}\n{nudge}\n"

    return f"{prefix}{reasoning_block}{content}"


def build_loop_recovery_prompt(
    base_prompt: str,
    loop_message: str,
    tool_name: str,
) -> str:
    """Build a prompt that steers the model out of a loop and into an alternative tool or action.

    Closes the assistant's <think> block with the guard's directive so the model
    cannot keep deliberating or repeat the failed call, and must immediately select an
    alternative action or valid file path.
    """
    clean_guidance = (
        f"{loop_message}\n\n"
        f"I will NOT repeat the `{tool_name}` call with these arguments. "
        "I will choose a different tool, use an exact file path, or take an alternative concrete step now."
    )
    p, think_open = prepare_for_think_injection(base_prompt)
    if think_open:
        return f"{p}\n{clean_guidance}\n{THINK_CLOSE}\n\n"
    if not p.endswith("\n"):
        p += "\n"
    return f"{p}{THINK_OPEN}\n{clean_guidance}\n{THINK_CLOSE}\n\n"


