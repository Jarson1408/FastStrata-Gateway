"""Gemma 4's chat format and native tool protocol -- the prompt half.

The gateway's prompt renderers are one module per family, so a new model is
a new file rather than another branch inside `normalizer.py`:

    ChatML (Qwen) ...... gateway/normalizer.py:format_chatml  (untouched; the
                         path every Qwen request takes, byte for byte as
                         before Gemma existed)
    Gemma 4 ............ this module
    output parsing ..... gateway/parsers/<family>.py
    tool value syntax .. gateway/parsers/gemma_values.py

`normalizer.render_prompt` is the one dispatch, on `config.prompt_format`.
This module borrows the neutralizers and tool-result helpers from
`normalizer` rather than copying them: a second copy of a neutralizer is a
second place for an escape to be missed.
"""
from __future__ import annotations

import json
import re
from typing import Any

from gateway.normalizer import (
    _media_markers,
    _neutralize_control_tokens,
    render_tool_calls,
    render_tool_results,
)


# Gemma 4's frame closes with `<word|>` rather than opening with `<|`:
# `<turn|>`, `<channel|>`, `<tool_call|>`, `<tool_response|>`. The ChatML
# neutralizer breaks openers only, so a body quoting `<turn|>` would reach the
# engine as token 106 -- end of turn -- and the rest of the file would become
# a turn nobody wrote. Applied by the Gemma renderer ONLY: the ChatML path
# stays byte-identical, which is what keeps every warm Qwen prefix warm.
_GEMMA_CLOSER_RE = re.compile(r"<(\w+)\|>")


def _break_gemma_closers(text: str) -> str:
    return _GEMMA_CLOSER_RE.sub(r"<\1 |>", text)


def _neutralize_gemma(text: object) -> str:
    """Render-safe form of a message body inside a Gemma 4 prompt."""
    return _break_gemma_closers(_neutralize_control_tokens(text))


#: What a Gemma 4 model turn opens with when it is NOT asked to reason: an
#: empty thought channel, exactly as the model's own chat_template.jinja
#: writes it when `enable_thinking` is unset. The model then answers directly.
GEMMA4_EMPTY_THOUGHT = "<|channel>thought\n<channel|>"


def _neutralize_gemma_value(value: Any) -> Any:
    """`_neutralize_gemma` applied to every string inside a call's arguments."""
    if isinstance(value, str):
        return _neutralize_gemma(value)
    if isinstance(value, dict):
        return {k: _neutralize_gemma_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_neutralize_gemma_value(v) for v in value]
    return value


def _gemma_native_calls(msg: dict) -> str:
    """An assistant message's calls, as Gemma 4 writes them."""
    from gateway.parsers.gemma_values import format_call_args

    out = ""
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw = fn.get("arguments")
        if isinstance(raw, str):
            try:
                args: Any = json.loads(raw) if raw.strip() else {}
            except (json.JSONDecodeError, ValueError):
                args = raw
        else:
            args = raw or {}
        if isinstance(args, dict):
            args = _neutralize_gemma_value(args)
        out += f"<|tool_call>call:{fn.get('name', '')}{format_call_args(args)}<tool_call|>"
    return out


def _gemma_native_response(tm: dict, names: dict[str, str]) -> str:
    """One tool result, as the template writes a tool response."""
    from gateway.parsers.gemma_values import format_response

    name = tm.get("name") or names.get(str(tm.get("tool_call_id") or ""), "")
    content = tm.get("content")
    if not isinstance(content, str):
        content = "" if content is None else json.dumps(content, ensure_ascii=False)
    body = format_response(name, _neutralize_gemma(content))
    return f"<|tool_response>{body}<tool_response|>{_media_markers(tm)}"


def _gemma_segment_lead(msg: dict, thinking: bool) -> str:
    """What preceded a segment the model GENERATED inside a tool loop.

    With thinking off, every generation in the loop began right after an
    empty thought channel; with it on, after the model's own reasoning (kept
    by preserve_thinking for the current loop only). Rendering exactly that
    is what keeps each round's prompt a strict extension of the last one.
    """
    if not thinking:
        return GEMMA4_EMPTY_THOUGHT
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if reasoning:
        return f"<|channel>thought\n{_neutralize_gemma(reasoning)}<channel|>"
    return ""


def format_gemma4(messages: list[dict], thinking: bool = False,
                  native_tools: bool = False) -> str:
    """Build a Gemma 4 prompt from normalized messages.

    The same walk as `format_chatml`, in the frame the model was trained on
    (`<bos><|turn>role ... <turn|>`, the assistant called `model`), checked
    against both the checkpoint's `chat_template.jinja` and exllamav3's own
    `Gemma4Config.default_chat_prompt`. Until 2026-10-15 every `/completion`
    prompt was ChatML, which Gemma would have read as plain text with
    `<|im_start|>` in it.

    **Two tool dialects.** With `native_tools` (Gemma's default, see
    `config.tool_dialect_for`) the tools are declared in the model's own
    protocol and a tool loop is ONE model turn, exactly as Gemma generates it:

        <|turn>model\\n{lead}text<|tool_call>call:read{...}<tool_call|>
        <|tool_response>response:read{value:<|"|>...<|"|>}<tool_response|>
        {lead}next call or answer ... <turn|>

    The model stops on `<|tool_response>` (id 50, an EOS in its generation
    config), the result is written into the SAME turn and generation resumes
    there. **This departs from the HF template on purpose**: the template
    puts every call of a message before every response and drops the
    thought-channel lead of each generation, which no live loop ever looked
    like -- so each round would re-prefill the whole turn, tool results and
    all. Rendered as generated, round N+1's prompt extends round N's byte for
    byte. Plain answers with no tools stay exactly as the template writes
    them (check_gemma4_prompt.py, 8/8).

    Without `native_tools` the injected Qwen XML dialect is used, rendered by
    the same helpers as ChatML, tool results going back as a USER turn -- the
    shape gemma-4-26B was measured answering in on 2026-09-13.

    * **Reasoning of finished turns is not replayed**; the template strips it.
    * **`<|think|>` opens the first system turn** when `thinking` is on; with
      no system message one is opened for it. With thinking off each
      generation starts after an EMPTY thought channel, which is how the
      template asks for a direct answer.
    """
    from gateway.parsers.gemma_values import format_declaration

    prompt = "<bos>"
    think_pending = thinking
    if thinking and not (messages and messages[0].get("role") == "system"):
        prompt += "<|turn>system\n<|think|><turn|>\n"
        think_pending = False

    # tool_call_id -> name, for results that arrive without a name.
    names: dict[str, str] = {}
    for m in messages:
        for tc in m.get("tool_calls") or []:
            if tc.get("id"):
                names[str(tc["id"])] = (tc.get("function") or {}).get("name", "")

    open_after_response = False
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")

        if role == "system":
            body = _neutralize_gemma(msg.get("content", "")).strip()
            lead = "<|think|>" if think_pending else ""
            think_pending = False
            decls = ""
            if native_tools:
                for tool in msg.get("_gateway_tools") or []:
                    decls += "<|tool>" + format_declaration(tool) + "<tool|>"
            prompt += f"<|turn>system\n{lead}{body}{_media_markers(msg)}{decls}<turn|>\n"

        elif role == "user":
            body = _neutralize_gemma(msg.get("content", "")).strip()
            prompt += f"<|turn>user\n{body}{_media_markers(msg)}<turn|>\n"

        elif role == "assistant" and native_tools:
            # One model turn: this message and every assistant/tool message
            # that continues its tool loop.
            prompt += "<|turn>model\n"
            in_loop = False
            open_after_response = False
            while i < len(messages) and messages[i].get("role") == "assistant":
                seg = messages[i]
                has_calls = bool(seg.get("tool_calls"))
                if has_calls or in_loop:
                    prompt += _gemma_segment_lead(seg, thinking)
                prompt += _neutralize_gemma(seg.get("content") or "").strip()
                prompt += _media_markers(seg)
                i += 1
                if not has_calls:
                    prompt += "<turn|>\n"
                    open_after_response = False
                    break
                prompt += _gemma_native_calls(seg)
                in_loop = True
                while i < len(messages) and messages[i].get("role") == "tool":
                    prompt += _gemma_native_response(messages[i], names)
                    i += 1
                open_after_response = True
            else:
                # The loop ended on a tool response with nothing after it in
                # this turn: either the live prompt (generation resumes here)
                # or a user message follows, which closes the turn first.
                if i < len(messages) and open_after_response:
                    prompt += "<turn|>\n"
                    open_after_response = False
            continue

        elif role == "assistant":
            prompt += "<|turn>model\n"
            # The template writes the content and closes the turn directly
            # (`cześć<turn|>`), so no newline is added unless something follows.
            content = _neutralize_gemma(msg.get("content") or "").strip()
            calls = _break_gemma_closers(render_tool_calls(msg))
            prompt += content
            if content and calls:
                prompt += "\n"
            prompt += _media_markers(msg) + calls
            prompt += "<turn|>\n"

        elif role == "tool":
            # Qwen-dialect results (or orphans) go back as a user turn.
            tool_msgs = [msg]
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                tool_msgs.append(messages[j])
                j += 1
            prompt += "<|turn>user\n"
            for tm in tool_msgs:
                prompt += _break_gemma_closers(render_tool_results([tm]))
                prompt += _media_markers(tm)
            prompt += "<turn|>\n"
            i = j
            continue

        i += 1

    if open_after_response:
        # Resume inside the open model turn, as the template does after a
        # tool response: no new `<|turn>model`.
        if not thinking:
            prompt += GEMMA4_EMPTY_THOUGHT
        return prompt
    prompt += "<|turn>model\n"
    if not thinking:
        prompt += GEMMA4_EMPTY_THOUGHT
    return prompt
