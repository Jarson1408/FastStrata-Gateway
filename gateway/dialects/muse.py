"""Muse Glimmer 30B's chat format and native tool protocol -- the prompt half.

The format, from the checkpoint's own chat_template.jinja:

    <|begin_of_text|>
    <|start|>system<|message|>{system}\\n\\nReasoning strength: high.
        \\n\\n{tool definitions, JSON}\\n\\n# Valid recipients: "self", "read.*", "user".<|eot|>
    <|start|>user<|message|>{text}<|eot|>
    <|start|>assistant to=self<|message|>{reasoning}<|eom|>
    <|start|>assistant to=read<|message|><atem:function_calls>...</atem:function_calls><|eot|>
    <|start|>tool read<|message|><tool_output name="read">\\n{result}\\n</tool_output><|eot|>
    <|start|>assistant           <- generation prompt; the MODEL picks the recipient

`<|eom|>` ends a message and not the turn, and is NOT an EOS; `<|eot|>`
(200008) and `<|end_of_text|>` (200001) are. `test_scripts/check_muse_prompt.py`
compares this renderer byte for byte with the template.

What deliberately differs from the template, each for the reason Gemma's
renderer gives (see gateway/dialects/gemma4.py):

* **Only the FIRST system message carries the footer** (reasoning strength,
  tool definitions, recipients). The template writes it into every system
  message; the gateway appends volatile system messages (the LODESTAR frame,
  loop warnings), and repeating fifteen tool schemas in each would be both
  noise to the model and the whole tool list re-prefilled every turn.
* **A preamble written before a call is kept**, as the `to=user ... <|eom|>`
  message the model generated it as. The template drops an assistant
  message's content when it has tool calls; replaying it is what keeps a
  tool loop's prompts strict extensions of one another.

Reasoning is replayed where the history still carries it (preserve_thinking
keeps it for the current tool loop only), which the template also does.
"""
from __future__ import annotations

import json
import re
from typing import Any

from gateway import prompts
from gateway.normalizer import _media_markers, _neutralize_control_tokens

#: The template's default when no system message is given.
DEFAULT_SYSTEM = "You are a helpful AI assistant."
KNOWLEDGE_CUTOFF = "2026-01-04"
REASONING_LEVELS = ("low", "medium", "high", "xhigh")

_TOOL_OUTPUT_CLOSE_RE = re.compile(r"</(tool_output)>")
_ATEM_CLOSE_RE = re.compile(r"</(atem:\w+)>")


def _tojson(value: Any) -> str:
    """transformers' `tojson` filter: no key sorting, no ASCII escaping."""
    return json.dumps(value, ensure_ascii=False)


def _neutralize(text: object) -> str:
    return _neutralize_control_tokens(text)


def _fn(tool: dict) -> dict:
    return tool.get("function", tool) if isinstance(tool, dict) else {}


def _namespaces(tools: list[dict]) -> list[str]:
    seen: list[str] = []
    for tool in tools:
        ns = str(_fn(tool).get("name", "")).split(".")[0]
        if ns not in seen:
            seen.append(ns)
    return seen


def render_tool_defs(tools: list[dict]) -> str:
    """The template's `render_tool_defs` macro.

    The two blocks of prose are the template's own words, kept verbatim in
    `gateway/prompts/muse_tool_intro.md` and `muse_tool_example.md` (the
    project keeps every model-visible text out of `.py`). Only the JSON in
    between is built here.
    """
    out = prompts.get("muse_tool_intro") + "\n"
    for ns in _namespaces(tools):
        out += '{"name": ' + _tojson(ns) + ', "description": ' + _tojson("") + "}\n"
    out += "// Function schemas"
    for tool in tools:
        fn = _fn(tool)
        out += ('\n{"name": ' + _tojson(fn.get("name")) + ', "description": '
                + _tojson(fn.get("description")) + ', "parameters": '
                + _tojson(fn.get("parameters")) + "}")
    return out + "\n\n" + prompts.get("muse_tool_example")


def render_system_meta(tools: list[dict]) -> str:
    recipients = ['"self"'] + [f'"{ns}.*"' for ns in (_namespaces(tools) if tools else [])]
    recipients.append('"user"')
    return "# Valid recipients: " + ", ".join(recipients) + "."


def _footer(reasoning: str, tools: list[dict]) -> str:
    out = "\n\n" + f"Reasoning strength: {reasoning or 'high'}."
    if tools:
        out += "\n\n" + render_tool_defs(tools)
    return out + "\n\n" + render_system_meta(tools)


def render_atem(name: str, args: Any) -> str:
    """The template's `render_atem` macro (arguments in their own order)."""
    out = f'<atem:function_calls>\n<atem:invoke name="{name}">\n'
    if isinstance(args, dict):
        for key, value in args.items():
            out += f'<atem:parameter name="{key}">'
            if isinstance(value, bool):
                out += "true" if value else "false"
            elif value is None:
                out += "null"
            elif isinstance(value, (dict, list, tuple)):
                out += _tojson(value)
            else:
                out += str(value)
            out += "</atem:parameter>\n"
    return out + "</atem:invoke>\n</atem:function_calls>"


def _call_args(tc: dict) -> Any:
    raw = (tc.get("function") or {}).get("arguments")
    if isinstance(raw, str):
        try:
            return json.loads(raw) if raw.strip() else {}
        except (json.JSONDecodeError, ValueError):
            return {}
    return raw or {}


def _safe_value(value: Any) -> Any:
    """Neutralize strings inside replayed call arguments."""
    if isinstance(value, str):
        return _ATEM_CLOSE_RE.sub(r"</ \1>", _neutralize(value))
    if isinstance(value, dict):
        return {k: _safe_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_value(v) for v in value]
    return value


def format_muse(messages: list[dict], reasoning: str = "high",
                native_tools: bool = True) -> str:
    """Build a Muse Glimmer prompt from normalized messages."""
    prompt = "<|begin_of_text|>"
    first_system = next((m for m in messages if m.get("role") == "system"), None)
    tools = list((first_system or {}).get("_gateway_tools") or []) if native_tools else []

    if first_system is None:
        prompt += ("<|start|>system<|message|>" + DEFAULT_SYSTEM
                   + f"\nKnowledge cutoff: {KNOWLEDGE_CUTOFF}."
                   + _footer(reasoning, tools) + "<|eot|>")

    names: dict[str, str] = {}
    for m in messages:
        for tc in m.get("tool_calls") or []:
            if tc.get("id"):
                names[str(tc["id"])] = (tc.get("function") or {}).get("name", "")

    for idx, msg in enumerate(messages):
        role = msg.get("role", "")
        nxt = messages[idx + 1].get("role") if idx + 1 < len(messages) else None

        if role == "system":
            body = _neutralize(msg.get("content", "")) + _media_markers(msg)
            footer = _footer(reasoning, tools) if msg is first_system else ""
            prompt += "<|start|>system<|message|>" + body + footer + "<|eot|>"

        elif role == "user":
            prompt += ("<|start|>user<|message|>" + _neutralize(msg.get("content", ""))
                       + _media_markers(msg) + "<|eot|>")

        elif role == "tool":
            tname = msg.get("name") or names.get(str(msg.get("tool_call_id") or ""), "")
            content = msg.get("content")
            if not isinstance(content, str):
                content = "" if content is None else _tojson(content)
            content = _TOOL_OUTPUT_CLOSE_RE.sub(r"</ \1>", _neutralize(content))
            prompt += (f"<|start|>tool {tname}<|message|><tool_output name=\"{tname}\">\n"
                       + content + _media_markers(msg) + "\n</tool_output><|eot|>")

        elif role == "assistant":
            end_token = "<|eom|>" if nxt == "assistant" else "<|eot|>"
            reasoning_text = msg.get("reasoning_content") or msg.get("reasoning") or ""
            if reasoning_text:
                prompt += ("<|start|>assistant to=self<|message|>"
                           + _neutralize(reasoning_text) + "<|eom|>")
            calls = msg.get("tool_calls") or []
            content = _neutralize(msg.get("content") or "")
            if calls:
                if content.strip():
                    prompt += "<|start|>assistant to=user<|message|>" + content + "<|eom|>"
                for n, tc in enumerate(calls):
                    name = (tc.get("function") or {}).get("name", "")
                    prompt += (f"<|start|>assistant to={name}<|message|>"
                               + render_atem(name, _safe_value(_call_args(tc)))
                               + (end_token if n == len(calls) - 1 else "<|eom|>"))
            else:
                prompt += ("<|start|>assistant to=user<|message|>" + content
                           + _media_markers(msg) + "<|eot|>")

    return prompt + "<|start|>assistant"
