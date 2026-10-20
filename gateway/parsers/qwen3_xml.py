"""Qwen 3.6 XML Finite State Machine parser.

Strict, high-performance XML FSM parser optimized for Qwen 3.6 (e.g. 35B A3B).
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from enum import Enum, auto
from typing import Any, Iterator

from gateway.parsers.base import (
    EventType,
    FsmEvent,
    extract_json_candidate,
    is_potential_tool_call,
    parse_json_object,
    process_speculative_json,
    resolve_json_tool_call,
    resolve_tool_call_dict,
)

_log = logging.getLogger("gateway.parsers.qwen3")

# Vocab control tokens the model occasionally leaks as text
_SPECIAL_TOKEN_RE = re.compile(r"<\|[A-Za-z0-9_]{1,30}\|>")


class _State(Enum):
    DETECT_THINKING = auto()  # Initial: buffer to detect <think>
    TEXT = auto()              # Normal content
    THINKING = auto()         # Inside <think>…</think>
    TOOL_CALL = auto()        # Inside <tool_call>, waiting for <function=
    FUNCTION = auto()         # Inside <function=NAME>, waiting for params
    PARAM_VALUE = auto()      # Inside <parameter=NAME>…</parameter>


class Qwen3XmlFSM:
    """Stateful parser for Qwen 3.6 XML tool-call format."""

    def __init__(
        self,
        tools: list[dict] | None = None,
        in_triple_backticks: bool = False,
        in_single_backtick: bool = False,
        start_in_thinking: bool = False,
        start_in_tool_call: bool = False,
    ):
        if start_in_tool_call:
            self._state = _State.TOOL_CALL
        elif start_in_thinking:
            self._state = _State.THINKING
        else:
            self._state = _State.DETECT_THINKING
        self._tools = tools or []
        self._buf = ""
        self._detect_buf = ""
        self._json_buf = ""
        self._pending_whitespace = ""
        self._function_body_buf = ""
        self._has_emitted_content = False

        self._tool_index = 0
        self._tool_id = ""
        self._func_name = ""
        self._params: dict[str, str] = {}

        self._param_name = ""
        self._param_type = "string"
        self._param_hold = ""
        self._is_first_param_chunk = False
        self._has_emitted_first_param = False
        self._speculative_buf = ""
        self._func_committed = False
        self._just_finished_tool_call = False
        self._in_triple_backticks = in_triple_backticks
        self._in_single_backtick = in_single_backtick
        self._consecutive_backticks = 0
        self._has_trailing_newline = False
        self._line_has_non_ws = False

    def _flush_backticks(self) -> None:
        if self._consecutive_backticks > 0:
            count = self._consecutive_backticks
            self._consecutive_backticks = 0
            if count >= 3 and not self._line_has_non_ws:
                self._in_triple_backticks = not self._in_triple_backticks
                self._in_single_backtick = False
            else:
                if not self._in_triple_backticks:
                    self._in_single_backtick = not self._in_single_backtick
            self._line_has_non_ws = True

    @property
    def tool_call_count(self) -> int:
        return self._tool_index

    @property
    def in_thinking(self) -> bool:
        """True while inside an unterminated <think> block (see BaseToolParser)."""
        return self._state == _State.THINKING

    def feed(self, chunk: str) -> Iterator[FsmEvent]:
        self._buf += chunk
        yield from self._process()

    def finish(self, finish_reason: str = "stop") -> Iterator[FsmEvent]:  # noqa: ARG002  # BaseToolParser declares it; nothing here decides from it, and renaming would break the override for a keyword caller
        self._flush_backticks()
        if self._state == _State.TOOL_CALL and self._speculative_buf:
            spec = self._speculative_buf.strip()
            self._speculative_buf = ""
            if spec.startswith("<tool_call>"):
                inner = spec[len("<tool_call>"):].strip()
            else:
                inner = spec
            tool_call = resolve_json_tool_call(inner, self._tools)
            if tool_call:
                yield from self._emit_resolved_tool_call(*tool_call)
                self._just_finished_tool_call = True
            else:
                yield from self._emit_text(spec)

        if self._state == _State.DETECT_THINKING and self._detect_buf:
            self._state = _State.TEXT
            self._buf = self._detect_buf + self._buf
            self._detect_buf = ""
            yield from self._process()

        if self._buf or self._state in (_State.PARAM_VALUE, _State.FUNCTION):
            if self._state == _State.PARAM_VALUE:
                if self._param_hold:
                    self._params[self._param_name] = self._params.get(self._param_name, "") + self._param_hold
                    yield FsmEvent(EventType.PARAM_CHUNK, {
                        "index": self._tool_index, "text": self._param_hold,
                        "param_type": self._param_type,
                    })
                    self._param_hold = ""
                yield FsmEvent(EventType.PARAM_END, {
                    "index": self._tool_index, "name": self._param_name,
                    "param_type": self._param_type,
                })
                yield FsmEvent(EventType.TOOL_CALL_END, {
                    "index": self._tool_index, "id": self._tool_id,
                    "name": self._func_name, "params": dict(self._params),
                    "auto_closed": True,
                })
                self._tool_index += 1
            elif self._state == _State.FUNCTION:
                if not self._params and self._function_body_buf.strip():
                    yield from self._try_heal_function_body_json()
                self._function_body_buf = ""
                yield FsmEvent(EventType.TOOL_CALL_END, {
                    "index": self._tool_index, "id": self._tool_id,
                    "name": self._func_name, "params": dict(self._params),
                    "auto_closed": True,
                })
                self._tool_index += 1
            elif self._buf:
                text = self._buf
                self._buf = ""
                if self._state == _State.THINKING:
                    yield FsmEvent(EventType.REASONING, text)
                else:
                    yield from self._emit_text(text)

        if self._json_buf or self._pending_whitespace:
            if self._json_buf:
                tool_calls, content_to_emit, remaining = process_speculative_json(self._json_buf, self._tools)
                if not tool_calls and remaining and is_potential_tool_call(remaining, self._tools):
                    from gateway.tool_schema import repair_json
                    try:
                        repaired = repair_json(remaining)
                        if isinstance(repaired, dict):
                            tc = resolve_tool_call_dict(repaired, self._tools)
                            if tc:
                                tool_calls = [tc]
                                remaining = ""
                    except Exception:
                        pass
                if self._pending_whitespace:
                    if content_to_emit:
                        content_to_emit = self._pending_whitespace + content_to_emit
                    self._pending_whitespace = ""
                if content_to_emit:
                    yield FsmEvent(EventType.CONTENT, content_to_emit)
                    self._has_emitted_content = True
                if tool_calls:
                    self._in_triple_backticks = False
                    self._consecutive_backticks = 0
                    self._in_single_backtick = False
                    for name, args in tool_calls:
                        yield from self._emit_resolved_tool_call(name, args)
                    self._just_finished_tool_call = True
                if remaining and remaining.strip():
                    yield FsmEvent(EventType.CONTENT, remaining)
                    self._has_emitted_content = True
                self._json_buf = ""
            if self._pending_whitespace:
                if self._pending_whitespace.strip():
                    yield FsmEvent(EventType.CONTENT, self._pending_whitespace)
                    self._has_emitted_content = True
                else:
                    _log.warning("Empty response guard: discarding %d bytes whitespace post-think",
                                 len(self._pending_whitespace))
                self._pending_whitespace = ""

    def _is_valid_tag_for_state(self, tag: str) -> bool:
        if self._state == _State.THINKING:
            return tag in ("</think>", "<tool_call>")
        if self._state == _State.PARAM_VALUE:
            # A NEW parameter opener ends the current value, even though the
            # model never closed it. Measured 2026-09-10: a `read` emitted with
            # no `</parameter>` anywhere put `<parameter=offset>` and
            # `<parameter=limit>` INSIDE filePath -- a 167-character path, and
            # the other two arguments simply absent. finish() then auto-closed
            # the call, so the client got a well-formed `read` naming a file
            # that cannot exist. Qwen25CoderFSM has accepted this since it was
            # written; the two parsers share this loop line for line and
            # differed only here.
            #
            # `</function>` and `</tool_call>` are deliberately NOT accepted in
            # this state, unlike in Qwen25CoderFSM: no session has shown the
            # model omitting `</parameter>` before them, their handlers here
            # have not been checked against arriving mid-value, and finish()
            # already auto-closes that shape correctly.
            return tag == "</parameter>" or (
                tag.startswith("<parameter=") and tag.endswith(">")
            )
        if self._state == _State.TEXT:
            self._flush_backticks()
            if self._in_triple_backticks or self._in_single_backtick:
                return False
            return tag in ("<think>", "</think>", "<tool_call>", "</function>", "</tool_call>", "</parameter>")
        if self._state == _State.TOOL_CALL:
            return tag == "</tool_call>" or (tag.startswith("<function=") and tag.endswith(">"))
        if self._state == _State.FUNCTION:
            return tag == "</function>" or (tag.startswith("<parameter=") and tag.endswith(">"))
        return False

    def _process(self) -> Iterator[FsmEvent]:
        while self._buf:
            if self._state == _State.DETECT_THINKING:
                yield from self._handle_detect_thinking()
                if self._state == _State.DETECT_THINKING:
                    break
                continue

            idx = self._buf.find("<")
            if idx == -1:
                text = self._buf
                self._buf = ""
                yield from self._handle_plain_text(text)
                break

            if idx > 0:
                text = self._buf[:idx]
                self._buf = self._buf[idx:]
                yield from self._handle_plain_text(text)

            if (
                len(self._buf) >= 2
                and self._buf[1] == "|"
                and self._state in (_State.TEXT, _State.THINKING)
            ):
                end_sp = self._buf.find("|>")
                if end_sp == -1:
                    if len(self._buf) < 40:
                        break
                else:
                    token = self._buf[:end_sp + 2]
                    if _SPECIAL_TOKEN_RE.fullmatch(token):
                        self._buf = self._buf[end_sp + 2:]
                        _log.warning("FSM: dropped leaked special token %r", token)
                        continue

            if len(self._buf) >= 2:
                next_char = self._buf[1]
                if not (next_char.isalpha() or next_char == '/' or next_char == '!'):
                    char = self._buf[0]
                    self._buf = self._buf[1:]
                    yield from self._handle_plain_text(char)
                    continue

            end = self._buf.find(">")
            if end == -1:
                if len(self._buf) >= 2:
                    next_char = self._buf[1]
                    if not (next_char.isalpha() or next_char == '/' or next_char == '!'):
                        char = self._buf[0]
                        self._buf = self._buf[1:]
                        yield from self._handle_plain_text(char)
                        continue
                if len(self._buf) > 150:
                    char = self._buf[0]
                    self._buf = self._buf[1:]
                    yield from self._handle_plain_text(char)
                    continue
                break

            # NO TAG THIS FSM KNOWS SPANS A LINE, and that is what makes a
            # newline before the `>` decisive: the `<` was punctuation.
            #
            # Measured 2026-09-10 on a live session. The model wrote
            # `... dla zakresu 10<V<=30 (czyli 15 V).` and then, two lines
            # later, a perfectly well-formed <tool_call>. The nearest `>` to
            # that first `<` is the one closing <tool_call>, so the candidate
            # tag became the prose PLUS the opener, failed validation, and was
            # emitted verbatim -- the user read raw XML and the turn carried no
            # tool call at all.
            #
            # The give-up above cannot catch it: it fires when NO `>` is found,
            # and here one is found. It just belongs to somebody else.
            #
            # Emitting a single character and rescanning is deliberate. The
            # real opener is still in the buffer, and dropping the whole span
            # would take it with it.
            nl = self._buf.find(chr(10))
            if 0 <= nl < end:
                char = self._buf[0]
                self._buf = self._buf[1:]
                yield from self._handle_plain_text(char)
                continue

            tag = self._buf[:end + 1]
            self._buf = self._buf[end + 1:]
            if self._is_valid_tag_for_state(tag):
                yield from self._handle_tag(tag)
            else:
                yield from self._handle_plain_text(tag)

    def _handle_detect_thinking(self) -> Iterator[FsmEvent]:
        self._detect_buf += self._buf
        self._buf = ""
        stripped = self._detect_buf.lstrip()

        if stripped.startswith("<think>"):
            self._state = _State.THINKING
            idx = stripped.find("<think>")
            self._buf = stripped[idx + len("<think>"):]
            self._detect_buf = ""
        elif len(stripped) >= len("<think>") or (stripped and not "<think>".startswith(stripped)):
            self._state = _State.TEXT
            self._buf = self._detect_buf
            self._detect_buf = ""
        return
        yield

    def _handle_plain_text(self, text: str) -> Iterator[FsmEvent]:
        if self._state == _State.TEXT:
            if text.strip():
                self._just_finished_tool_call = False
            for ch in text:
                if ch == "`":
                    self._consecutive_backticks += 1
                else:
                    self._flush_backticks()
                    if ch == "\n":
                        if self._has_trailing_newline and not self._in_triple_backticks:
                            # Blank line (two newlines separated only by optional whitespace)
                            # resets inline code backtick state (CommonMark §6.1).
                            self._in_single_backtick = False
                        self._has_trailing_newline = True
                        self._line_has_non_ws = False
                    elif ch in (" ", "\t", "\r"):
                        pass
                    else:
                        self._line_has_non_ws = True
                        self._has_trailing_newline = False
            yield from self._emit_text(text)
        elif self._state == _State.THINKING:
            yield FsmEvent(EventType.REASONING, text)
        elif self._state == _State.PARAM_VALUE:
            yield from self._handle_param_value_text(text)
        elif self._state == _State.TOOL_CALL:
            if self._func_committed:
                pass
            else:
                self._speculative_buf += text
        elif self._state == _State.FUNCTION:
            self._function_body_buf += text

    def _emit_text(self, text: str) -> Iterator[FsmEvent]:
        self._json_buf += text
        tool_calls, content_to_emit, remaining = process_speculative_json(self._json_buf, self._tools)

        if content_to_emit:
            if not self._has_emitted_content:
                if not content_to_emit.strip():
                    self._pending_whitespace += content_to_emit
                    content_to_emit = ""
                elif self._pending_whitespace:
                    content_to_emit = self._pending_whitespace + content_to_emit
                    self._pending_whitespace = ""

            if content_to_emit:
                yield FsmEvent(EventType.CONTENT, content_to_emit)
                self._has_emitted_content = True

        if tool_calls:
            self._pending_whitespace = ""
            self._in_triple_backticks = False
            self._consecutive_backticks = 0
            self._in_single_backtick = False
            for name, args in tool_calls:
                yield from self._emit_resolved_tool_call(name, args)
            self._just_finished_tool_call = True

        self._json_buf = remaining
        if self._json_buf:
            limit = 262144 if is_potential_tool_call(self._json_buf, self._tools) else 4096
            if len(self._json_buf) > limit:
                to_flush = self._pending_whitespace + self._json_buf
                self._pending_whitespace = ""
                yield FsmEvent(EventType.CONTENT, to_flush)
                self._has_emitted_content = True
                self._json_buf = ""

    def _handle_param_value_text(self, text: str) -> Iterator[FsmEvent]:
        if self._is_first_param_chunk:
            self._is_first_param_chunk = False
            if text.startswith("\r\n"):
                text = text[2:]
            elif text.startswith("\n"):
                text = text[1:]

        full_text = self._param_hold + text
        if full_text.endswith("\r\n"):
            self._param_hold = "\r\n"
            to_emit = full_text[:-2]
        elif full_text.endswith("\n"):
            self._param_hold = "\n"
            to_emit = full_text[:-1]
        else:
            self._param_hold = ""
            to_emit = full_text

        if to_emit:
            self._params[self._param_name] = self._params.get(self._param_name, "") + to_emit
            yield FsmEvent(EventType.PARAM_CHUNK, {
                "index": self._tool_index, "text": to_emit,
                "param_type": self._param_type,
            })

    def _handle_tag(self, tag: str) -> Iterator[FsmEvent]:
        if tag == "<think>":
            self._state = _State.THINKING
        elif tag == "</think>":
            if self._state == _State.THINKING:
                self._state = _State.TEXT
            elif self._state == _State.TEXT:
                # Orphan closing </think> (model emitted reasoning without opening <think>):
                # Drop it silently so it never leaks into content.
                _log.debug("FSM: dropped orphan </think> in TEXT state")
        elif tag == "<tool_call>":
            self._state = _State.TOOL_CALL
            self._speculative_buf = "<tool_call>"
            self._func_committed = False
        elif tag == "</tool_call>":
            if self._state == _State.TEXT:
                pass
            elif self._func_committed:
                self._state = _State.TEXT
                self._func_committed = False
                self._just_finished_tool_call = True
            else:
                spec = self._speculative_buf.strip()
                self._speculative_buf = ""
                self._state = _State.TEXT
                if spec:
                    if spec.startswith("<tool_call>"):
                        inner = spec[len("<tool_call>"):].strip()
                    else:
                        inner = spec
                    tool_call = resolve_json_tool_call(inner, self._tools)
                    if tool_call:
                        yield from self._emit_resolved_tool_call(*tool_call)
                        self._just_finished_tool_call = True
                    else:
                        yield from self._emit_text(spec + tag)
        elif tag.startswith("<function=") and tag.endswith(">"):
            self._function_body_buf = ""
            name = tag[len("<function="):-1].strip()
            is_valid_func = True
            if self._tools:
                registered_names = {
                    t.get("function", {}).get("name")
                    for t in self._tools
                    if t.get("function", {}).get("name")
                }
                from gateway.tool_schema import TOOL_NAME_ALIASES
                resolved: str | None = name
                if name == "skill":
                    is_valid_func = True
                else:
                    if name not in registered_names:
                        resolved = TOOL_NAME_ALIASES.get(name)
                    if resolved is None or resolved not in registered_names:
                        is_valid_func = False

            if not is_valid_func:
                spec = self._speculative_buf
                self._speculative_buf = ""
                self._state = _State.TEXT
                yield from self._emit_text(spec + tag)
            else:
                self._state = _State.FUNCTION
                self._speculative_buf = ""
                self._func_committed = True
                self._func_name = name
                self._tool_id = f"call_{uuid.uuid4().hex[:10]}"
                self._has_emitted_first_param = False
                self._params = {}
                yield FsmEvent(EventType.TOOL_CALL_START, {
                    "index": self._tool_index,
                    "id": self._tool_id,
                    "name": name,
                })
        elif tag == "</function>":
            if self._state == _State.TEXT:
                pass
            else:
                self._state = _State.TOOL_CALL
                if not self._params and self._function_body_buf.strip():
                    yield from self._try_heal_function_body_json()
                self._function_body_buf = ""
                yield FsmEvent(EventType.TOOL_CALL_END, {
                    "index": self._tool_index,
                    "id": self._tool_id,
                    "name": self._func_name,
                    "params": dict(self._params),
                    "auto_closed": False,
                })
                self._tool_index += 1
                self._tool_id = ""
                self._func_name = ""
                self._params = {}
        elif tag.startswith("<parameter=") and tag.endswith(">"):
            if self._state == _State.PARAM_VALUE:
                # Reached from an unterminated value: close it before opening
                # the next, or the emitter is handed two overlapping ones.
                self._param_hold = ""
                yield FsmEvent(EventType.PARAM_END, {
                    "index": self._tool_index,
                    "name": self._param_name,
                    "param_type": self._param_type,
                })
            param_name = tag[len("<parameter="):-1].strip()
            self._state = _State.PARAM_VALUE
            self._param_name = param_name
            self._param_type = self._resolve_param_type(param_name)
            self._is_first_param_chunk = True
            self._param_hold = ""
            yield FsmEvent(EventType.PARAM_START, {
                "index": self._tool_index,
                "name": param_name,
                "type": self._param_type,
                "is_first": not self._has_emitted_first_param,
            })
            self._has_emitted_first_param = True
        elif tag == "</parameter>":
            if self._state == _State.TEXT:
                pass
            else:
                self._state = _State.FUNCTION
                self._param_hold = ""
                yield FsmEvent(EventType.PARAM_END, {
                    "index": self._tool_index,
                    "name": self._param_name,
                    "param_type": self._param_type,
                })
        else:
            yield from self._handle_plain_text(tag)

    def _emit_resolved_tool_call(self, name: str, args: dict[str, Any]) -> Iterator[FsmEvent]:
        tool_id = f"call_{uuid.uuid4().hex[:10]}"
        yield FsmEvent(EventType.TOOL_CALL_START, {
            "index": self._tool_index, "id": tool_id, "name": name
        })
        if isinstance(args, dict):
            is_first = True
            for k, v in args.items():
                p_type = self._resolve_param_type_for(name, k)
                yield FsmEvent(EventType.PARAM_START, {
                    "index": self._tool_index, "name": k, "type": p_type, "is_first": is_first
                })
                is_first = False
                val_str = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
                self._params[k] = val_str
                yield FsmEvent(EventType.PARAM_CHUNK, {
                    "index": self._tool_index, "text": val_str, "param_type": p_type
                })
                yield FsmEvent(EventType.PARAM_END, {
                    "index": self._tool_index, "name": k, "param_type": p_type
                })
        yield FsmEvent(EventType.TOOL_CALL_END, {
            "index": self._tool_index, "id": tool_id, "name": name,
            "params": dict(self._params),
            "auto_closed": False
        })
        self._tool_index += 1
        self._params = {}

    def _try_heal_function_body_json(self) -> Iterator[FsmEvent]:
        cleaned = extract_json_candidate(self._function_body_buf)
        if cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                data = parse_json_object(cleaned)
                if isinstance(data, dict):
                    args = data.get("arguments") or data.get("parameters") or data.get("input") or data
                    if isinstance(args, dict):
                        is_first = True
                        for k, v in args.items():
                            p_type = self._resolve_param_type_for(self._func_name, k)
                            yield FsmEvent(EventType.PARAM_START, {
                                "index": self._tool_index, "name": k, "type": p_type, "is_first": is_first
                            })
                            is_first = False
                            val_str = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
                            self._params[k] = val_str
                            yield FsmEvent(EventType.PARAM_CHUNK, {
                                "index": self._tool_index, "text": val_str, "param_type": p_type
                            })
                            yield FsmEvent(EventType.PARAM_END, {
                                "index": self._tool_index, "name": k, "param_type": p_type
                            })
            except Exception:
                pass

    def _resolve_param_type_for(self, func_name: str, param_name: str) -> str:
        for t in self._tools:
            fn = t.get("function") or {}
            if fn.get("name") == func_name:
                params = fn.get("parameters") or {}
                props = params.get("properties") or {}
                param_def = props.get(param_name) or {}
                t_val = param_def.get("type", "string")
                if isinstance(t_val, list):
                    non_null = [x for x in t_val if x != "null"]
                    t_val = non_null[0] if non_null else "string"
                if t_val == "string" and "items" in param_def:
                    return "array"
                return str(t_val)
        return "string"

    def _resolve_param_type(self, param_name: str) -> str:
        return self._resolve_param_type_for(self._func_name, param_name)

