"""Muse Glimmer 30B output parser: messages addressed to recipients.

Muse Glimmer does not interleave tags inside one answer the way Qwen does. A
turn is a sequence of MESSAGES, each with a recipient, written by its chat
template as

    <|start|>assistant to=self<|message|>{reasoning}<|eom|>
    <|start|>assistant to=user<|message|>{answer}<|eot|>
    <|start|>assistant to=read<|message|><atem:function_calls>
    <atem:invoke name="read">
    <atem:parameter name="filePath">C:/a.txt</atem:parameter>
    </atem:invoke>
    </atem:function_calls><|eot|>

The gateway's prompt ends with `<|start|>assistant`, so a generation begins
INSIDE the first header (` to=self<|message|>...`). `<|eot|>` (200008) and
`<|end_of_text|>` (200001) are the EOS ids and never arrive; `<|eom|>` is not
an EOS, which is why one generation carries the reasoning AND the answer or
call. All markers are special tokens, decoded as text because server.py's
profile for this checkpoint turns `decode_special_tokens` on.

**Why this is not a delegation to Qwen3XmlFSM**, unlike `gemma.py`: there is
no tag scan to share. The routing is by message header, and a tool call is a
whole `<atem:function_calls>` block whose own docs say it is "not expected to
be valid XML and is parsed with regular expressions". What IS shared with the
Qwen parser is the contract downstream: the same events, values as raw
strings, `param_type` resolved from the tool's schema the way
`Qwen3XmlFSM._resolve_param_type` does it, so tool_schema coercion treats a
Muse call exactly like a Qwen one.
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import Iterator

from gateway.parsers.base import EventType, FsmEvent

_log = logging.getLogger("gateway.parsers.muse")

START = "<|start|>"
MESSAGE = "<|message|>"
EOM = "<|eom|>"
EOT = "<|eot|>"
END_OF_TEXT = "<|end_of_text|>"
_MARKERS = (START, MESSAGE, EOM, EOT, END_OF_TEXT)
_ENDS = (EOM, EOT, END_OF_TEXT, START)

_RECIPIENT_RE = re.compile(r"to=(\S+)")
_INVOKE_RE = re.compile(r'<atem:invoke\s+name="([^"]+)"\s*>(.*?)</atem:invoke>', re.DOTALL)
_PARAM_RE = re.compile(r'<atem:parameter\s+name="([^"]+)"\s*>(.*?)</atem:parameter>', re.DOTALL)

_HEADER, _BODY, _BETWEEN = "header", "body", "between"


class MuseToolParser:
    """Routes Muse Glimmer messages to reasoning, content and tool calls."""

    def __init__(
        self,
        tools: list[dict] | None = None,
        in_triple_backticks: bool = False,
        in_single_backtick: bool = False,
        start_in_thinking: bool = False,
        start_in_tool_call: bool = False,
    ):
        # Accepted for the factory's shared signature. Backtick state and the
        # Qwen prefills have no meaning here: the prompt always ends with an
        # open `<|start|>assistant` header, which is where every generation
        # begins.
        del in_triple_backticks, in_single_backtick, start_in_thinking, start_in_tool_call
        self._tools = tools or []
        self._state = _HEADER
        self._header = ""
        self._recipient = ""
        self._body = ""          # a tool message's body, held whole
        self._hold = ""          # a tail that may still grow into a marker
        self._next_index = 0

    # -- BaseToolParser ------------------------------------------------------

    @property
    def tool_call_count(self) -> int:
        return self._next_index

    @property
    def in_thinking(self) -> bool:
        """Inside a message to `self` -- what every reasoning guard asks."""
        return self._state == _BODY and self._recipient == "self"

    def feed(self, chunk: str) -> Iterator[FsmEvent]:
        text = self._hold + (chunk or "")
        text, self._hold = self._split_holdback(text)
        yield from self._process(text)

    def finish(self, finish_reason: str = "stop") -> Iterator[FsmEvent]:
        del finish_reason
        tail, self._hold = self._hold, ""
        if tail:
            yield from self._process(tail)
        if self._state == _BODY:
            yield from self._end_message()
        elif self._state == _HEADER and self._header.strip():
            # A header with no body is not a message. Shown rather than lost.
            yield FsmEvent(EventType.CONTENT, self._header)
        self._state = _BETWEEN

    # -- Routing -------------------------------------------------------------

    def _process(self, text: str) -> Iterator[FsmEvent]:
        while text:
            if self._state == _HEADER:
                idx = text.find(MESSAGE)
                if idx == -1:
                    self._header += text
                    return
                self._header += text[:idx]
                text = text[idx + len(MESSAGE):]
                m = _RECIPIENT_RE.search(self._header)
                # No `to=` is a message to the user, the template's default.
                self._recipient = m.group(1) if m else "user"
                self._header = ""
                self._state = _BODY
                continue

            if self._state == _BETWEEN:
                idx = text.find(START)
                if idx == -1:
                    # Text outside any message. Not expected; surfaced as
                    # content so it is visible rather than swallowed.
                    stray = text
                    for marker in _ENDS:
                        stray = stray.replace(marker, "")
                    if stray.strip():
                        yield FsmEvent(EventType.CONTENT, stray)
                    return
                if text[:idx].strip():
                    yield FsmEvent(EventType.CONTENT, text[:idx])
                text = text[idx + len(START):]
                self._state = _HEADER
                continue

            # _BODY: up to the first marker that ends a message.
            cut = min((i for i in (text.find(m) for m in _ENDS) if i != -1),
                      default=-1)
            piece = text if cut == -1 else text[:cut]
            yield from self._body_text(piece)
            if cut == -1:
                return
            yield from self._end_message()
            if text.startswith(START, cut):
                text = text[cut + len(START):]
                self._state = _HEADER
            else:
                marker = next(m for m in _ENDS if text.startswith(m, cut))
                text = text[cut + len(marker):]
                self._state = _BETWEEN

    def _body_text(self, piece: str) -> Iterator[FsmEvent]:
        if not piece:
            return
        if self._recipient == "self":
            yield FsmEvent(EventType.REASONING, piece)
        elif self._recipient == "user":
            yield FsmEvent(EventType.CONTENT, piece)
        else:
            self._body += piece

    def _end_message(self) -> Iterator[FsmEvent]:
        recipient, body = self._recipient, self._body
        self._recipient, self._body = "", ""
        self._state = _BETWEEN
        if recipient in ("self", "user", ""):
            return
        invokes = _INVOKE_RE.findall(body)
        if not invokes:
            _log.warning("MUSE: message to '%s' carried no readable "
                         "<atem:invoke>; shown as text rather than guessed.", recipient)
            if body.strip():
                yield FsmEvent(EventType.CONTENT, body)
            return
        for name, inner in invokes:
            yield from self._emit_call(name.strip(), _PARAM_RE.findall(inner))

    # -- Tool calls ----------------------------------------------------------

    def _resolve_param_type(self, func: str, param: str) -> str:
        """Same answer as `Qwen3XmlFSM._resolve_param_type`."""
        for t in self._tools:
            fn = t.get("function") or {}
            if fn.get("name") == func:
                props = (fn.get("parameters") or {}).get("properties") or {}
                param_def = props.get(param) or {}
                t_val = param_def.get("type", "string")
                if isinstance(t_val, list):
                    non_null = [x for x in t_val if x != "null"]
                    t_val = non_null[0] if non_null else "string"
                if t_val == "string" and "items" in param_def:
                    return "array"
                return str(t_val)
        return "string"

    def _emit_call(self, name: str, params: list[tuple[str, str]]) -> Iterator[FsmEvent]:
        """One complete call. Emitted whole, so a downstream gate can still
        drop it -- it never could once half the arguments had streamed."""
        index = self._next_index
        self._next_index += 1
        call_id = f"call_{uuid.uuid4().hex[:24]}"
        yield FsmEvent(EventType.TOOL_CALL_START, {"index": index, "id": call_id, "name": name})
        values: dict[str, str] = {}
        for n, (key, value) in enumerate(params):
            ptype = self._resolve_param_type(name, key)
            values[key] = value
            yield FsmEvent(EventType.PARAM_START, {
                "index": index, "name": key, "type": ptype, "is_first": n == 0})
            yield FsmEvent(EventType.PARAM_CHUNK, {
                "index": index, "text": value, "param_type": ptype})
            yield FsmEvent(EventType.PARAM_END, {
                "index": index, "name": key, "param_type": ptype})
        yield FsmEvent(EventType.TOOL_CALL_END, {
            "index": index, "id": call_id, "name": name,
            "params": values, "auto_closed": False})

    # -- Holding back a split marker ----------------------------------------

    @staticmethod
    def _split_holdback(text: str) -> tuple[str, str]:
        longest = max(len(m) for m in _MARKERS)
        for keep in range(min(longest - 1, len(text)), 0, -1):
            tail = text[-keep:]
            if any(m.startswith(tail) for m in _MARKERS):
                return text[:-keep], tail
        return text, ""
