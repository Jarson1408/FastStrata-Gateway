"""Gemma 4 tool-call parser.

READ THIS BEFORE TRUSTING ANY PART OF IT, because half of this file is a
measurement and the other half is not.

**What is certain.** `inject_tools_to_prompt` describes ONE dialect to every
model this gateway serves -- the Qwen3-Coder XML one -- and deletes the `tools`
array so llama-server's own parser never sees it. So a Gemma model here is
ASKED for `<tool_call><function=name><parameter=k>`, exactly like every other
model, and that path is not speculative at all: it is the same scan engine,
reused rather than re-implemented.

**What was a guess, and what it cost.** This file used to say Gemma's thinking
tag was `<thought> ... </thought>`, taken from this project's own design note
rather than from a live model, with a paragraph admitting it was a guess "until
somebody downloads the weights and checks".

Somebody did, on 2026-09-13, and the guess was wrong. The model emits:

    <|channel>thought\n ... reasoning ... <channel|> ... answer

Read back verbatim from gemma-4-26B-A4B-it-qat, and confirmed against the
model's own chat_template.jinja, which renders a stored `reasoning_content` as
`'<|channel>thought\n' + text + '<channel|>'`. The two markers are NOT mirror
images, and `<channel|>` is generic: the same template frames tool calls and
tool responses with `<|tool_call>`/`<tool_call|>` and `<|tool_response>`/
`<tool_response|>`. So the closer is rewritten only while a thought channel is
open -- see `_rewrite`, where that statefulness is the whole point.

**What this cost while it was wrong:** nothing crashed. The reasoning simply
arrived as `content`, raw markup and all, instead of as `reasoning_content`,
and the first end-to-end run PASSED because the answer it was checking for was
in there somewhere. A check that looks for a substring cannot see a turn whose
structure is wrong.

`<thought>` is kept as tolerance and labelled as never-observed.
`call:name{"a": 1}` is still a guess from the same note -- it is cheap and
cannot fire by accident, since `call:` must be followed by a DECLARED tool name
and a complete object, but no model has produced it here.
`test_scripts/check_gemma_dialect.py` is what turns any of this into a
measurement; it exists now, which it did not when this paragraph first
promised it.

WHY THIS DELEGATES INSTEAD OF BEING A THIRD FSM

AGENTS.md records what two implementations of one scan loop cost: the
prose-`<` defect of 2026-09-10 was in BOTH `qwen3_xml.py` and
`qwen25_coder.py`, and was found in neither for months, "because nobody reads
them side by side". A third copy would be a third place for the next one to
hide. So the tag scanning, the `<` handling, the missing-closer repair and the
auto-close on `finish()` all come from `Qwen3XmlFSM`, and this class adds only
what is genuinely different: a thinking tag with another name, and one extra
call shape.

The cost of delegating is that tool-call indices arrive from two sources, so
this class owns the index space and renumbers. That is what `_index_map` is
for, and it is the one piece of machinery here worth a careful read.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Iterator

from gateway.parsers.base import EventType, FsmEvent
from gateway.parsers.gemma_values import GemmaSyntaxError, parse_args, split_object
from gateway.parsers.qwen3_xml import Qwen3XmlFSM

_log = logging.getLogger("gateway.parsers.gemma")

#: Gemma's thinking markers, rewritten to the ones the base FSM knows. A
#: rewrite rather than a second state: `in_thinking` is what every reasoning
#: guard in this gateway asks about, and a parser with its own private notion
#: of "thinking" would be invisible to all of them.
#:
#: **MEASURED 2026-09-13 against gemma-4-26B-A4B-it-qat on llama-server.** The
#: model opens with `<|channel>thought` and closes with `<channel|>` -- note
#: that the two are not mirror images of each other, and that neither is the
#: `<thought>` this file guessed at from a design note. Read back verbatim:
#:
#:     '<|channel>thought\n*   Question: ... <channel|>17 times 3 is 51.'
#:
#: The same pair appears in the model's own chat_template.jinja, which renders
#: a stored `reasoning_content` as `'<|channel>thought\n' + text + '<channel|>'`
#: -- so this is the protocol rather than one sample's habit.
_CHANNEL_OPEN = "<|channel>thought"
_CHANNEL_CLOSE = "<channel|>"

#: The design note's guess, kept as tolerance and labelled as what it is: no
#: Gemma has ever emitted it here. It costs one comparison and cannot fire by
#: accident on this model, which does not use the shape at all.
_THINK_OPEN = "<thought>"
_THINK_CLOSE = "</thought>"

#: Rewrites that need no state: an opener always means the same thing.
_OPEN_REWRITES = ((_CHANNEL_OPEN, "<think>"), (_THINK_OPEN, "<think>"))
_CLOSE_REWRITES = ((_CHANNEL_CLOSE, "</think>"), (_THINK_CLOSE, "</think>"))

#: The longest prefix of any marker, so one split across two chunks is held
#: rather than emitted half-recognised.
_MAX_TAG_LEN = max(len(t) for t, _ in _OPEN_REWRITES + _CLOSE_REWRITES)

#: The call frame, tracked but never rewritten: the base FSM reads these
#: itself. They are held back across chunks all the same, because knowing
#: whether we are inside a call is what decides the parameter rewrite below.
_FRAME_TAGS = ("<tool_call>", "</tool_call>")

#: `call:name{...}` -- the name is matched permissively and VALIDATED against
#: the declared tools afterwards, so `call:` in prose cannot become a call.
_JSON_CALL_RE = re.compile(r"call:\s*([A-Za-z_][A-Za-z0-9_.\-]*)\s*(\{)")

#: Gemma 4's NATIVE call frame, which the gateway asks for when the model's
#: `tool_dialect` is "gemma4" (its default). All three are single special
#: tokens (48, 49, 50) and arrive as text because server.py decodes special
#: tokens for this checkpoint. The model stops ON `<|tool_response>` -- it is
#: in its generation config's EOS list -- so it normally never reaches here;
#: it is dropped if it does, because it is a request for input, not output.
_NATIVE_OPEN = "<|tool_call>"
_NATIVE_CLOSE = "<tool_call|>"
_NATIVE_RESPONSE = "<|tool_response>"
_NATIVE_CALL_RE = re.compile(r"\s*call:\s*([A-Za-z_][A-Za-z0-9_.\-]*)\s*(\{.*)\Z", re.DOTALL)


class GemmaToolParser:
    """Qwen3XmlFSM, plus Gemma's thinking tag and its JSON call shape."""

    def __init__(
        self,
        tools: list[dict] | None = None,
        in_triple_backticks: bool = False,
        in_single_backtick: bool = False,
        start_in_thinking: bool = False,
        start_in_tool_call: bool = False,
    ):
        self._inner = Qwen3XmlFSM(
            tools=tools,
            in_triple_backticks=in_triple_backticks,
            in_single_backtick=in_single_backtick,
            start_in_thinking=start_in_thinking,
            start_in_tool_call=start_in_tool_call,
        )
        self._tools = tools or []
        self._tool_names = {
            (t.get("function") or {}).get("name")
            for t in self._tools
            if (t.get("function") or {}).get("name")
        }

        #: Whether a thought channel is open, so a `<channel|>` closer is
        #: rewritten only when it belongs to us. `start_in_thinking` means the
        #: prompt already opened one, which is exactly this state.
        self._in_thought = bool(start_in_thinking)

        #: Where we are in a tool call, so the bare-name parameter rewrite
        #: fires there and nowhere else. `start_in_tool_call` is the LODESTAR
        #: prefill, which ends the prompt with an open `<tool_call>`.
        self._in_tool_call = bool(start_in_tool_call)

        #: The bare-name parameter tag currently open, "" when none. While one
        #: is open the value may contain any markup at all, so nothing but its
        #: own closer is recognised.
        self._open_param = ""

        #: Every argument name the declared tools have. The bare-name rewrite
        #: fires only for these, which is what keeps it off prose -- and with
        #: no tools registered the set is empty and it cannot fire at all.
        self._param_names: set[str] = set()
        for tool in self._tools:
            schema = ((tool.get("function") or {}).get("parameters") or {})
            for key in (schema.get("properties") or {}):
                if isinstance(key, str) and key:
                    self._param_names.add(key)

        #: Held-back tail of the incoming stream: a chunk may end halfway
        #: through `<|channel>thought`, and rewriting half a marker is worse
        #: than waiting one chunk for the rest of it.
        self._hold = ""

        #: Content the base FSM produced that might still turn into a
        #: `call:name{...}`. Empty almost always; see `_drain_content`.
        self._content_hold = ""

        #: The index space is this class's, because calls arrive from two
        #: sources. `_index_map` translates the inner FSM's numbering into it.
        self._next_index = 0
        self._index_map: dict[int, int] = {}
        self._own_calls = 0

        #: The body of a native `<|tool_call>` still arriving, or None. Held
        #: whole until its closer, because a call is emitted at once -- a
        #: downstream gate may still drop it, which it never could if half the
        #: arguments had already reached the client.
        self._native_buf: str | None = None

    # -- BaseToolParser ------------------------------------------------------

    @property
    def tool_call_count(self) -> int:
        return self._next_index

    @property
    def in_thinking(self) -> bool:
        """Delegated, deliberately.

        `<|channel>thought` is rewritten to `<think>` before the inner FSM sees
        it, so the inner FSM's answer is the true one -- and it is the answer
        every reasoning guard in this gateway reads.
        """
        return self._inner.in_thinking

    def feed(self, chunk: str) -> Iterator[FsmEvent]:
        text = self._hold + (chunk or "")
        text, self._hold = self._split_holdback(text)
        if not text:
            return
        yield from self._feed_text(text)

    def finish(self, finish_reason: str = "stop") -> Iterator[FsmEvent]:
        # Whatever was held back for a tag that never completed is ordinary
        # text: the turn is over, so nothing more is coming to finish it.
        tail, self._hold = self._hold, ""
        if tail:
            yield from self._feed_text(tail)
        if self._native_buf is not None:
            # The turn ended inside a native call. Its closer may simply not
            # have been decoded; try the body as it stands, and if it is not a
            # complete object it goes out as the text it was.
            body, self._native_buf = self._native_buf, None
            yield from self._emit_native(body)
        for event in self._inner.finish(finish_reason):
            yield from self._translate(event)
        yield from self._drain_content(final=True)

    # -- The native call frame ----------------------------------------------

    def _feed_inner(self, text: str) -> Iterator[FsmEvent]:
        if text:
            for event in self._inner.feed(self._rewrite(text)):
                yield from self._translate(event)

    def _feed_text(self, text: str) -> Iterator[FsmEvent]:
        """Route `<|tool_call>...<tool_call|>` around the base FSM.

        Text before an opener is fed through FIRST, so `_in_thought` is
        current when the opener is judged: a call written inside the thought
        channel is the model reasoning about a call, not making one.
        """
        while text:
            if self._native_buf is not None:
                self._native_buf += text
                text = ""
                end = self._native_buf.find(_NATIVE_CLOSE)
                if end == -1:
                    return
                body = self._native_buf[:end]
                text = self._native_buf[end + len(_NATIVE_CLOSE):]
                self._native_buf = None
                yield from self._emit_native(body)
                continue
            text = text.replace(_NATIVE_RESPONSE, "")
            idx = text.find(_NATIVE_OPEN)
            if idx == -1:
                yield from self._feed_inner(text)
                return
            yield from self._feed_inner(text[:idx])
            text = text[idx + len(_NATIVE_OPEN):]
            if self._in_thought:
                yield from self._feed_inner(_NATIVE_OPEN)
                continue
            # Anything held as possible content must leave before the call,
            # or the client sees the two out of order.
            yield from self._drain_content(final=True)
            self._native_buf = ""

    def _emit_native(self, body: str) -> Iterator[FsmEvent]:
        """`call:NAME{...}` in Gemma's own value syntax -> one tool call."""
        match = _NATIVE_CALL_RE.match(body)
        params = None
        name = ""
        if match:
            name = match.group(1)
            obj, rest = split_object(match.group(2))
            if obj is not None and not rest.strip():
                try:
                    params = parse_args(obj)
                except GemmaSyntaxError as exc:
                    _log.warning("GEMMA: native call:%s did not parse (%s); "
                                 "emitting it as text rather than guessing.",
                                 name, exc)
        if params is None:
            # Visible beats silent: a call the gateway cannot read is shown as
            # what it was, so neither the user nor the model mistakes it for
            # one that ran.
            yield FsmEvent(EventType.CONTENT, _NATIVE_OPEN + body + _NATIVE_CLOSE)
            return
        yield from self._emit_call(name, params)

    # -- Thinking markers ----------------------------------------------------

    def _param_rewrite(self, text: str, i: int) -> tuple[str, int] | None:
        """`<filePath>` -> `<parameter=filePath>`, and its closer to match.

        **MEASURED 2026-09-13.** Asked for the dialect this gateway describes
        to every model, Gemma 4 answered:

            <tool_call>
            <function=read>
            <filePath>C:/Users/.../notes.txt</filePath>
            </function>
            </tool_call>

        The `<function=` and the frame are exactly right; the parameter drops
        the `parameter=` and uses the argument's NAME as the tag. The base FSM
        saw a tag that is not `<parameter=...>`, collected nothing, and emitted
        `read({})` -- a call that parses as JSON and cannot read anything. The
        gateway logged `CRITICAL FIELD MISSING: tool=read field=filePath` for
        the same turn while an end-to-end check called it a pass.

        **Scoped so it cannot fire on prose.** Three conditions together: we
        are inside a `<tool_call>`, we are not inside a parameter VALUE (which
        may legitimately contain any tag at all), and the name is one the
        declared tools actually have. A model with no tools registered rewrites
        nothing, because the candidate set is empty.
        """
        if not self._in_tool_call or self._open_param or not self._param_names:
            return None
        if not text.startswith("<", i):
            return None
        close = text.find(">", i + 1)
        if close == -1 or "\n" in text[i:close]:
            # No tag this parser knows spans a line -- the same rule the base
            # FSM learned from `10<V<=30` being read as a tool call opener.
            return None
        name = text[i + 1:close]
        if name not in self._param_names:
            return None
        self._open_param = name
        return f"<parameter={name}>", close + 1

    def _rewrite(self, text: str) -> str:
        """Gemma's thinking markers into the base FSM's, left to right.

        **Stateful, and the closer is why.** `<channel|>` closes whichever
        channel is open, and `thought` is only one of them -- the model's
        template also frames tool calls and tool responses. A blind
        `str.replace` would turn some future `<channel|>` into a `</think>`
        with no `<think>` before it, which is a fabricated closer: the base
        FSM would leave thinking it was never in, and everything after it
        would be labelled wrong. So a closer is rewritten only while a thought
        channel is actually open, and otherwise passes through as text, where
        it is at least VISIBLE.

        Left to right rather than one `replace` per marker, because the order
        markers appear in is what decides whether a given closer belongs to us.
        """
        out: list[str] = []
        i = 0
        while i < len(text):
            # Frame bookkeeping first, and it CONSUMES NOTHING: `<tool_call>`
            # and `<function=` are the base FSM's own tags and must reach it
            # byte for byte. All this does is remember where we are, so the
            # parameter rewrite below knows whether it is allowed to fire.
            self._observe_frame(text, i)

            replacement, width = self._match_rewrite(text, i)
            if replacement is not None:
                out.append(replacement)
                i += width
                continue

            out.append(text[i])
            i += 1
        return "".join(out)

    def _observe_frame(self, text: str, i: int) -> None:
        """Where we are in a tool call. Reads; never rewrites."""
        if text.startswith("<tool_call>", i):
            self._in_tool_call = True
            self._open_param = ""
        elif text.startswith("</tool_call>", i):
            self._in_tool_call = False
            self._open_param = ""

    def _match_rewrite(self, text: str, i: int) -> tuple[str | None, int]:
        """The one rewrite that applies at `i`, if any, and how much it eats."""
        # A parameter VALUE may contain anything, including every marker here,
        # so while one is open the only thing recognised is its own closer.
        if self._open_param:
            closer = f"</{self._open_param}>"
            if text.startswith(closer, i):
                self._open_param = ""
                return "</parameter>", len(closer)
            return None, 0

        if not self._in_thought:
            for marker, replacement in _OPEN_REWRITES:
                if text.startswith(marker, i):
                    self._in_thought = True
                    return replacement, len(marker)
        else:
            for marker, replacement in _CLOSE_REWRITES:
                if text.startswith(marker, i):
                    self._in_thought = False
                    return replacement, len(marker)
            # Inside a thought channel nothing else is rewritten: the model is
            # reasoning, and reasoning about a tool call is not one.
            return None, 0

        opened = self._param_rewrite(text, i)
        if opened is not None:
            return opened[0], opened[1] - i
        return None, 0

    # -- Holding back a split tag -------------------------------------------

    def _split_holdback(self, text: str) -> tuple[str, str]:
        """(safe to process now, keep for the next chunk).

        A tail is held only when it could still GROW into one of the markers
        -- `<`, `<|chan`, `</thou`. Holding on any `<` at all would stall the
        stream on ordinary prose, which is the failure the base FSM already
        pays for elsewhere.
        """
        markers = [m for m, _ in _OPEN_REWRITES + _CLOSE_REWRITES]
        # The bare-name parameter tags are as long as the longest argument
        # name the declared tools have, so the hold-back window grows with
        # them. A fixed window would split `<filePath>` and emit half of it.
        markers += [f"<{name}>" for name in self._param_names]
        markers += [f"</{name}>" for name in self._param_names]
        # **The frame tags too, even though nothing rewrites them.**
        # `_observe_frame` has to see `<tool_call>` whole to know where it is,
        # and at one character per chunk it never would: the hold-back only
        # kept prefixes of the REWRITES, so the frame arrived two characters
        # at a time and was never recognised. The bare-name parameter rewrite
        # then fired nowhere, on a stream, while passing on whole strings --
        # which is the exact failure this file's four chunk sizes exist for.
        markers += _FRAME_TAGS
        markers += (_NATIVE_OPEN, _NATIVE_CLOSE, _NATIVE_RESPONSE)
        longest = max(len(m) for m in markers)
        for keep in range(min(longest - 1, len(text)), 0, -1):
            tail = text[-keep:]
            if any(marker.startswith(tail) for marker in markers):
                return text[:-keep], tail
        return text, ""

    # -- Index translation ---------------------------------------------------

    def _translate(self, event: FsmEvent) -> Iterator[FsmEvent]:
        """Renumber an inner event into this parser's index space.

        Content is not passed straight through: it goes to `_drain_content`,
        which is where a `call:name{...}` would be found.
        """
        if event.type is EventType.CONTENT:
            self._content_hold += str(event.data or "")
            yield from self._drain_content()
            return

        if event.type is EventType.REASONING:
            yield event
            return

        data = event.data if isinstance(event.data, dict) else {}
        raw_index = data.get("index")
        inner_index = int(raw_index) if isinstance(raw_index, int) else -1
        if event.type is EventType.TOOL_CALL_START:
            # Anything still held is text that came BEFORE this call, and it
            # has to leave first or the client sees them out of order.
            yield from self._drain_content(final=True)
            self._index_map[inner_index] = self._next_index
            self._next_index += 1
        mapped = self._index_map.get(inner_index, inner_index)
        if mapped != inner_index:
            data = {**data, "index": mapped}
            event = FsmEvent(event.type, data)
        yield event

    # -- The JSON call shape -------------------------------------------------

    def _drain_content(self, final: bool = False) -> Iterator[FsmEvent]:
        """Emit held content, pulling out any `call:name{...}` on the way.

        Content is held only from the first `call:` onwards, so ordinary prose
        streams with no added latency at all. That matters: this runs on every
        content token of every turn.
        """
        while self._content_hold:
            match = _JSON_CALL_RE.search(self._content_hold)
            if match is None:
                if final:
                    text, self._content_hold = self._content_hold, ""
                    yield FsmEvent(EventType.CONTENT, text)
                    return
                # A call that has begun but not finished arriving. Holding
                # from `call:` onward is what makes this work at all on a real
                # stream: the first version held back only a partial copy of
                # the WORD, so `call:read` went out as text one chunk before
                # its `{` arrived and the call was never recognised. Found by
                # feeding a fixture seven characters at a time, which is the
                # only way this class of defect shows up.
                start = self._partial_call_start(self._content_hold)
                if start is None:
                    start = self._safe_cut(self._content_hold)
                if start:
                    text = self._content_hold[:start]
                    self._content_hold = self._content_hold[start:]
                    yield FsmEvent(EventType.CONTENT, text)
                return

            name = match.group(1)
            if name not in self._tool_names:
                # `call:` followed by something that is not a tool. Prose.
                # Emit up to and including the name so the scan moves on and
                # cannot loop on the same position forever.
                end = match.end(1)
                text = self._content_hold[:end]
                self._content_hold = self._content_hold[end:]
                yield FsmEvent(EventType.CONTENT, text)
                continue

            before = self._content_hold[:match.start()]
            body, rest = _split_json_object(self._content_hold[match.start(2):])
            if body is None:
                if not final:
                    # The object is still arriving.
                    if before:
                        self._content_hold = self._content_hold[match.start():]
                        yield FsmEvent(EventType.CONTENT, before)
                    return
                # The turn ended mid-object. Nothing can be done with half a
                # call, and inventing arguments is how a gateway destroys a
                # file -- so it goes out as the text it literally was.
                text, self._content_hold = self._content_hold, ""
                yield FsmEvent(EventType.CONTENT, text)
                return

            if before:
                yield FsmEvent(EventType.CONTENT, before)
            self._content_hold = rest
            yield from self._emit_json_call(name, body)

    @staticmethod
    def _partial_call_start(text: str) -> int | None:
        """Where a call that is still arriving begins, or None.

        Viability is checked rather than assumed, because `call:` also occurs
        in prose -- "call: me later" -- and holding content to the end of the
        turn on the strength of a colon would stall the stream for a sentence
        that was never a call. Only a keyword followed by something that could
        still GROW into `name{` is held.
        """
        index = text.find("call:")
        if index == -1:
            return None
        rest = text[index + len("call:"):]
        stripped = rest.lstrip()
        if not stripped:
            return index          # nothing after it yet, or only whitespace
        if stripped[0].isalpha() or stripped[0] == "_":
            consumed = 0
            while consumed < len(stripped) and (
                    stripped[consumed].isalnum() or stripped[consumed] in "_.-"):
                consumed += 1
            if consumed == len(stripped):
                return index      # the name may still be growing
            tail = stripped[consumed:]
            # A `{` here means the full pattern would have matched already, so
            # what is incomplete is the object rather than the opener.
            if tail[0] == "{" or not tail.strip():
                return index
            return None
        if stripped[0] == "{":
            return index
        return None

    @staticmethod
    def _safe_cut(text: str) -> int:
        """How much of `text` cannot be the start of a `call:` prefix."""
        keyword = "call:"
        for keep in range(min(len(keyword) - 1, len(text)), 0, -1):
            if keyword.startswith(text[-keep:]):
                return len(text) - keep
        return len(text)

    def _emit_json_call(self, name: str, body: str) -> Iterator[FsmEvent]:
        """Turn one `call:name{...}` into the same events the XML path emits.

        The whole call is emitted at once rather than streamed argument by
        argument, and that is correct here: the object was already complete
        before this ran, so there is nothing to stream. It also means the call
        can still be DROPPED by a downstream gate -- a call whose arguments
        have already reached the client can never be, which is the rule
        AGENTS.md states for every gate that decides at TOOL_CALL_END.
        """
        try:
            params = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            _log.warning("GEMMA: call:%s carried unparseable JSON; emitting it "
                         "as text rather than guessing at the arguments.", name)
            yield FsmEvent(EventType.CONTENT, f"call:{name}{body}")
            return
        if not isinstance(params, dict):
            yield FsmEvent(EventType.CONTENT, f"call:{name}{body}")
            return
        yield from self._emit_call(name, params)

    def _emit_call(self, name: str, params: dict) -> Iterator[FsmEvent]:
        """One complete call, as the same events the XML path emits."""
        index = self._next_index
        self._next_index += 1
        self._own_calls += 1
        call_id = f"call_{uuid.uuid4().hex[:24]}"
        yield FsmEvent(EventType.TOOL_CALL_START,
                       {"index": index, "id": call_id, "name": name})
        first = True
        string_params: dict[str, str] = {}
        for key, value in params.items():
            rendered = (value if isinstance(value, str)
                        else json.dumps(value, ensure_ascii=False))
            string_params[key] = rendered
            param_type = "string" if isinstance(value, str) else "json"
            yield FsmEvent(EventType.PARAM_START, {
                "index": index, "name": key, "type": param_type,
                "is_first": first,
            })
            yield FsmEvent(EventType.PARAM_CHUNK, {
                "index": index, "text": rendered, "param_type": param_type,
            })
            yield FsmEvent(EventType.PARAM_END, {
                "index": index, "name": key, "param_type": param_type,
            })
            first = False
        yield FsmEvent(EventType.TOOL_CALL_END, {
            "index": index, "id": call_id, "name": name,
            "params": string_params, "auto_closed": False,
        })


def _split_json_object(text: str) -> tuple[str | None, str]:
    """(the first complete JSON object, the rest), or (None, text).

    Brace counting with string awareness, because an argument value that
    contains `}` is the ordinary case here -- a `write` whose content is code
    has plenty. A naive scan to the first `}` would truncate every one of them.
    """
    if not text or text[0] != "{":
        return None, text
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[:i + 1], text[i + 1:]
    return None, text
