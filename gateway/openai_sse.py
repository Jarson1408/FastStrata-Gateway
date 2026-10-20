"""
OpenAI SSE emitter — converts FSM events into OpenAI Chat Completion chunks.

Responsibilities:
  - Format SSE data frames in OpenAI format
  - JSON-escape string parameter values in tool call arguments
  - Track per-tool-call state for proper JSON construction
  - Filter hallucinated tool calls (blocked/unknown names)
  - Resolve tool name aliases (e.g. read_file → read)
  - Heartbeat generation

This module does NOT handle:
  - Missing argument filling (that's the pipeline's job)
  - Value coercion (that's tool_schema / _legacy's job)
  - Network I/O (that's upstream's job)
"""
from __future__ import annotations

import json
import logging
from typing import Any

from gateway.qwen_xml_fsm import EventType, FsmEvent
from gateway.tool_schema import TOOL_NAME_ALIASES, TOOL_FAMILIES, BLOCKED_TOOL_NAMES

_log = logging.getLogger("gateway.sse")


class OpenAISSEEmitter:
    """Builds OpenAI-format SSE chunks for streaming responses."""

    def __init__(self, chat_id: str, model: str, created: int,
                 tool_names: set[str] | None = None,
                 allow_skills: bool = True):
        self._chat_id = chat_id
        self._model = model
        self._created = created
        self._tool_names = tool_names or set()  # registered tool names from client
        self._allow_skills = allow_skills

        # Per-tool-call state
        self._has_param: dict[int, bool] = {}     # index → has any param been started
        self._param_type: dict[int, str] = {}      # index → current param type
        self._suppressed: set[int] = set()          # indices of filtered-out tool calls
        self._buffered_skills: set[int] = set()    # indices of skill calls being buffered
        self._buffered_tasks: set[int] = set()     # indices of task calls (subagent marker rewrite)
        self._deferred: set[int] = set()           # indices buffered until TOOL_CALL_END
        # indices whose opening chunk has reached the client: from here on a
        # call can only be completed, never withdrawn without a parse error
        self._streamed: set[int] = set()

        # Content & reasoning tracking (for auto-healing empty content)
        self._emitted_content_chars: int = 0
        self._emitted_tool_call_count: int = 0
        self._reasoning_chunks: list[str] = []
        # What the client actually received as assistant text. For a subagent
        # this is the whole deliverable its parent will read, so the gateway
        # has to be able to see whether it amounts to anything.
        self._content_chunks: list[str] = []

    @property
    def has_emitted_content(self) -> bool:
        return self._emitted_content_chars > 0

    @property
    def has_emitted_tool_calls(self) -> bool:
        return self._emitted_tool_call_count > 0

    @property
    def reasoning_text(self) -> str:
        return "".join(self._reasoning_chunks)

    def reset_reasoning(self) -> None:
        """Forget the reasoning so far — a new attempt at the same turn.

        The gateway can abandon a looping generation and start another within
        the same client request. Without this the guards would judge the fresh
        attempt against the abandoned one's text and cut it instantly.
        """
        self._reasoning_chunks.clear()

    def reset_tool_call_state(self) -> None:
        """Forget per-attempt tool call filtering/buffering state for a fresh generation attempt."""
        self._has_param.clear()
        self._param_type.clear()
        self._suppressed.clear()
        self._buffered_skills.clear()
        self._buffered_tasks.clear()
        self._deferred.clear()
        self._streamed.clear()

    @property
    def emitted_content_chars(self) -> int:
        """How much assistant text has reached the client, ignoring whitespace."""
        return self._emitted_content_chars

    @property
    def content_text(self) -> str:
        return "".join(self._content_chunks)

    def has_emitted_param(self, index: int) -> bool:
        """Check if any parameter was emitted for a tool call."""
        return self._has_param.get(index, False)

    def is_suppressed(self, index: int) -> bool:
        """Check if a tool call index is being filtered out."""
        return index in self._suppressed

    def suppress(self, index: int) -> None:
        """Mark a tool call index as filtered out (no chunks, no finish=tool_calls)."""
        self._suppressed.add(index)

    def defer(self, index: int) -> None:
        """Buffer a tool call: emit nothing until TOOL_CALL_END decides its fate."""
        self._deferred.add(index)

    def is_deferred(self, index: int) -> bool:
        return index in self._deferred

    def has_streamed(self, index: int) -> bool:
        """True once this call's opening chunk has gone to the client.

        Every gate that decides at TOOL_CALL_END has to ask this first. A call
        suppressed after its name and arguments streamed never receives its
        closing brace, and the client reports `tool input could not be parsed
        as JSON` -- the failure deferral exists to prevent. Measured
        2026-10-17: three `read` calls broken that way by the intra-turn
        duplicate guard, and a 15-turn `glob {}` loop the day before.
        """
        return index in self._streamed

    def is_buffered_task(self, index: int) -> bool:
        """True if a `task` call is being held for the subagent-directive rewrite."""
        return index in self._buffered_tasks

    def is_buffered_skill(self, index: int) -> bool:
        """True if a `skill` call is being held for the read rewrite."""
        return index in self._buffered_skills

    def resolve_name(self, name: str) -> str | None:
        """Canonical tool name, or None when the client never declared it.

        Public because the request pipeline has to resolve aliases before
        enforcement — checking the raw alias against the declared tools used to
        drop a perfectly good `read_file` as hallucinated.
        """
        return self._resolve_name(name)

    def _resolve_name(self, name: str) -> str | None:
        """Resolve a tool name — returns canonical name or None if blocked/unknown."""
        if name.lower() in BLOCKED_TOOL_NAMES:
            return None
        if self._tool_names and name in self._tool_names:
            return name
        if name == "skill" and self._allow_skills:
            return "skill"
        # Check families (e.g. bash <-> shell, task <-> subagent, patch <-> apply_patch)
        if self._tool_names:
            for family in TOOL_FAMILIES:
                if name in family:
                    for member in family:
                        if member in self._tool_names:
                            _log.info("STREAM TOOL FAMILY ALIAS: %s → %s", name, member)
                            return member
        # Check alias
        canonical = TOOL_NAME_ALIASES.get(name)
        if canonical and self._tool_names and canonical in self._tool_names:
            _log.info("STREAM TOOL NAME ALIAS: %s → %s", name, canonical)
            return canonical
        if not self._tool_names:
            # No tool schema available — pass through as-is
            return name
        return None

    # -- Low-level chunk builders ----------------------------------------------

    def _chunk(self, delta: dict, finish: str | None = None, usage: dict | None = None) -> bytes:
        c: dict[str, Any] = {
            "id": self._chat_id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self._model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage is not None:
            c["usage"] = usage
        return b"data: " + json.dumps(c).encode("utf-8") + b"\n\n"

    def role_chunk(self) -> bytes:
        return self._chunk({"role": "assistant"})

    def content_chunk(self, text: str) -> bytes:
        if isinstance(text, str):
            if text:
                self._content_chunks.append(text)
            if text.strip():
                self._emitted_content_chars += len(text.strip())
        return self._chunk({"content": text})

    def reasoning_chunk(self, text: str) -> bytes:
        if isinstance(text, str):
            self._reasoning_chunks.append(text)
        return self._chunk({"reasoning_content": text, "reasoning": text})

    def tool_call_start_chunk(self, index: int, call_id: str, name: str) -> bytes:
        self._emitted_tool_call_count += 1
        self._streamed.add(index)
        return self._chunk({
            "tool_calls": [{
                "index": index,
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": ""},
            }]
        })

    def tool_call_arg_chunk(self, index: int, arguments: str) -> bytes:
        return self._chunk({
            "tool_calls": [{
                "index": index,
                "function": {"arguments": arguments},
            }]
        })

    def finish_chunk(self, finish_reason: str, usage: dict | None = None) -> bytes:
        return self._chunk({}, finish=finish_reason, usage=usage)

    def done(self) -> bytes:
        return b"data: [DONE]\n\n"

    def heartbeat(self) -> bytes:
        return b": heartbeat\n\n"

    def error_chunk(self, message: str) -> bytes:
        return b"data: " + json.dumps({"error": {"message": message}}).encode() + b"\n\n"

    # -- Event-to-SSE conversion -----------------------------------------------

    def emit_event(self, event: FsmEvent, is_summary: bool = False) -> list[bytes]:
        """Convert an FSM event into a list of SSE byte chunks.

        TOOL_CALL_END is NOT handled here — the pipeline must handle it
        to compute missing arguments and emit the closing brace.

        Returns an empty list for suppressed events (summary mode).
        """
        t = event.type
        d = event.data

        if t == EventType.CONTENT:
            return [self.content_chunk(d)]

        if t == EventType.REASONING:
            return [self.reasoning_chunk(d)]

        if t == EventType.TOOL_CALL_START:
            if is_summary:
                return []
            idx = d["index"]
            raw_name = d["name"]
            resolved = self._resolve_name(raw_name)
            if resolved is None:
                _log.warning(
                    "STREAM: HALLUCINATED TOOL CALL SUPPRESSED index=%d name=%r",
                    idx, raw_name,
                )
                self._suppressed.add(idx)
                return []
            if resolved == "skill":
                if self._tool_names and "skill" in self._tool_names:
                    # Client declared native skill tool (OpenCode 2.0): emit directly!
                    pass
                else:
                    self._buffered_skills.add(idx)
                    return []
            if resolved in ("task", "subagent"):
                # Buffer task/subagent calls: the pipeline appends the LODESTAR
                # subagent directive to the prompt at TOOL_CALL_END so the
                # spawned subagent is detectable and skips the protocol.
                self._buffered_tasks.add(idx)
                return []
            if resolved != raw_name:
                d["name"] = resolved
            return [self.tool_call_start_chunk(idx, d["id"], resolved)]

        if t == EventType.PARAM_START:
            if is_summary or d["index"] in self._suppressed or d["index"] in self._buffered_skills or d["index"] in self._buffered_tasks or d["index"] in self._deferred:
                return []
            idx = d["index"]
            self._has_param[idx] = True
            self._param_type[idx] = d["type"]
            if d["type"] == "string":
                prefix = ('{\n  "' if d["is_first"] else ',\n  "') + d["name"] + '": "'
            else:
                prefix = ('{\n  "' if d["is_first"] else ',\n  "') + d["name"] + '": '
            return [self.tool_call_arg_chunk(idx, prefix)]

        if t == EventType.PARAM_CHUNK:
            if is_summary or d["index"] in self._suppressed or d["index"] in self._buffered_skills or d["index"] in self._buffered_tasks or d["index"] in self._deferred:
                return []
            idx = d["index"]
            text = d["text"]
            ptype = d.get("param_type", self._param_type.get(idx, "string"))
            if ptype == "string":
                escaped = json.dumps(text, ensure_ascii=False)[1:-1]
                return [self.tool_call_arg_chunk(idx, escaped)]
            return [self.tool_call_arg_chunk(idx, text)]

        if t == EventType.PARAM_END:
            if is_summary or d["index"] in self._suppressed or d["index"] in self._buffered_skills or d["index"] in self._buffered_tasks or d["index"] in self._deferred:
                return []
            idx = d["index"]
            ptype = d.get("param_type", self._param_type.get(idx, "string"))
            suffix = '"' if ptype == "string" else ""
            if suffix:
                return [self.tool_call_arg_chunk(idx, suffix)]
            return []

        # TOOL_CALL_END is handled by the pipeline (not here)
        return []
