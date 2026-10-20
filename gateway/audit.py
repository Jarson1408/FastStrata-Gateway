"""
JSONL audit logger with rotation.

Provides two log channels:
  1. Diagnostic log (gateway.log) — human-readable, debug-level
  2. Audit log (gateway_events.jsonl) — machine-readable, structured events

Both use RotatingFileHandler with configurable max size and backup count.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, TYPE_CHECKING

from gateway.prompts import prompts_version

if TYPE_CHECKING:
    from gateway.config import GatewayConfig

from gateway import metrics

_log = logging.getLogger("gateway.audit")

# Delimiters of the block context_injector appends to the system message, and
# the header of the checklist nested inside it. Measured rather than imported
# as content: the point is to know how much of the prompt the gateway wrote.
_LODESTAR_OPEN = "=== LODESTAR PROTOCOL ==="
_LODESTAR_CLOSE = "=== END LODESTAR PROTOCOL ==="
# The volatile frame the injector emits as the last message. It is measured
# together with the block above: both are prompt the gateway wrote, and
# splitting them was a KV-cache fix, not a change in what we are accountable for.
_TURN_OPEN = "=== LODESTAR TURN ==="
_TURN_CLOSE = "=== END LODESTAR TURN ==="
_TODO_MARKER = "[LODESTAR TODO]"


def summarize_prompt(prompt: str) -> dict[str, Any]:
    """Measure a rendered prompt without storing it.

    Everything the gateway adds to a request — the protocol block, the state
    directive, the checklist, the prefill — lands in this one string, and none
    of it was logged before: the audit recorded that a request happened, not
    what it said. A digest plus the size of the gateway-authored block is
    enough to answer the two questions that matter after a bad session: did
    the prompt change between these turns, and how much of it was ours.

    The digest is over the exact bytes sent upstream, so it also catches the
    KV-prefix churn that silently costs a full prefill.
    """
    text = prompt or ""

    def _span(open_marker: str, close_marker: str) -> int:
        start = text.find(open_marker)
        if start == -1:
            return 0
        end = text.find(close_marker, start)
        if end == -1:
            # Truncated block: still worth reporting what reached the model.
            return len(text) - start
        return end + len(close_marker) - start

    lodestar_chars = (_span(_LODESTAR_OPEN, _LODESTAR_CLOSE)
                      + _span(_TURN_OPEN, _TURN_CLOSE))
    return {
        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "prompt_chars": len(text),
        "lodestar_chars": lodestar_chars,
        "has_todo": _TODO_MARKER in text,
    }


class AuditLogger:
    """JSONL structured event logger.

    Events are written to gateway_events.jsonl as one JSON object per line.
    This file can be ingested by log aggregators (ELK, Loki, etc.) or
    analyzed with jq/Python for debugging.

    Usage::

        audit = AuditLogger(config)
        audit.log_request(request_id="req_abc123", model="coding-35b", ...)
        audit.log_tool_call(request_id="req_abc123", tool_name="bash", ...)
        audit.log_response(request_id="req_abc123", finish_reason="stop", ...)
    """

    def __init__(self, config: "GatewayConfig"):
        self._logger = logging.getLogger("gateway.events")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False  # don't duplicate to root logger

        log_dir = config.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        max_bytes = getattr(config, "audit_max_bytes", 50 * 1024 * 1024)  # 50MB
        backup_count = getattr(config, "audit_backup_count", 3)

        handler = RotatingFileHandler(
            str(log_dir / "gateway_events.jsonl"),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        # Raw JSON — no extra formatting from the logging framework
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

    def _emit(self, event_type: str, **fields: Any) -> None:
        """Write a single JSON line to the audit log."""
        record = {
            "ts": time.time(),
            "event": event_type,
            **fields,
        }
        try:
            self._logger.info(json.dumps(record, ensure_ascii=False, default=str))
        except Exception as e:
            _log.warning("Audit log write failed: %s", e)

        # /metrics hangs off this one call rather than off each guard: every
        # audit event passes through here, so a new event type is counted
        # without anybody remembering to, and an event that stops being logged
        # stops being counted instead of reporting a stale number. It never
        # raises -- see gateway/metrics.py.
        metrics.record_event(event_type, fields)

    # -- Request lifecycle events ------------------------------------------

    def log_request(
        self,
        request_id: str,
        model: str,
        stream: bool,
        message_count: int,
        tools_count: int,
        is_summary: bool = False,
        session_id: str | None = None,
        lodestar_state: int | None = None,
    ) -> None:
        """Log an incoming chat completion request.

        `prompts_version` is stamped automatically: without it a change in model
        behaviour cannot be tied to the prompt revision that caused it, which is
        the whole point of versioning them separately from the code.

        `session_id` is the workspace the request came from. request_id is
        unique per HTTP call, so without it there is no way to reassemble the
        turns of one task from the JSONL — which is what makes the file usable
        for anything beyond grepping single requests.
        """
        self._emit(
            "request",
            request_id=request_id,
            session_id=session_id,
            model=model,
            stream=stream,
            message_count=message_count,
            tools_count=tools_count,
            is_summary=is_summary,
            lodestar_state=lodestar_state,
            prompts_version=prompts_version(),
        )

    def log_prompt(
        self,
        request_id: str,
        prompt: str,
        session_id: str | None = None,
        model: str = "",
        lodestar_state: int | None = None,
        prefill: bool = False,
        sampling: dict[str, Any] | None = None,
        spill_count: int = 0,
        compaction: bool = False,
        dump_dir: Path | None = None,
    ) -> None:
        """Log what the model was actually shown, one event per upstream call.

        `log_request` records the request as the client sent it. By the time it
        reaches llama-server the gateway has merged system messages, injected
        the protocol block, pruned, possibly prefilled the assistant turn and
        adjusted sampling — so the two events describe different things, and
        only this one can be compared against the model's behaviour.

        `dump_dir` writes the full prompt to `<request_id>.txt` beside the
        JSONL. Off unless the operator turns it on: it contains user source.
        """
        fields = summarize_prompt(prompt)
        if dump_dir is not None:
            try:
                dump_dir.mkdir(parents=True, exist_ok=True)
                (dump_dir / f"{request_id}.txt").write_text(prompt or "", encoding="utf-8")
                fields["dump"] = str(dump_dir / f"{request_id}.txt")
            except OSError as e:
                # Never fail a request over a diagnostic write.
                _log.warning("Full-prompt dump failed for %s: %s", request_id, e)

        sampling = sampling or {}
        self._emit(
            "prompt",
            request_id=request_id,
            session_id=session_id,
            model=model,
            lodestar_state=lodestar_state,
            prefill=prefill,
            spill_count=spill_count,
            compaction=compaction,
            temperature=sampling.get("temperature"),
            top_p=sampling.get("top_p"),
            top_k=sampling.get("top_k"),
            min_p=sampling.get("min_p"),
            n_predict=sampling.get("n_predict"),
            **fields,
        )

    def log_response(
        self,
        request_id: str,
        finish_reason: str,
        tokens_in: int,
        tokens_out: int,
        duration_ms: int,
        tool_call_count: int = 0,
        session_id: str | None = None,
        synthesized_count: int = 0,
        ttft_ms: int | None = None,
        queue_wait_ms: int | None = None,
        cached_tokens: int | None = None,
    ) -> None:
        """Log a completed response.

        `synthesized_count` is how many tool calls the gateway had to make on
        the model's behalf — the direct measure of how well the model follows
        the protocol unaided.

        `duration_ms` alone cannot answer "why did that take so long", because
        it adds three unrelated things together. The other two split it:

          queue_wait_ms  waiting for the single upstream slot — someone else's
                         request, not this one's cost
          ttft_ms        upstream request sent -> first token back. With a warm
                         KV prefix this is small whatever the context size; a
                         large value on a large tokens_in means the prefix was
                         missed and the whole prompt was re-processed. That is
                         the number that made the 2026-08-31 system-message
                         split an argument from measurement instead of theory.

        Both are None where they cannot be observed: ttft_ms on the
        non-streaming path, which receives the whole body at once.

        `cached_tokens` is the third, and it is the only one of the three that
        is not inferred: ExLlamaV3 counts the KV pages a prompt actually hit
        and reports them at eos. Everything this project knows about prefix
        warmth on that engine was, until now, read off the RELATIONSHIP
        between ttft_ms and tokens_in -- a rate whose denominator had to be
        guessed at. None means nobody reported one (llama.cpp does not), and
        that is deliberately distinct from 0, which means the prefix was cold.
        """
        self._emit(
            "response",
            request_id=request_id,
            session_id=session_id,
            finish_reason=finish_reason,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
            ttft_ms=ttft_ms,
            queue_wait_ms=queue_wait_ms,
            cached_tokens=cached_tokens,
            tool_call_count=tool_call_count,
            synthesized_count=synthesized_count,
        )

    # -- Tool call events --------------------------------------------------

    def log_tool_call(
        self,
        request_id: str,
        tool_name: str,
        parse_status: str,  # "ok" | "failed" | "blocked"
        repair_applied: bool = False,
        schema_valid: bool = True,
        emitted: bool = True,
        blocked_reason: str | None = None,
        session_id: str | None = None,
        lodestar_state: int | None = None,
        arguments: str | None = None,
    ) -> None:
        """Log a tool call parse event.

        `parse_status` records what the gateway did with the call: "ok" when it
        went through untouched, "blocked" when the enforcer stopped it,
        "synthesized" when the gateway made the call the model should have
        made, "rewritten" when it was rewritten before emission.

        `arguments` is recorded only for calls the gateway itself authored.
        Those are the ones worth reading back — the client executes them
        unattended and the model never asked for them — and unlike a model's
        `write` they carry no user source. Model-issued arguments stay out of
        the log for the same reason full prompts do.
        """
        self._emit(
            "tool_call",
            request_id=request_id,
            session_id=session_id,
            tool_name=tool_name,
            parse_status=parse_status,
            repair_applied=repair_applied,
            schema_valid=schema_valid,
            emitted=emitted,
            blocked_reason=blocked_reason,
            lodestar_state=lodestar_state,
            arguments=arguments,
        )

    # -- Error events ------------------------------------------------------

    def log_error(
        self,
        request_id: str,
        error_type: str,
        message: str,
    ) -> None:
        """Log an error event."""
        self._emit(
            "error",
            request_id=request_id,
            error_type=error_type,
            message=message,
        )

    # -- Escalation events -------------------------------------------------

    def log_escalation(
        self,
        request_id: str,
        from_model: str,
        to_model: str,
        retry_depth: int,
        upstream: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Log a request routed to a different model because of a retry loop.

        Rare and consequential — it changes which machine sees the code — so
        it is its own event rather than a field on `prompt`.
        """
        self._emit(
            "escalation",
            request_id=request_id,
            session_id=session_id,
            from_model=from_model,
            to_model=to_model,
            retry_depth=retry_depth,
            upstream=upstream,
        )

    # -- Spill events ------------------------------------------------------

    def log_spill(
        self,
        request_id: str,
        spilled_count: int,
        bytes_before: int,
        bytes_after: int,
        session_id: str | None = None,
    ) -> None:
        """Log oversized tool results moved out of the prompt.

        Recorded per request rather than per result: what matters afterwards is
        how much context the policy actually bought, and whether it fired at
        all on the turn where the model lost the thread.
        """
        self._emit(
            "spill",
            request_id=request_id,
            session_id=session_id,
            spilled_count=spilled_count,
            bytes_before=bytes_before,
            bytes_after=bytes_after,
        )

    # -- Pruning events ----------------------------------------------------

    def log_prune(
        self,
        request_id: str,
        original_count: int,
        pruned_count: int,
        tokens_before: int,
        tokens_after: int,
        session_id: str | None = None,
    ) -> None:
        """Log a context pruning event."""
        self._emit(
            "prune",
            request_id=request_id,
            session_id=session_id,
            original_count=original_count,
            pruned_count=pruned_count,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )
