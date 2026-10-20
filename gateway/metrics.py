"""Prometheus-format counters for what only the gateway can see.

`gateway_events.jsonl` already records every request, response, tool call,
spill, prune and error, and `session_report.py` reads it. What it cannot do is
answer "how is it doing right now" without parsing a file that is 6.4 MB and
growing -- so `/metrics` keeps the same facts as in-process counters, and
scraping costs a string join.

**Everything hangs off `AuditLogger._emit`, which is the one place every audit
event passes through.** That is the whole integration: one call site, and the
rule "if it is in the audit it is in the metrics" keeps the two from drifting.
Counting at each guard instead would have been a dozen edits and a dozen
chances to forget the next one -- this file already records what happens to a
guard that has to be remembered in four places.

**No label may carry a value the gateway does not control.** A Prometheus
label is a separate time series per distinct value, so labelling by
`session_id` (a workspace path), by model name or by TOOL name -- all of which
come from the client -- turns a scrape endpoint into an unbounded dictionary
that is never freed. Only bounded, code-authored vocabularies appear here:
finish reasons, error types, parse statuses, LODESTAR states. `_ALLOWED`
enforces it rather than trusting the next person to remember, and an
unexpected value is folded into `other` rather than dropped, because a metric
that silently loses events is worse than a coarse one.

No dependency: the text format is a dozen lines and the four runtime
dependencies of this package are deliberately the four it needs.
"""
from __future__ import annotations

import threading
import time
from typing import Iterable

#: Seconds. The first bucket is well under a warm turn and the last is past the
#: 43.5-minute worst case measured on 2026-09-12, because a histogram whose top
#: bucket is +Inf-only cannot tell "slow" from "catastrophic".
_LATENCY_BUCKETS: tuple[float, ...] = (
    0.05, 0.25, 1.0, 5.0, 15.0, 60.0, 300.0, 900.0, 3600.0)

#: Bounded vocabularies, per metric label. Anything outside becomes "other".
_ALLOWED: dict[tuple[str, str], frozenset[str]] = {
    ("llmgw_requests_total", "kind"): frozenset({"agentic", "summary", "plain"}),
    ("llmgw_responses_total", "finish_reason"): frozenset(
        {"stop", "length", "tool_calls", "error", "none"}),
    ("llmgw_errors_total", "type"): frozenset({
        "bad_json", "unsupported_params", "upstream_busy", "upstream_error",
        "upstream_status", "upstream_stream_error", "upstream_stream_lost",
        "upstream_reported_error",
        "gateway_error", "stream_error", "repaired", "client_stripped_media"}),
    ("llmgw_tool_calls_total", "status"): frozenset(
        {"ok", "failed", "blocked", "synthesized", "rewritten"}),
    ("llmgw_lodestar_requests_total", "state"): frozenset({"1", "2", "3"}),
    ("llmgw_tokens_total", "direction"): frozenset({"in", "out"}),
}

_HELP: dict[str, tuple[str, str]] = {
    "llmgw_requests_total": ("counter", "Requests accepted, by kind"),
    "llmgw_responses_total": ("counter", "Responses completed, by finish reason"),
    "llmgw_errors_total": ("counter", "Errors recorded in the audit, by type"),
    "llmgw_tool_calls_total": ("counter", "Tool calls seen, by what happened to them"),
    "llmgw_lodestar_requests_total": ("counter", "Requests by LODESTAR state"),
    "llmgw_tokens_total": ("counter", "Tokens reported by the upstream"),
    "llmgw_cached_prompt_tokens_total": (
        "counter", "Prompt tokens the engine served from its KV cache"),
    "llmgw_spills_total": ("counter", "Oversized tool results written to disk"),
    "llmgw_prunes_total": ("counter", "Conversations pruned to fit the window"),
    "llmgw_escalations_total": ("counter", "Requests sent to the escalation model"),
    "llmgw_synthesized_calls_total": (
        "counter", "Tool calls the gateway made on the model's behalf"),
    "llmgw_queue_wait_seconds": (
        "histogram", "Wait for the single upstream slot"),
    "llmgw_ttft_seconds": ("histogram", "Upstream request sent to first token"),
    "llmgw_request_seconds": ("histogram", "Generation time, excluding the queue"),
    "llmgw_slot_in_use": ("gauge", "Upstream slots currently held"),
    "llmgw_queue_depth": ("gauge", "Requests waiting for a slot"),
    "llmgw_queue_depth_interactive": (
        "gauge", "Waiting requests somebody is watching"),
    "llmgw_uptime_seconds": ("gauge", "Seconds since this process started"),
}


class _Histogram:
    __slots__ = ("buckets", "counts", "total", "count")

    def __init__(self, buckets: tuple[float, ...]):
        self.buckets = buckets
        self.counts = [0] * (len(buckets) + 1)
        self.total = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        self.total += value
        self.count += 1
        for i, edge in enumerate(self.buckets):
            if value <= edge:
                self.counts[i] += 1
                return
        self.counts[-1] += 1

    def cumulative(self) -> list[int]:
        running = 0
        out = []
        for n in self.counts:
            running += n
            out.append(running)
        return out


class Metrics:
    """Counters, a few histograms, and no state the client can grow."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}
        self._histograms: dict[str, _Histogram] = {}
        self._started = time.time()

    # -- recording ----------------------------------------------------------
    def _key(self, name: str, labels: dict[str, str]):
        clean: list[tuple[str, str]] = []
        for label, value in sorted(labels.items()):
            allowed = _ALLOWED.get((name, label))
            if allowed is not None and value not in allowed:
                value = "other"
            clean.append((label, value))
        return (name, tuple(clean))

    def inc(self, name: str, value: int = 1, **labels: str) -> None:
        if value <= 0:
            return
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def observe(self, name: str, seconds: float) -> None:
        if seconds < 0:
            return
        with self._lock:
            hist = self._histograms.get(name)
            if hist is None:
                hist = self._histograms[name] = _Histogram(_LATENCY_BUCKETS)
            hist.observe(seconds)

    def reset(self) -> None:
        """Tests only. A scrape endpoint that can be zeroed from outside is a
        scrape endpoint somebody will zero by accident."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._started = time.time()

    # -- rendering ----------------------------------------------------------
    def render(self, gauges: dict[str, float] | None = None,
               info: dict[str, str] | None = None) -> str:
        lines: list[str] = []
        with self._lock:
            counters = dict(self._counters)
            histograms = {k: (h.buckets, h.cumulative(), h.total, h.count)
                          for k, h in self._histograms.items()}
            uptime = time.time() - self._started

        if info:
            lines.append("# HELP llmgw_build_info Version of the running gateway")
            lines.append("# TYPE llmgw_build_info gauge")
            lines.append(f"llmgw_build_info{{{_labels(info)}}} 1")

        for name in sorted({n for n, _ in counters}):
            lines.extend(_header(name))
            for (n, labels), value in sorted(counters.items()):
                if n == name:
                    lines.append(_sample(n, dict(labels), value))

        for name in sorted(histograms):
            buckets, cumulative, total, count = histograms[name]
            lines.extend(_header(name))
            # strict=False on purpose: cumulative() has one more entry than
            # buckets (the +Inf overflow), which is emitted by hand below.
            for edge, running in zip(buckets, cumulative, strict=False):
                lines.append(f'{name}_bucket{{le="{edge}"}} {running}')
            lines.append(f'{name}_bucket{{le="+Inf"}} {count}')
            lines.append(f"{name}_sum {total:.6f}")
            lines.append(f"{name}_count {count}")

        live = dict(gauges or {})
        live["llmgw_uptime_seconds"] = uptime
        for name in sorted(live):
            lines.extend(_header(name))
            lines.append(f"{name} {live[name]:g}")

        return "\n".join(lines) + "\n"


def _header(name: str) -> Iterable[str]:
    kind, help_text = _HELP.get(name, ("untyped", name))
    return (f"# HELP {name} {help_text}", f"# TYPE {name} {kind}")


def _escape(value: str) -> str:
    return value.replace("\\", r"\\").replace('"', r"\"").replace("\n", r"\n")


def _labels(labels: dict[str, str]) -> str:
    return ",".join(f'{k}="{_escape(v)}"' for k, v in sorted(labels.items()))


def _sample(name: str, labels: dict[str, str], value: int) -> str:
    return f"{name}{{{_labels(labels)}}} {value}" if labels else f"{name} {value}"


#: One registry per process. The gateway serves one upstream and one /metrics.
REGISTRY = Metrics()


def record_event(event: str, fields: dict) -> None:
    """Translate one audit event into counters. Never raises.

    Called from `AuditLogger._emit`, which is the single place every audit
    event passes through -- so a new event type becomes visible here without
    anybody remembering to add a counter, and an event that stops being logged
    stops being counted rather than quietly reporting a stale number.

    Wrapped in a bare except on purpose: this is bookkeeping hanging off the
    logging path, and a scrape endpoint is never worth failing a request for.
    """
    try:
        _record(event, fields)
    except Exception:  # noqa: BLE001 - see the docstring
        pass


def _record(event: str, fields: dict) -> None:
    m = REGISTRY
    if event == "request":
        kind = ("summary" if fields.get("is_summary")
                else "agentic" if fields.get("tools_count") else "plain")
        m.inc("llmgw_requests_total", kind=kind)
        state = fields.get("lodestar_state")
        if state is not None:
            m.inc("llmgw_lodestar_requests_total", state=str(state))

    elif event == "response":
        m.inc("llmgw_responses_total",
              finish_reason=str(fields.get("finish_reason") or "none"))
        m.inc("llmgw_tokens_total", int(fields.get("tokens_in") or 0),
              direction="in")
        m.inc("llmgw_tokens_total", int(fields.get("tokens_out") or 0),
              direction="out")
        # Its own counter rather than a third `direction` on the one above:
        # cached tokens are a SUBSET of tokens_in, not a third flow, and a
        # reader summing the directions of one counter should get the total.
        # Only ExLlamaV3 reports it, so this counter describes that engine's
        # turns alone -- which is the comparison worth having anyway, since
        # llama.cpp cannot distinguish a read prompt from a reused one.
        cached = fields.get("cached_tokens")
        if cached is not None:
            m.inc("llmgw_cached_prompt_tokens_total", int(cached))
        m.inc("llmgw_synthesized_calls_total",
              int(fields.get("synthesized_count") or 0))
        for field, metric in (("queue_wait_ms", "llmgw_queue_wait_seconds"),
                              ("ttft_ms", "llmgw_ttft_seconds"),
                              ("duration_ms", "llmgw_request_seconds")):
            value = fields.get(field)
            if value is not None:
                m.observe(metric, float(value) / 1000.0)

    elif event == "error":
        m.inc("llmgw_errors_total", type=str(fields.get("error_type") or "other"))

    elif event == "tool_call":
        # Deliberately NOT labelled by tool name: that vocabulary belongs to
        # the client, and a label the gateway does not control is a series
        # count it cannot bound.
        m.inc("llmgw_tool_calls_total",
              status=str(fields.get("parse_status") or "other"))

    elif event == "spill":
        m.inc("llmgw_spills_total")

    elif event == "prune":
        m.inc("llmgw_prunes_total")

    elif event == "escalation":
        m.inc("llmgw_escalations_total")
