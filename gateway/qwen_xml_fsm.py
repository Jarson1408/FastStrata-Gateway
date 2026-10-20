"""
Qwen XML Finite State Machine parser (Legacy Module & Compatibility Layer).

This module re-exports components from gateway.parsers to maintain full
backward compatibility with existing imports and test suites.
"""
from __future__ import annotations


from gateway.parsers import (
    EventType,
    FsmEvent,
    BaseToolParser,
    Qwen3XmlFSM,
    Qwen25CoderFSM,
    GemmaToolParser,
    create_parser,
    detect_model_family,
)

# Backward-compatible alias
QwenXmlFSM = Qwen3XmlFSM


def collect_events(
    full_content: str,
    finish_reason: str,
    tools: list[dict] | None = None,
    in_triple_backticks: bool = False,
    in_single_backtick: bool = False,
    model_name: str | None = None,
    start_in_thinking: bool = False,
    start_in_tool_call: bool = False,
    family: str | None = None,
) -> tuple[list[FsmEvent], BaseToolParser]:
    """Run the appropriate FSM on complete content. Returns (events, fsm)."""
    fsm = create_parser(
        model_name=model_name,
        tools=tools,
        in_triple_backticks=in_triple_backticks,
        in_single_backtick=in_single_backtick,
        start_in_thinking=start_in_thinking,
        start_in_tool_call=start_in_tool_call,
        family=family,
    )
    events = list(fsm.feed(full_content))
    events.extend(fsm.finish(finish_reason))
    return events, fsm


__all__ = [
    "EventType",
    "FsmEvent",
    "BaseToolParser",
    "QwenXmlFSM",
    "Qwen3XmlFSM",
    "Qwen25CoderFSM",
    "GemmaToolParser",
    "create_parser",
    "detect_model_family",
    "collect_events",
]
