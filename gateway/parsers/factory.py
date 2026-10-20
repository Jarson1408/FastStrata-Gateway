"""Parser factory for resolving the appropriate tool parser based on model family."""
from __future__ import annotations

import logging

from gateway.parsers.base import BaseToolParser
from gateway.parsers.gemma import GemmaToolParser
from gateway.parsers.muse import MuseToolParser
from gateway.parsers.qwen3_xml import Qwen3XmlFSM
from gateway.parsers.qwen25_coder import Qwen25CoderFSM

_log = logging.getLogger("gateway.parsers.factory")

#: Families that have a parser of their own. Anything else falls back to
#: `qwen3`, which is the dialect `inject_tools_to_prompt` asks every model for.
KNOWN_FAMILIES = ("qwen3", "qwen25", "gemma", "muse")


def detect_model_family(model_name: str | None,
                        declared: str | None = None) -> str:
    """Which parser this model needs.

    `declared` comes from the model registry and WINS, because it is a fact
    somebody wrote down rather than an inference from a filename. The sniffing
    below is what runs when nothing was declared -- a preset with no `models`
    list, or a model the client named directly -- and it is worth knowing how
    weak it is: it reads "27b" out of a name and answers `qwen3`, which is
    right for every model on this rig today and silently wrong the first time
    a differently-named one arrives.
    """
    if declared:
        family = declared.strip().lower()
        if family in KNOWN_FAMILIES:
            return family
        _log.warning("Model declares family '%s', which has no parser; "
                     "falling back to the dialect the gateway asks for.", declared)

    if not model_name:
        return "qwen3"
    name = model_name.lower()
    # Gemma first: a name may carry both a family and a size, and the family
    # is the more specific fact.
    if "gemma" in name:
        return "gemma"
    if "muse" in name or "glimmer" in name:
        return "muse"
    # Prioritize Qwen 3.x / 35B / 27B models
    if "3.6" in name or "35b" in name or "27b" in name:
        return "qwen3"
    # Qwen 2.5 or 3B models
    if "2.5" in name or "3b" in name or "qwen2" in name:
        return "qwen25"
    return "qwen3"


def create_parser(
    model_name: str | None = None,
    tools: list[dict] | None = None,
    in_triple_backticks: bool = False,
    in_single_backtick: bool = False,
    start_in_thinking: bool = False,
    start_in_tool_call: bool = False,
    family: str | None = None,
) -> BaseToolParser:
    """Instantiate the appropriate tool calling parser for the given model."""
    resolved = detect_model_family(model_name, declared=family)
    # Spelled out rather than passed as **kwargs: every parser takes the same
    # five arguments, and a dict of them loses its per-key types on the way
    # through, which mypy is right to object to. Three near-identical calls is
    # the cheaper half of that trade.
    if resolved == "qwen25":
        _log.debug("Using Qwen25CoderFSM for model %s", model_name)
        return Qwen25CoderFSM(
            tools=tools,
            in_triple_backticks=in_triple_backticks,
            in_single_backtick=in_single_backtick,
            start_in_thinking=start_in_thinking,
            start_in_tool_call=start_in_tool_call,
        )
    if resolved == "muse":
        _log.debug("Using MuseToolParser for model %s", model_name)
        return MuseToolParser(
            tools=tools,
            in_triple_backticks=in_triple_backticks,
            in_single_backtick=in_single_backtick,
            start_in_thinking=start_in_thinking,
            start_in_tool_call=start_in_tool_call,
        )
    if resolved == "gemma":
        _log.debug("Using GemmaToolParser for model %s", model_name)
        return GemmaToolParser(
            tools=tools,
            in_triple_backticks=in_triple_backticks,
            in_single_backtick=in_single_backtick,
            start_in_thinking=start_in_thinking,
            start_in_tool_call=start_in_tool_call,
        )
    _log.debug("Using Qwen3XmlFSM for model %s", model_name)
    return Qwen3XmlFSM(
        tools=tools,
        in_triple_backticks=in_triple_backticks,
        in_single_backtick=in_single_backtick,
        start_in_thinking=start_in_thinking,
        start_in_tool_call=start_in_tool_call,
    )
