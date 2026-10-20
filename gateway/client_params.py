"""What this gateway does with every OpenAI chat-completions parameter.

`normalizer.build_generate_body` composes llama-server's `/completion` body
from scratch rather than forwarding the client's, which is what lets the
gateway control sampling, cap `n_predict` and render its own ChatML. The cost
is that anything it does not explicitly read is **dropped in silence** -- and
for several parameters silence is worse than an error, because the client
cannot tell "ignored" from "honoured and had no effect".

Measured 2026-09-12 on the request path: `tool_choice`, `response_format`, `n`,
`stop`, `logprobs`, `logit_bias`, `stream_options` and `parallel_tool_calls`
were all read by nothing. A client sending `tool_choice: "none"` got tool calls
anyway; one sending `response_format: json_object` got prose. OpenCode sends
none of them, which is why it never showed.

So every parameter gets a declared disposition, and there are only three:

  HONOURED   the pipeline reads it. The table says where.
  IGNORED    accepted and deliberately without effect. The table says why --
             every one of these is a documented design decision, not an
             oversight, and if a reason cannot be written the parameter does
             not belong here.
  REJECTED   400, naming the parameter. Reserved for the ones where proceeding
             would hand the client an answer to a different question.

Adding a parameter to IGNORED is the dangerous direction. Ask what the client
does with the response when the answer silently ignores what it asked for: if
it crashes, mis-parses or waits forever, the parameter is REJECTED, not
IGNORED.
"""
from __future__ import annotations

from typing import Any

HONOURED: dict[str, str] = {
    "model": "resolved through config.model_aliases",
    "messages": "normalized, then rendered as ChatML",
    "stream": "SSE or a single JSON body",
    "tools": "injected into the system prompt; llama-server's own parser is bypassed",
    "tool_choice": "none/auto/required; see check() for the shapes",
    "response_format": "json_object and json_schema, as the `json_schema` field both engines take",
    "stream_options": "include_usage=false suppresses the usage object; see wants_usage",
    "temperature": "config.get_sampling_params, client override allowed",
    "top_p": "config.get_sampling_params, client override allowed",
    "top_k": "config.get_sampling_params, client override allowed",
    "seed": "config.get_sampling_params, client override allowed",
    "max_tokens": "capped at config.max_n_predict",
    "max_completion_tokens": "capped at config.max_n_predict",
    "repeat_penalty": "passed through when present",
}

IGNORED: dict[str, str] = {
    # These three are gateway-controlled on purpose: adaptive sampling moves
    # temperature and min_p from retry depth and context pressure, and a
    # client that also sets a repetition penalty fights it. config.py has the
    # measurement behind that decision, including why no anti-repetition
    # sampler runs here at all.
    "presence_penalty": "gateway-controlled; see config.get_sampling_params",
    "frequency_penalty": "gateway-controlled; see config.get_sampling_params",
    # Bookkeeping fields with no effect on generation. Ignoring them changes
    # no answer, which is the whole test for belonging in this table.
    "user": "bookkeeping; nothing here is keyed on it",
    "metadata": "bookkeeping; nothing here reads it",
    "store": "no conversation store exists",
    "service_tier": "one upstream, one slot",
}


def _bad(name: str, why: str) -> str:
    return f"{name}: {why}"


def check(body: dict[str, Any], *, can_constrain: bool,
          can_schema: bool | None = None) -> list[str]:
    """Reasons this request cannot be served as asked. Empty means proceed.

    `can_constrain` says whether the upstream accepts llama.cpp's GBNF
    `grammar` field, which is what pins a NAMED `tool_choice` to one function.
    ExLlamaV3 constrains at its own sampler and has no such field.

    `can_schema` is a different question and was conflated with it until
    2026-09-15: a JSON schema goes to llguidance, which the ExLlamaV3 server
    already runs for tool-call syntax, so it can honour `response_format` even
    though it takes no grammar. While the two shared one flag,
    `response_format` was refused for every request this rig serves. It
    defaults to `can_constrain` so a caller that has only asked the old
    question keeps the old, conservative answer.
    """
    if can_schema is None:
        can_schema = can_constrain
    problems: list[str] = []

    n = body.get("n")
    if n is not None and int(n) > 1:
        problems.append(_bad(
            "n", f"asked for {n} choices; this gateway serves one upstream slot "
            "and returns exactly one choice. Send n=1 or omit it."))

    stop = body.get("stop")
    if stop:
        problems.append(_bad(
            "stop", "stop sequences are not applied. llama-server here runs on "
            "native EOS detection and the streaming parser would have to "
            "retract text it has already emitted. Trim client-side."))

    if body.get("logprobs") or body.get("top_logprobs"):
        problems.append(_bad(
            "logprobs", "not returned. The SSE emitter carries content, "
            "reasoning_content and tool calls only."))

    if body.get("logit_bias"):
        problems.append(_bad(
            "logit_bias", "not applied. Sampling here is gateway-controlled "
            "and adaptive; a per-token bias would fight it silently."))

    if body.get("parallel_tool_calls") is False:
        problems.append(_bad(
            "parallel_tool_calls", "cannot be disabled. The parser emits every "
            "call the model makes in a turn, and dropping the extras would "
            "hand back a turn that did not happen."))

    problems.extend(_check_tool_choice(body, can_constrain=can_constrain))
    problems.extend(_check_response_format(body, can_schema=can_schema))
    return problems


def _check_tool_choice(body: dict[str, Any], *, can_constrain: bool) -> list[str]:
    choice = body.get("tool_choice")
    if choice is None:
        return []
    if isinstance(choice, str):
        if choice in ("none", "auto", "required"):
            return []
        return [_bad("tool_choice", f"unknown value {choice!r}; "
                                    "expected none, auto or required")]
    if isinstance(choice, dict):
        name = ((choice.get("function") or {}).get("name")
                if choice.get("type") == "function" else None)
        if not name:
            return [_bad("tool_choice", "expected {'type': 'function', "
                                        "'function': {'name': ...}}")]
        # Forcing a NAMED function needs a grammar. The prefill that serves
        # `required` opens <tool_call> so a call is the only plausible
        # continuation, but it cannot make the model pick `read` out of a list
        # of nine -- and an instruction asking it to is exactly the kind of
        # guidance this model is documented for ignoring.
        from gateway import grammar

        if not can_constrain:
            return [_bad(
                "tool_choice", f"cannot force the function {name!r} on this "
                "upstream: it has no grammar field, and asking the model "
                "nicely is not the promise this parameter makes.")]
        if not grammar.is_pinnable(str(name)):
            return [_bad(
                "tool_choice", f"the function name {name!r} cannot be pinned; "
                "expected letters, digits, underscore or hyphen.")]
        return []
    return [_bad("tool_choice", "expected a string or an object")]


def _check_response_format(body: dict[str, Any], *, can_schema: bool) -> list[str]:
    fmt = body.get("response_format")
    if not isinstance(fmt, dict):
        return [] if fmt is None else [_bad("response_format", "expected an object")]
    kind = fmt.get("type")
    if kind in (None, "text"):
        return []
    if kind not in ("json_object", "json_schema"):
        return [_bad("response_format", f"unknown type {kind!r}")]
    if not can_schema:
        return [_bad(
            "response_format", f"{kind} needs an upstream that can be handed a "
            "schema, and this one cannot. Drop response_format and parse what "
            "comes back.")]
    if kind == "json_schema":
        schema = (fmt.get("json_schema") or {}).get("schema")
        if not isinstance(schema, dict):
            return [_bad("response_format",
                         "json_schema needs json_schema.schema as an object")]
    return []


def json_schema_for(body: dict[str, Any]) -> dict[str, Any] | None:
    """The schema to hand llama-server, or None.

    `json_object` has no schema of its own, so it becomes the schema that
    accepts any object -- which is what the OpenAI field means and what
    llama.cpp needs to build a grammar from.
    """
    fmt = body.get("response_format")
    if not isinstance(fmt, dict):
        return None
    if fmt.get("type") == "json_object":
        return {"type": "object"}
    if fmt.get("type") == "json_schema":
        schema = (fmt.get("json_schema") or {}).get("schema")
        return schema if isinstance(schema, dict) else None
    return None


def tool_choice_mode(body: dict[str, Any]) -> tuple[str, str | None]:
    """(mode, function name). Mode is one of none / auto / required / function."""
    choice = body.get("tool_choice")
    if choice is None:
        return "auto", None
    if isinstance(choice, str):
        return (choice if choice in ("none", "auto", "required") else "auto"), None
    if isinstance(choice, dict) and choice.get("type") == "function":
        name = (choice.get("function") or {}).get("name")
        if name:
            return "function", str(name)
    return "auto", None


def wants_usage(body: dict[str, Any]) -> bool:
    """Should the final streamed chunk carry a usage object?

    A deliberate deviation from the OpenAI shape, and the reason is a client
    that already exists. The spec sends usage only when
    `stream_options.include_usage` is true; this gateway has always sent it
    unconditionally, and OpenCode reads it. Switching to the spec default would
    take token counts away from the one client that uses this thing daily, to
    satisfy a field nobody sets.

    So: an explicit `false` is honoured, and absence keeps the old behaviour.
    That is the half of the parameter where silence would actually mislead --
    a client that asked NOT to be sent usage and is sent it anyway.
    """
    opts = body.get("stream_options")
    if isinstance(opts, dict) and "include_usage" in opts:
        return bool(opts["include_usage"])
    return True
