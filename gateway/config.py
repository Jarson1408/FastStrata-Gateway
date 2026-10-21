"""
Gateway configuration — loaded from environment variables with sane defaults.

Sampling presets allow switching between tuned parameter sets without
code changes.  Individual parameters can still be overridden via env vars.
"""
from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gateway.model_registry import ModelRegistry, ModelSpec

_log = logging.getLogger("gateway.config")

# ---------------------------------------------------------------------------
# Sampling presets
# ---------------------------------------------------------------------------

SAMPLING_PRESETS: dict[str, dict[str, Any]] = {
    # Recommended for coding with tools — low temperature for accurate text
    # reproduction (oldString in edit, file paths, etc.) while min_p allows
    # creative reasoning in <think> blocks where the model is confident.
    "qwen-coder-precise": {
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.05,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        # NO anti-repetition sampler here, and the reasoning is measured.
        #
        # DRY was added 2026-08-31 to stop a <think> loop, on the argument
        # that repetition_penalty punishes individual tokens and so degrades
        # verbatim reproduction of oldString, paths and identifiers, whereas
        # DRY penalises a repeated *sequence*. That got the trade exactly
        # backwards. A file path IS a sequence, and one that must be repeated
        # verbatim from earlier in the context — which is the thing DRY exists
        # to prevent. Within four hours the model was mangling one project
        # path in a new way every turn (PROJEKTY -> PROJEKT, PRO JEKTY,
        # PROJEKTOACME_HW, PROJSKTY), while still writing it correctly in its
        # own reasoning immediately beforehand.
        #
        # Reproduced against the live model, same seed, DRY the only variable:
        #     DRY on  (multiplier 0.8, allowed_length 4)  -> .../PROJEKT/...
        #     DRY off                                     -> .../PROJEKTY/...
        #     DRY on, allowed_length 20                   -> still not verbatim
        #
        # The <think> loop it was meant to solve is already handled, and
        # deterministically, by reasoning_budget_chars. A sampler that trades
        # correct file paths for a redundant loop brake is not a trade worth
        # making. dry_* remain overridable from the environment for anyone who
        # wants to experiment; they are off by default because this preset's
        # whole purpose is reproducing text exactly.
    },
    "qwen-coder-balanced": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.05,
        "presence_penalty": 0.0,
        # See qwen-coder-precise: DRY breaks verbatim path reproduction.
    },
    "qwen-coder-deterministic": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
    },
    "qwen-coder-exploratory": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.05,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
    },
    "qwen-coder-high": {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.05,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
    },
    "legacy": {
        "temperature": 0.5,
        "min_p": 0.05,
        "presence_penalty": 0.0,
    },
}

# ---------------------------------------------------------------------------
# Model aliases  (alias → the model id the registry serves)
# ---------------------------------------------------------------------------
# OpenCode hardcodes temperature=0.55 for any model whose id contains "qwen".
# Aliases without "qwen" bypass that branch.

DEFAULT_MODEL_ALIASES: dict[str, str] = {
    "qwen3.6-35b-a3b-ud-q4_k_m": "qwen3.6-35b-a3b",
}


# ---------------------------------------------------------------------------
# Main config dataclass
# ---------------------------------------------------------------------------

# FastStrata's server, told apart by its port: the address the gateway starts
# the engine on, and the one predicate the lifespan, /health and the stream
# connect path ask, so a fourth caller cannot invent a fourth spelling.
POLYSTRATA_PORT = 8080


def is_polystrata_upstream(url: object) -> bool:
    return f":{POLYSTRATA_PORT}" in str(url or "")


def upstream_can_constrain(url: object) -> bool:
    """True when the upstream accepts llama.cpp's GBNF `grammar` field.

    That is what pins a named `tool_choice` to one function, which a prefill
    cannot do. FastStrata's `/completion` takes the llama.cpp body, and so does
    a remote llama.cpp server.
    """
    return bool(url)


def upstream_takes_json_schema(url: object) -> bool:
    """True when the upstream can be handed a JSON schema for the whole answer.

    A separate question from the one above: GBNF is a request FIELD llama.cpp
    parses, a JSON schema is the `json_schema` field. Kept apart so that an
    engine taking one and not the other changes one predicate, not both.
    """
    return bool(url)


#: Characters per token, MEASURED on 2 590 real reasoning blocks against the
#: upstream's own tokenizer (56 285 characters = 15 947 tokens), not assumed.
CHARS_PER_TOKEN = 3.13

#: What share of the output ceiling the reasoning budget may occupy. 40 000
#: chars against a 16 384-token ceiling is 78%, which is where the measured
#: default sits; keeping the RATIO rather than the number is what makes the
#: budget-before-ceiling ordering survive a model with a different ceiling.
REASONING_BUDGET_RATIO = 0.78


def derive_reasoning_budget(max_n_predict: int) -> int:
    """The reasoning budget that keeps its documented place below the ceiling.

    The two limits fail differently and that is the whole point of the
    ordering: the budget abandons the generation and retries with `</think>`
    closed, and the client never sees a seam, while `n_predict` ends the turn
    `length` with no answer, no tool call and no retry. Whichever fires first
    decides which failure the user gets.
    """
    return max(1000, int(max_n_predict * CHARS_PER_TOKEN * REASONING_BUDGET_RATIO))


def _same_endpoint(a: object, b: object) -> bool:
    """Whether two upstream URLs name the same server.

    Compared without the trailing slash and case-insensitively, which is all
    the variation these strings actually carry -- they come from settings.json
    and from this file's own defaults, never from a client.
    """
    return str(a or "").rstrip("/").lower() == str(b or "").rstrip("/").lower()


#: Families whose `/completion` prompt is NOT ChatML. Anything absent from
#: here -- including an undeclared family -- renders as ChatML, which is what
#: this gateway did for every model before Gemma 4 arrived.
_PROMPT_FORMATS = {"gemma": "gemma4", "muse": "muse"}


def prompt_format_for_family(family: str | None) -> str:
    """The chat format a model of this family is prompted in."""
    return _PROMPT_FORMATS.get((family or "").strip().lower(), "chatml")


#: The tool dialect a family is trained on, where the gateway speaks it.
_NATIVE_TOOL_DIALECTS = {"gemma": "gemma4", "muse": "muse"}
TOOL_DIALECTS = ("qwen_xml", "gemma4", "muse")


def tool_dialect_for(family: str | None, declared: str | None = None) -> str:
    """The tool dialect a model uses: its declaration, else its family's.

    A declared native dialect that does not belong to the model's own prompt
    format falls back to "qwen_xml" -- the Gemma protocol written into a
    ChatML prompt would be markup the model never saw.
    """
    fam = (family or "").strip().lower()
    native = _NATIVE_TOOL_DIALECTS.get(fam, "qwen_xml")
    want = (declared or "").strip().lower()
    if want == "qwen_xml":
        return "qwen_xml"
    if want and want != native:
        _log.warning("Model declares tool_dialect '%s', which its family '%s' "
                     "cannot speak; using '%s'.", want, fam or "?", native)
    return native


def _default_lodestar_context_max_chars() -> int:
    """The one definition of this default lives with the code that applies it.

    Imported inside the function: config.py is imported very early and by
    everything, and the injector pulls in the prompt loader.
    """
    from gateway.lodestar.context_injector import DEFAULT_CONTEXT_MAX_CHARS

    return DEFAULT_CONTEXT_MAX_CHARS


@dataclass
class GatewayConfig:
    """Centralised configuration for the gateway."""

    # Network
    host: str = "127.0.0.1"
    port: int = 18000
    upstream: str = "http://127.0.0.1:8080"
    auth_token: str | None = None

    # Paths the catch-all proxy route may forward upstream. Everything else
    # gets a 404. Before this list existed the catch-all accepted ANY path,
    # which meant POST /completion reached llama-server without a token and
    # without the LODESTAR/FSM pipeline — and with BIND_ALL=1 that was an open
    # proxy on the LAN. Observed traffic only ever used /v1/chat/completions,
    # /v1/embeddings and /health, so this list is deliberately narrow.
    passthrough_allowlist: tuple[str, ...] = ("/props", "/tokenize", "/v1/models")

    # Sampling
    sampling_preset: str = "qwen-coder-precise"

    # Context management
    context_reserve: int = 16384       # tokens reserved for the response

    # Byte size past which a tool result is written to a file and replaced in
    # the prompt by a preview plus its path. Pruning works on whole
    # conversational blocks, so without this one `pytest` run can evict the
    # user's original request. 0 disables spilling.
    spill_max_inline_bytes: int = 8192

    # How long a spilled tool result stays on disk. These are literal copies of
    # files from every project the gateway serves — 91 files across three
    # unrelated projects on the day of the 2026-09-03 audit — and until then
    # nothing ever deleted one, nor said anywhere that the directory existed.
    # Cleaned at startup only: this gateway restarts after every code change
    # (115 starts in gateway.log), so a timer would buy nothing and cost a
    # coroutine to cancel on shutdown. 0 keeps everything, for anyone who
    # wants the directory to be an archive. G-2026-01.
    spill_retention_days: int = 30

    # Leave a factual note in place of evicted turns, so the model can see it
    # already tried something instead of trying it again.
    compaction_enabled: bool = True

    # What to do with a model-issued call recognised as destructive
    # (`rm -rf`, `git reset --hard`, a write outside the workspace…):
    # "off", "warn" (log it, let it through) or "confirm" (block it and tell
    # the model to ask the user first). Default "warn": the gateway executes
    # nothing itself, so this is a tripwire, not confinement, and silently
    # changing what reaches the client would break existing sessions.
    destructive_action: str = "warn"

    # Tiered routing. When the model has repeated the same failing action
    # `escalation_retry_depth` times, adaptive sampling has already tried
    # both greedy and exploratory temperatures; if the model is simply too
    # small for the step, temperature will not save it.
    #
    # Off unless `escalation_model` is set, and it only means anything with a
    # second backend: one llama-server holds one model in VRAM, so switching
    # models without `escalation_upstream` would reload weights mid-task.
    # Point `escalation_upstream` at a cloud endpoint only on purpose — that
    # is how source code leaves the machine by accident.
    escalation_model: str | None = None
    escalation_upstream: str | None = None
    escalation_retry_depth: int = 4
    n_ctx: int | None = None           # auto-discovered from upstream /props
    fallback_n_ctx: int = 32768        # assumed context window when upstream cannot be asked
    default_n_predict: int = 8192
    # Hard ceiling on n_predict, whatever the client asks for. OpenCode sends
    # max_tokens=32000; at ~12 tok/s that is 45 minutes of single-slot time
    # that a runaway turn can burn before llama-server stops it on its own.
    # A legitimate turn in an agentic loop is a few hundred tokens; 16k still
    # allows writing a very large file in one go.
    #: Pixels above which an incoming image is downscaled before anything
    #: else in the pipeline sees it. 0 disables.
    #:
    #: **A property of the CARD, not of the model**, which is why it is not in
    #: `ModelSpec`: what it protects is the vision tower's activation peak, and
    #: that is bounded by the GPU the tower runs on.
    #:
    #: 8 294 400 is 3840x2160, the largest image measured to fit. The
    #: preprocessor accepts up to **16 384 image tokens** (16 777 216 px,
    #: `pixels / 1024` exactly), and measured 2026-09-14 the peak grows 97 KiB
    #: per image token: a 1440p screenshot leaves 1 313 MiB free at a full
    #: 196 608 context, 4K leaves 879, and the 16 384-token ceiling is about
    #: 1 217 MiB above the 1440p case -- which nothing on this card can pay.
    #: Nobody reads a cursor readout better at sixteen megapixels, so the cap
    #: costs nothing and closes the one remaining route to an out-of-memory in
    #: whichever process asks for the next allocation.
    image_max_pixels: int = 8_294_400
    max_n_predict: int = 16384
    inject_context: bool = True        # Auto-inject project rules and status context files
    lodestar_prefill: bool = True      # Prefill assistant <think> with the next required LODESTAR action (states 1-2)
    lodestar_enabled: bool = True      # Global toggle for LODESTAR protocol gating/injection

    # FastStrata auto-start / auto-stop. `polystrata_dir` empty means the
    # folder this gateway was installed into (`model_registry.
    # default_polystrata_dir`); `polystrata_home()` is the one reader.
    polystrata_dir: str = ""
    polystrata_config: str = "strata-qwen36-ud-q4_k_m.json"
    polystrata_autostart: bool = True
    polystrata_autostop: bool = True
    polystrata_restart_cooldown: float = 30.0

    # -- Several models, one card -------------------------------------------
    #
    # `models` is the preset's registry: every model this rig serves, the
    # engine that serves it, and the settings that are its own. Empty is the
    # normal state for a preset with a single upstream, and an empty registry
    # changes nothing — the alias path and `upstream` work exactly as before.
    # gateway/model_registry.py is what reads this; keeping it as plain dicts
    # here means settings.json stays the one format the GUI edits.
    models: list[dict[str, Any]] = field(default_factory=list)

    #: What one image is assumed to cost when pruning decides what fits.
    #:
    #: The pruner counts tokens by rendering a message to ChatML and asking
    #: /tokenize, and after normalization an image is the 13-character string
    #: `<__media__>`. So without this an image counts as four tokens and
    #: really costs hundreds to thousands — the exact failure mode the prune
    #: limit exists to prevent, in the one place nothing would look. The
    #: number matches llama.cpp's own `--mtmd-batch-max-tokens` default,
    #: which is the largest batch it will encode an image in; over-charging
    #: costs a turn of history, under-charging costs the request.
    image_tokens_estimate: int = 1024

    # Streaming
    # 2 hours safety valve, aligned with semaphore_timeout.
    #
    # 0 DISABLES it, like every other limit in this file. Until 2026-09-10 it
    # was the one exception: the check was a bare `elapsed > limit`, so a
    # configured 0 meant "already exceeded" and killed every stream on the
    # first loop iteration, ~100 ms in. The only value that reads as "turn
    # this off" was the one that turned the gateway off.
    stream_max_duration: int = 7200
    heartbeat_interval: float = 10.0

    # Reasoning budget — the brake on a <think> loop.
    #
    # stream_max_duration is 2 hours and n_predict is capped in the tens of
    # thousands, so neither notices a model that has been talking to itself
    # for ten minutes. Measured on a healthy agentic turn: 335-515 tokens of
    # total output. The observed loop (2026-08-31, ACME_HW) passed 10,000
    # tokens inside a single <think> and was still going.
    #
    # Counted in characters rather than tokens because the gateway sees text,
    # not tokens. Measured 2026-09-12 with the upstream's own tokenizer over
    # the six longest real blocks: 3.13 chars/token (56,285 chars = 15,947
    # tokens). 0 disables the guard.
    #
    # **Raised from 24000 on 2026-09-12, because the guard was censoring its
    # own evidence.** After it landed, every stored reasoning block above
    # 24,000 characters sat at 24,00x -- its own cuts, ten of them, 1.1% of
    # blocks -- so nothing could be learned from them. The distribution from
    # BEFORE it landed is uncensored: 1,701 blocks, ten of them over 24,000
    # and up to 56,285. Every one of those ten was checked with this project's
    # own detectors and came back clean: **zero loops, zero code-in-reasoning.**
    # Audit enumeration, test analysis, report planning, each ending on a
    # decision to act. The guard was cutting genuine work at 47% of what
    # n_predict allows.
    #
    # 40000 = 12,780 tokens = 78% of max_n_predict, and that ratio is the
    # point rather than the number. The budget must fire BEFORE n_predict,
    # because the two failures are not alike: this one abandons the generation
    # and retries with `</think>` closed, and the client never sees a seam,
    # while n_predict ends the turn `length` with no answer, no tool call and
    # no retry. TestTheBudgetFiresBeforeTheCeiling holds the ordering.
    #
    # It is not free, and the honest number is worth keeping: cutting at 24000
    # took 5.9 minutes plus a short retry, about 6.8 minutes in total, while
    # letting the same turn finish takes 7.1-11.5. **You pay up to five
    # minutes on 0.65% of turns and buy an answer written from complete
    # reasoning instead of reasoning truncated at 47%.**
    reasoning_budget_chars: int = 40000

    # The turn that ends with the answer stuck in <think>.
    #
    # A subagent is rescued by subagent_min_report_chars, because for a
    # subagent a short last turn is ALWAYS a failure: the client ends it the
    # moment a turn carries no tool call, and whatever text is there becomes
    # the report its parent reads. In a main session a short answer is often
    # correct -- "Gotowe." after a moment's thought is exactly right -- so the
    # same rule there would nag constantly.
    #
    # Measured 2026-09-12 over 736 real turns that ended with no tool call:
    # 301 of them had under 500 characters of reasoning AND under 200 of
    # content, and every one was a legitimate acknowledgement. The failure
    # shape is the CONJUNCTION -- substantial reasoning and almost no content
    # -- and it isolates 11 turns, 1.5%, of which 8 had no content at all.
    #
    # So the trigger is reasoning >= this AND content < subagent_min_report_chars.
    # 0 disables the rescue.
    answer_rescue_min_reasoning: int = 2000

    # How many times one turn may be asked to write its answer out. Bounded
    # for the same reason as reasoning_retry_max: a model that ignores the ask
    # twice will not comply on the third, and past the limit the older
    # behaviour -- extract the conclusion from the reasoning -- takes over.
    answer_rescue_max: int = 1

    # The same loop, caught by its shape instead of its size: the share of
    # DISTINCT lines in the recent tail of the reasoning. Counting repeats was
    # the first attempt and was wrong — measured on the real block, three
    # identical lines arrived while the model was doing genuine analysis and
    # merely restating its intention. What separates that from a loop is
    # whether anything new appears between the repeats. Substantive work
    # measured 0.90-1.00 distinct across nineteen thousand characters; the
    # loop reached 0.30. 0 disables.
    reasoning_loop_max_distinct: float = 0.6

    # Consecutive lines of source code inside <think> before the turn is cut
    # and the model is told to emit the call instead. The most expensive
    # failure measured so far: eleven thousand characters of finished Python
    # written into a reasoning block and discarded when the turn ended without
    # a write. Separable from a sketched formula by a wide margin — measured on
    # that block, engineering analysis never produced a run longer than 2 lines
    # and the calculator produced 234. 0 disables.
    reasoning_code_lines: int = 25

    # The same measure as `reasoning_loop_max_distinct`, applied to CONTENT --
    # the half of a turn no guard could see until 2026-09-14. All three
    # reasoning guards are conjunctive on `in_thinking`, and every continuation
    # builds its parser with `start_in_thinking=False`, so on the path where a
    # model has just been asked to produce the answer it failed to produce,
    # `RunawayWatch` was called on every tick and could not fire once.
    #
    # Measured that day against the turn it was written for and against every
    # text block the same session produced:
    #
    #     the loop (11 reads, 11 edits, one sentence over and over)   0.571
    #     27 real content blocks, lowest of all of them               0.938
    #
    # That gap is wider than the 0.30-vs-0.90 one the reasoning threshold was
    # tuned in, and **0.6 falls inside both** -- so this is a second number
    # that happens to coincide with the first, exactly like
    # `spill_max_inline_bytes` and `fallback_n_ctx` landing on 65536 for
    # unrelated reasons. It is its own knob so it can be disabled on its own;
    # do not fold the two together. 0 disables.
    content_loop_max_distinct: float = 0.6

    # A hard stop on how many tool calls ONE turn may execute. Not a loop
    # detector -- the one above catches a loop that talks, and this catches the
    # one that does not: a model hammering the filesystem in silence.
    #
    # Measured 2026-09-14 over 7 318 turns: p50 0, p90 1, p99 6, p99.9 22.
    # Above 25 there are four turns in the whole log -- 34, 36, 60 and 140 --
    # holding the single slot for 2.4 to 14.3 minutes apiece. The 21-24 band is
    # nine turns of real batch editing and is deliberately left alone. So this
    # touches one turn in 1 829, and every one it touches was pathological.
    # 0 disables.
    max_tool_calls_per_turn: int = 25

    # How many times the gateway may abandon a looping generation and start
    # another within the SAME client request. The client sees one continuous
    # response; the user is not asked to say "continue" for a problem the
    # gateway diagnosed itself. The retry closes the </think> block in the
    # prompt, so the model physically cannot resume deliberating — which is
    # the only reliable way to stop a loop that has already ignored the text
    # telling it to stop. 0 restores the old behaviour: cut, explain, end.
    reasoning_retry_max: int = 1

    # How many times the gateway may automatically and invisibly continue a
    # generation that was cut off mid-content by token/context limit
    # (finish_reason == "length"). Prompt history is compacted on the fly
    # to free up KV-cache headroom while preserving emitted content verbatim.
    # The client sees a seamless, uninterrupted response. 0 disables.
    content_continue_max: int = 2

    # How many times the gateway may automatically attempt in-stream recovery
    # when loop_breaker suppresses repeating tool calls in a turn with 0 remaining
    # calls. Closes <think> with directive guidance to force an alternative action.
    # 0 disables loop recovery (terminates turn with guard notice).
    loop_recovery_max: int = 1

    # How long to let the upstream reader shut itself down cleanly before the
    # connection is severed by force. llama-server only notices a dropped
    # client when it next writes an SSE chunk, so a clean close is what makes
    # it abort the slot instead of generating into a socket nobody reads.
    upstream_close_grace: float = 2.0

    # Agent-loop limits. The agent loop runs in the client, so the gateway
    # cannot count iterations directly — but it can measure the current
    # LODESTAR cycle from the history it receives. Past these limits it stops
    # acting for the model and says so, instead of quietly burning the single
    # upstream slot for hours (audit finding C2, open since 2026-06-18).
    max_cycle_turns: int = 60          # assistant turns in one cycle
    # Until 2026-09-01 max_cycle_turns only wrote a log line — a configured
    # limit that stopped nothing, which is worse than no limit because it
    # reads as cover. Past the budget the model is told to close out; past
    # budget + grace its tool calls stop being executed. Recoverable by the
    # user: their next message opens a new cycle and resets the count.
    cycle_turn_grace: int = 20
    max_synthesis_per_cycle: int = 5   # calls the gateway makes on its behalf

    # How much of active-context.md the volatile LODESTAR frame may carry.
    # That frame is re-prefilled on every turn by construction, so its size is
    # a tax the whole conversation pays; active-context.md is the only part of
    # it a project can grow without bound. The reasoning behind the number is
    # in gateway/lodestar/context_injector.py, which owns the default — there
    # is one definition of it, imported here rather than repeated. 0 disables.
    lodestar_context_max_chars: int = field(
        default_factory=lambda: _default_lodestar_context_max_chars())

    # Backstop for the above. The per-cycle count is read from the history the
    # client sends, and OpenCode compacts long conversations — synthesized
    # calls vanish with the summarized turns and the budget silently refills.
    # A rolling per-workspace rate is keyed on wall-clock time instead, which
    # the client cannot rewrite. Generous on purpose: a healthy cycle needs at
    # most three synthesized calls, so this allows several heavily-assisted
    # tasks an hour while still stopping a loop.
    max_synthesis_per_hour: int = 20
    synthesis_window_seconds: float = 3600.0

    # Subagents skip LODESTAR entirely, so every limit above — all of them
    # derived from cycle state — leaves them completely unbounded. Measured
    # 2026-09-01: one ran 51 tool calls over 2h52m on a single-slot rig, for a
    # task whose output is a paragraph. Past the budget it is asked to report;
    # past budget + grace its tool calls stop being executed. 0 disables.
    max_subagent_turns: int = 25
    subagent_turn_grace: int = 10
    # A subagent's last turn IS its whole deliverable: the client ends the
    # subagent as soon as a turn carries no tool call, and whatever text is in
    # that turn becomes the task_result its parent receives. Below this many
    # characters it is not a report, and the gateway completes it from the
    # reasoning. Measured 2026-09-01: two subagents returned 80 and 150
    # characters of "Let me now read ..." while their <think> held the actual
    # findings. 0 disables.
    subagent_min_report_chars: int = 400

    # Semaphore wait
    semaphore_timeout: float = 7200.0  # 2 hours (aligned with stream_max_duration)
    semaphore_keepalive_interval: float = 15.0
    concurrency_limit: int = 1

    # Logging
    log_level: str = "INFO"
    log_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # Whether the audit log keeps the full rendered prompt of every request.
    # The digest, the injected-block sizes and the sampling parameters are
    # always logged and are enough to tell two requests apart; the prompt text
    # itself contains the user's source code, so writing it to disk is an
    # explicit choice rather than a default.
    audit_full_prompts: bool = False

    #: Sampling parameters the ACTIVE model declares, applied over its preset.
    #: Populated by `for_spec` per request and empty otherwise. It is a field
    #: rather than an argument threaded through `get_sampling_params` because
    #: every caller of that method already holds a config, and a new argument
    #: would have to be remembered at each of them -- which this file keeps
    #: recording as the way a rule ends up present in three places out of four.
    model_sampling: dict[str, Any] = field(default_factory=dict)

    #: The lowest temperature adaptive sampling may move the ACTIVE model to;
    #: 0.0 means no floor. Set per model by `for_spec`. A thinking model is the
    #: case: Qwen's own model cards warn that greedy decoding sends it into
    #: endless repetition, and the gateway's precision rules (TOOL_FAILURES,
    #: TOOL_HEAVY, greedy at retry depth 2) all push the other way. Raising the
    #: temperature to break a loop is untouched -- this is a floor, not a pin.
    temperature_floor: float = 0.0

    #: Which chat format the ACTIVE model's `/completion` prompt is rendered
    #: in. "chatml" is Qwen's and every model this gateway served until
    #: 2026-10-15; "gemma4" is `<|turn>role ... <turn|>`. A property of the
    #: model FAMILY, set by `for_spec`, for the same reason `model_sampling`
    #: is a field: the renderer and the prompt-surgery gate both read it from
    #: the config they already hold.
    prompt_format: str = "chatml"
    #: The active model's `thinking` switch (see ModelSpec.thinking); only a
    #: format with a switch reads it.
    thinking: bool = False
    #: How the ACTIVE model is told about tools and how its calls are read:
    #: "qwen_xml" (the injected Qwen3-Coder dialect, every model until
    #: 2026-10-15) or "gemma4" (Gemma 4's native protocol). Set by `for_spec`.
    tool_dialect: str = "qwen_xml"
    #: The model `/props` named alongside the discovered `n_ctx`, recorded by
    #: `adopt_upstream_n_ctx`; "" until a /health poll has said. See `_n_ctx_for`.
    n_ctx_model: str = ""
    #: The active model's reasoning strength, for a family whose format takes
    #: one (Muse Glimmer: low/medium/high/xhigh, in the system footer). Set by
    #: `for_spec`; "" means the family's own default.
    reasoning: str = ""

    # Model aliases
    model_aliases: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODEL_ALIASES))
    _reverse_aliases: dict[str, str] = field(default_factory=dict, repr=False)

    #: Cache for `registry()`, with the list it was built from beside it so a
    #: caller that replaced `models` gets a registry that agrees with it.
    _registry: "ModelRegistry | None" = field(default=None, repr=False, compare=False)
    _registry_src: "list[dict[str, Any]] | None" = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._reverse_aliases = {v: k for k, v in self.model_aliases.items()}

    # -- Factory ---------------------------------------------------------------

    @classmethod
    def from_env(cls, config_path: Path | None = None) -> GatewayConfig:
        """Build configuration from environment variables and settings.json."""
        from gateway.config_file import active_models, load_settings_from_file
        file_settings = load_settings_from_file(config_path)

        # Environment variables override file settings, which override defaults
        upstream_default = file_settings.get("upstream", "http://127.0.0.1:8080")
        port_default = file_settings.get("port", 18000)
        reserve_default = file_settings.get("context_reserve", 16384)
        lodestar_enabled_default = file_settings.get("lodestar_enabled", False)
        model_aliases_default = file_settings.get("model_aliases", dict(DEFAULT_MODEL_ALIASES))

        allowlist_default = file_settings.get(
            "passthrough_allowlist", ["/props", "/tokenize", "/v1/models"])
        allowlist_env = os.environ.get("LOCAL_LLM_GATEWAY_PASSTHROUGH_ALLOWLIST")
        if allowlist_env is not None:
            allowlist_default = [p.strip() for p in allowlist_env.split(",") if p.strip()]
        allowlist = tuple(
            p if p.startswith("/") else f"/{p}" for p in allowlist_default)

        # The ceiling is resolved BEFORE the constructor call, because the
        # reasoning budget is derived from it and a hardcoded fallback here is
        # exactly how the documented raise never reached a running gateway.
        # The field default moved 24000 -> 40000 on 2026-09-12 with a section
        # of AGENTS.md arguing it from measurements; this fallback kept the old
        # number, settings.json carried no override, and `from_env` is the
        # constructor every real start uses. So every budget trip in
        # gateway.log after that change still reads `limit 24000` -- including
        # one on 2026-09-14 that cut a seven-image analysis at 24,002. The
        # tests could not see it: they all build `GatewayConfig(...)` directly,
        # which is the constructor nobody runs.
        #
        # Derived rather than respelled, so the two cannot drift again: the
        # RATIO is the invariant, not the number.
        max_n_predict_default = int(os.environ.get(
            "LOCAL_LLM_GATEWAY_MAX_N_PREDICT",
            str(file_settings.get("max_n_predict", 16384))))
        reasoning_budget_default = file_settings.get(
            "reasoning_budget_chars",
            derive_reasoning_budget(max_n_predict_default))

        cfg = cls(
            host=os.environ.get("LOCAL_LLM_GATEWAY_HOST", "127.0.0.1"),
            port=int(os.environ.get("LOCAL_LLM_GATEWAY_PORT", str(port_default))),
            upstream=os.environ.get("LOCAL_LLM_GATEWAY_UPSTREAM", upstream_default),
            auth_token=os.environ.get("LOCAL_LLM_GATEWAY_TOKEN") or None,
            sampling_preset=os.environ.get("LOCAL_LLM_GATEWAY_SAMPLING_PRESET", "qwen-coder-precise"),
            context_reserve=int(os.environ.get("LOCAL_LLM_GATEWAY_CONTEXT_RESERVE", str(reserve_default))),
            fallback_n_ctx=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_FALLBACK_N_CTX",
                str(file_settings.get("fallback_n_ctx", 32768)))),
            default_n_predict=int(os.environ.get("LOCAL_LLM_GATEWAY_N_PREDICT", "8192")),
            max_n_predict=max_n_predict_default,
            image_max_pixels=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_IMAGE_MAX_PIXELS",
                str(file_settings.get("image_max_pixels", 8_294_400)))),
            reasoning_retry_max=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_REASONING_RETRY_MAX",
                str(file_settings.get("reasoning_retry_max", 1)))),
            content_continue_max=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_CONTENT_CONTINUE_MAX",
                str(file_settings.get("content_continue_max", 2)))),
            loop_recovery_max=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_LOOP_RECOVERY_MAX",
                str(file_settings.get("loop_recovery_max", 1)))),
            reasoning_code_lines=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_REASONING_CODE_LINES",
                str(file_settings.get("reasoning_code_lines", 25)))),
            reasoning_loop_max_distinct=float(os.environ.get(
                "LOCAL_LLM_GATEWAY_REASONING_LOOP_MAX_DISTINCT",
                str(file_settings.get("reasoning_loop_max_distinct", 0.6)))),
            content_loop_max_distinct=float(os.environ.get(
                "LOCAL_LLM_GATEWAY_CONTENT_LOOP_MAX_DISTINCT",
                str(file_settings.get("content_loop_max_distinct", 0.6)))),
            max_tool_calls_per_turn=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_MAX_TOOL_CALLS_PER_TURN",
                str(file_settings.get("max_tool_calls_per_turn", 25)))),
            reasoning_budget_chars=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_REASONING_BUDGET_CHARS",
                str(reasoning_budget_default))),
            upstream_close_grace=float(os.environ.get(
                "LOCAL_LLM_GATEWAY_UPSTREAM_CLOSE_GRACE",
                str(file_settings.get("upstream_close_grace", 2.0)))),
            spill_max_inline_bytes=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_SPILL_MAX_INLINE_BYTES",
                str(file_settings.get("spill_max_inline_bytes", 8192)))),
            spill_retention_days=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_SPILL_RETENTION_DAYS",
                str(file_settings.get("spill_retention_days", 30)))),
            compaction_enabled=os.environ.get(
                "LOCAL_LLM_GATEWAY_COMPACTION",
                "1" if file_settings.get("compaction_enabled", True) else "0",
            ).strip() == "1",
            destructive_action=os.environ.get(
                "LOCAL_LLM_GATEWAY_DESTRUCTIVE_ACTION",
                str(file_settings.get("destructive_action", "warn"))).strip().lower(),
            escalation_model=(os.environ.get(
                "LOCAL_LLM_GATEWAY_ESCALATION_MODEL",
                file_settings.get("escalation_model") or "") or None),
            escalation_upstream=(os.environ.get(
                "LOCAL_LLM_GATEWAY_ESCALATION_UPSTREAM",
                file_settings.get("escalation_upstream") or "") or None),
            escalation_retry_depth=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_ESCALATION_RETRY_DEPTH",
                str(file_settings.get("escalation_retry_depth", 4)))),
            inject_context=os.environ.get("LOCAL_LLM_GATEWAY_INJECT_CONTEXT", "1").strip() == "1",
            lodestar_prefill=os.environ.get("LOCAL_LLM_GATEWAY_PREFILL", "1").strip() == "1",
            stream_max_duration=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_STREAM_MAX_DURATION",
                str(file_settings.get("stream_max_duration", 7200)))),
            polystrata_dir=os.environ.get(
                "LOCAL_LLM_GATEWAY_POLYSTRATA_DIR",
                str(file_settings.get("polystrata_dir") or "")),
            polystrata_config=os.environ.get(
                "LOCAL_LLM_GATEWAY_POLYSTRATA_CONFIG",
                str(file_settings.get("polystrata_config", "strata-qwen36-ud-q4_k_m.json"))),
            polystrata_autostart=os.environ.get(
                "LOCAL_LLM_GATEWAY_POLYSTRATA_AUTOSTART",
                "1" if file_settings.get("polystrata_autostart", True) else "0",
            ).strip() == "1",
            polystrata_autostop=os.environ.get(
                "LOCAL_LLM_GATEWAY_POLYSTRATA_AUTOSTOP",
                "1" if file_settings.get("polystrata_autostop", True) else "0",
            ).strip() == "1",
            polystrata_restart_cooldown=float(os.environ.get(
                "LOCAL_LLM_GATEWAY_POLYSTRATA_RESTART_COOLDOWN",
                str(file_settings.get("polystrata_restart_cooldown", 30.0)))),
            # Through the shared resolver, not straight off the top level:
            # a settings.json written before the registry existed has no
            # `models` key at all, and reading only the top level gave a
            # freshly restarted gateway an EMPTY registry while the dashboard
            # showed four models. See `active_models`.
            models=active_models(config_path),
            image_tokens_estimate=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_IMAGE_TOKENS_ESTIMATE",
                str(file_settings.get("image_tokens_estimate", 1024)))),
            heartbeat_interval=float(os.environ.get("LOCAL_LLM_GATEWAY_HEARTBEAT_INTERVAL", "10.0")),
            semaphore_timeout=float(os.environ.get(
                "LOCAL_LLM_GATEWAY_SEMAPHORE_TIMEOUT",
                str(file_settings.get("semaphore_timeout", 7200.0)))),
            semaphore_keepalive_interval=float(os.environ.get("LOCAL_LLM_GATEWAY_SEMAPHORE_KEEPALIVE_INTERVAL", "15.0")),
            concurrency_limit=int(os.environ.get("LOCAL_LLM_GATEWAY_CONCURRENCY", "1")),
            max_cycle_turns=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_MAX_CYCLE_TURNS",
                str(file_settings.get("max_cycle_turns", 60)))),
            max_synthesis_per_cycle=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_MAX_SYNTHESIS_PER_CYCLE",
                str(file_settings.get("max_synthesis_per_cycle", 5)))),
            lodestar_context_max_chars=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_LODESTAR_CONTEXT_MAX_CHARS",
                str(file_settings.get("lodestar_context_max_chars",
                                      _default_lodestar_context_max_chars())))),
            cycle_turn_grace=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_CYCLE_TURN_GRACE",
                str(file_settings.get("cycle_turn_grace", 20)))),
            subagent_min_report_chars=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_SUBAGENT_MIN_REPORT_CHARS",
                str(file_settings.get("subagent_min_report_chars", 400)))),
            max_subagent_turns=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_MAX_SUBAGENT_TURNS",
                str(file_settings.get("max_subagent_turns", 25)))),
            subagent_turn_grace=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_SUBAGENT_TURN_GRACE",
                str(file_settings.get("subagent_turn_grace", 10)))),
            max_synthesis_per_hour=int(os.environ.get(
                "LOCAL_LLM_GATEWAY_MAX_SYNTHESIS_PER_HOUR",
                str(file_settings.get("max_synthesis_per_hour", 20)))),
            synthesis_window_seconds=float(os.environ.get(
                "LOCAL_LLM_GATEWAY_SYNTHESIS_WINDOW_SECONDS",
                str(file_settings.get("synthesis_window_seconds", 3600.0)))),
            log_level=os.environ.get("LOCAL_LLM_GATEWAY_LOG_LEVEL", "INFO"),
            audit_full_prompts=os.environ.get(
                "LOCAL_LLM_GATEWAY_AUDIT_FULL_PROMPTS",
                "1" if file_settings.get("audit_full_prompts", False) else "0",
            ).strip() == "1",
            lodestar_enabled=lodestar_enabled_default,
            model_aliases=model_aliases_default,
            passthrough_allowlist=allowlist,
        )
        # Bind-all override (backwards compat with v1 default of 0.0.0.0)
        if os.environ.get("LOCAL_LLM_GATEWAY_BIND_ALL", "").strip() == "1":
            cfg.host = "0.0.0.0"
        return cfg

    # -- Helpers ---------------------------------------------------------------

    def resolve_model(self, name: str) -> str:
        """Translate an alias to the real model name. Pass-through if not aliased."""
        return self.model_aliases.get(name, name)

    def polystrata_home(self) -> str:
        """The FastStrata folder: `polystrata_dir` from settings, else the one
        this gateway was installed into (`model_registry.default_polystrata_dir`)."""
        from gateway.model_registry import default_polystrata_dir
        return self.polystrata_dir or default_polystrata_dir()

    def for_spec(self, spec: "ModelSpec | None") -> GatewayConfig:
        """This configuration as it applies to ONE model.

        Returns a per-request COPY with the model's own limits applied, or
        `self` when the model overrides nothing — so a rig whose models all
        have the same shape allocates nothing and behaves exactly as before.

        **This exists because a shared prune limit was a correctness hole, not
        an inconvenience.** `get_prune_limit()` read one process-wide `n_ctx`,
        discovered from `config.upstream` and re-adopted by /health from the
        same address, and knew nothing about where a given request was going.
        On the eGPU preset that meant every request was pruned to 212 992
        tokens:

            qwen3.8-27b          window 229 376   prompt+output 229 376   ok
            qwen3.8-27b-vision   window 196 608   prompt+output 229 376   +32 768
            gemma-4-26b          window  32 768   prompt+output 229 376  +196 608

        and it was not even transient: while llama-server holds the card,
        /health asks :5000, gets nothing, and skips the adopt — so `n_ctx`
        stays at the text model's value indefinitely. That is A-2026-01 in a
        new costume, and this repository already knows what the upstream does
        with a prompt past its window: keeps the head and the tail and throws
        the MIDDLE away, reporting it to nobody.

        WHO IS AUTHORITATIVE ABOUT `n_ctx`, and why it differs by engine:

        - **FastStrata: the declaration wins.** This gateway starts that
          process from the run config whose `--max-context` the declaration
          copies, so the number is not a guess about somebody's server.
        - **Remote: discovery wins**, but only for the address
          discovery was actually performed against. The process is somebody
          else's and may have come up with a different model entirely; that is
          exactly the failure A-2026-01 records. A declaration is used only
          when discovery has nothing to say about that address.
        """
        if spec is None:
            return self

        changes: dict[str, Any] = {}

        n_ctx = self._n_ctx_for(spec)
        if n_ctx and n_ctx != self.n_ctx:
            changes["n_ctx"] = n_ctx
        if spec.context_reserve:
            changes["context_reserve"] = spec.context_reserve
        if spec.spill_max_inline_bytes:
            changes["spill_max_inline_bytes"] = spec.spill_max_inline_bytes
        if spec.max_n_predict:
            changes["max_n_predict"] = spec.max_n_predict
            # The reasoning budget is DERIVED from the ceiling rather than
            # given a knob of its own, because the two are not independent:
            # the budget must fire FIRST or the turn ends `length` with no
            # answer, no tool call and no retry, where the budget would have
            # abandoned the generation and retried invisibly. A per-model
            # ceiling with a global budget breaks that ordering silently on
            # whichever model is smallest -- so a model that moves one moves
            # both. TestTheBudgetFiresBeforeTheCeiling holds it.
            changes["reasoning_budget_chars"] = derive_reasoning_budget(
                spec.max_n_predict)
        if spec.sampling_preset:
            changes["sampling_preset"] = spec.sampling_preset
        if spec.sampling:
            changes["model_sampling"] = dict(spec.sampling)
        if spec.temperature_floor:
            changes["temperature_floor"] = spec.temperature_floor
        fmt = prompt_format_for_family(spec.family)
        if fmt != self.prompt_format:
            changes["prompt_format"] = fmt
        if bool(spec.thinking) != self.thinking:
            changes["thinking"] = bool(spec.thinking)
        dialect = tool_dialect_for(spec.family, spec.tool_dialect)
        if dialect != self.tool_dialect:
            changes["tool_dialect"] = dialect
        if spec.reasoning != self.reasoning:
            changes["reasoning"] = spec.reasoning

        if not changes:
            return self
        return dataclasses.replace(self, **changes)

    def _n_ctx_for(self, spec: "ModelSpec") -> int:
        """The window that applies to this model. See `for_spec`."""
        from gateway.model_registry import ENGINE_POLYSTRATA

        if spec.engine == ENGINE_POLYSTRATA and spec.n_ctx:
            return spec.n_ctx
        # **One address, several windows: discovery describes the RESIDENT
        # model, not the one asked for.** A remote server may hot-swap
        # between checkpoints on one port that do not share a window. `/props` then
        # answers for whichever model the LAST request loaded -- so a Gemma
        # turn arriving while Qwen is resident would be pruned against Qwen's
        # window and handed to an engine that is about to have far less room,
        # which is A-2026-01 exactly. The declaration is the only per-model
        # answer there is, and `TestTheDeclaredWindowMatchesTheServer` holds
        # every declaration to what server.py actually allocates.
        #
        # **Except for the model that IS resident**: for that one, discovery
        # describes the very process that will serve the request, so it wins
        # exactly as it did before a second window existed -- which keeps the
        # A-2026-01 protection for Qwen in the case that matters most, Qwen
        # serving Qwen. Only a model that is NOT loaded falls back to its
        # declaration, because nothing about it can be discovered until the
        # swap has happened.
        if spec.n_ctx and self._endpoint_serves_several_windows(spec.upstream):
            resident = getattr(self, "n_ctx_model", "")
            if not (self.n_ctx and resident
                    and _same_endpoint(spec.upstream, self.upstream)
                    and resident in (spec.target, spec.id)):
                return spec.n_ctx
        if self.n_ctx and _same_endpoint(spec.upstream, self.upstream):
            return self.n_ctx
        return spec.n_ctx or (self.n_ctx or 0)

    def _endpoint_serves_several_windows(self, upstream: str) -> bool:
        """Whether models declared on this address disagree about the window."""
        windows = {s.n_ctx for s in self.registry()
                   if s.n_ctx and _same_endpoint(s.upstream, upstream)}
        return len(windows) > 1

    def effective_sampling(self) -> dict[str, dict[str, Any]]:
        """What each model is actually sampled with, keyed by model id.

        `/health` reported `sampling_preset` alone, which names a preset and
        says nothing about what reaches a token. On 2026-09-21 that was
        misleading twice over: the upstream of that day read no sampling field
        at all, so `qwen-coder-precise` described nothing; and once it did,
        the per-model `sampling` dict overrode most of the preset anyway. A
        name is not a value, and the question a reader has at `/health` is
        always "what is this model being sampled with right now".

        Resolved the same way a request resolves it -- `for_spec` then
        `get_sampling_params` -- rather than re-derived, so this cannot
        describe an ordering the request path does not use. Per-model
        `sampling` beats the CLIENT, so for a model that declares a
        parameter, the value here is what gets sent and nothing can move it.

        **It is where a turn STARTS, not a promise about every turn.**
        Adaptive sampling still moves temperature and min_p per request, from
        retry depth and context pressure; when it does, `gateway.log` says so
        on an `ADAPTIVE SAMPLING:` line, which is the only place a deviation
        is visible. Nothing here can show that, because it has no request.

        An empty registry answers under `"*"`: a preset with no `models` list
        serves every name the same way, and inventing an entry per alias
        would suggest a per-model resolution that is not happening.
        """
        registry = self.registry()
        if not len(registry):
            return {"*": self.get_sampling_params()}
        return {spec.id: self.for_spec(spec).get_sampling_params()
                for spec in registry}

    def registry(self) -> "ModelRegistry":
        """The preset's model registry, built once and kept.

        Built lazily rather than in `__post_init__` because `models` is plain
        data that the GUI and the tests both assign directly; caching on first
        use means whoever changed it gets a registry that agrees with it, and
        the request path does not rebuild one per turn.
        """
        from gateway.model_registry import ModelRegistry

        cached = getattr(self, "_registry", None)
        if cached is not None and getattr(self, "_registry_src", None) is self.models:
            return cached
        registry = ModelRegistry.from_list(self.models)
        for err in registry.errors:
            _log.warning("MODEL REGISTRY: %s", err)
        self._registry = registry
        self._registry_src = self.models
        return registry

    def spec_for(self, name: str) -> "ModelSpec | None":
        """The registry entry a client's model name refers to, if any.

        Tried on the name the client SENT and on what the alias map makes of
        it, in that order. A registry id is the direct route; going through
        the aliases as well is what keeps `qwen3.8-27b` working for a rig
        that had it as an alias before it was a registry entry.
        """
        registry = self.registry()
        if not len(registry):
            return None
        return registry.resolve(name) or registry.resolve(self.resolve_model(name))

    def adopt_upstream_n_ctx(self, reported: Any, model: Any = None) -> int | None:
        """Take a context window freshly reported by upstream `/props`.

        Returns the PREVIOUS value when it changed (the caller logs it), and
        None when nothing happened — which is every ordinary poll.

        Until 2026-09-03 `n_ctx` was discovered once, in the lifespan, and
        assigned nowhere else, so switching the model on the upstream left the
        gateway pruning against a window that no longer existed. In the
        direction that matters it is a hard failure, not waste:
        the Qwen2.5-Coder-3B launcher served on the SAME port with
        `CTX_SIZE=32768`, so a gateway started against the 35B model went on
        sending prompts four times the window. A-2026-01. That launcher was
        deleted on 2026-09-10, but the shape survives it: any second server on
        a port a preset points at can do the same.

        Identity is checked before assigning on purpose. The prune limit
        decides how much history survives, so moving it rewrites the prompt
        and costs the whole KV prefix; it may move when the upstream really
        changed and at no other time.
        """
        if not isinstance(reported, int) or isinstance(reported, bool) or reported <= 0:
            return None
        # Which model that window belongs to. Recorded on every poll, changed
        # or not, because it is what lets `_n_ctx_for` keep believing
        # discovery for the RESIDENT model when one address serves several
        # windows. "unknown" is no answer, so it is not recorded as one.
        if isinstance(model, str) and model.strip() and model.strip() != "unknown":
            self.n_ctx_model = model.strip()
        if reported == self.n_ctx:
            return None
        previous = self.n_ctx
        self.n_ctx = reported
        return previous

    #: Context window assumed when upstream could not be asked. Every model
    #: this project serves is configured with at least this much, so it is the
    #: only assumption that cannot overflow one of them. The previous value
    #: here was a hardcoded 120 000 — a number the gateway had never checked
    #: against anything, in force for the 3 startups in 115 where discovery
    #: failed.
    FALLBACK_N_CTX = 32_768

    def get_prune_limit(self) -> int:
        """Max tokens for context pruning = n_ctx − reserve.

        With no discovery the fallback is deliberately pessimistic: too small
        a limit prunes more history than necessary, too large a one produces a
        prompt the server rejects. While the desktop app is running this lasts
        at most until the first `/health` poll, which adopts the real value.
        """
        # The floor is on BOTH branches. It used to be on the fallback only,
        # so a reserve larger than a discovered window produced a negative
        # limit and a prompt pruned to nothing -- reachable the moment a
        # window stopped being 229k, which is what per-model limits made
        # ordinary. `ModelSpec.from_dict` refuses that combination up front;
        # this is the backstop for a window that arrives from discovery.
        if self.n_ctx:
            return max(4096, self.n_ctx - self.context_reserve)
        return max(4096, self.fallback_n_ctx - self.context_reserve)

    def get_sampling_params(self, client_overrides: dict | None = None) -> dict[str, Any]:
        """Compute final sampling parameters.

        Priority (highest → lowest):
          1. Per-parameter env overrides  (LOCAL_LLM_GATEWAY_TEMPERATURE etc.)
          2. Client request overrides     (only temperature, top_p, top_k, seed)
          3. Preset defaults

        The client is NOT allowed to override repetition_penalty,
        presence_penalty or any dry_* parameter — those are
        gateway-controlled.
        """
        overrides = client_overrides or {}
        preset = dict(SAMPLING_PRESETS.get(self.sampling_preset, SAMPLING_PRESETS["legacy"]))

        # Client may override a limited set
        CLIENT_ALLOWED = {"temperature", "top_p", "top_k", "seed"}
        for key in CLIENT_ALLOWED:
            if key in overrides:
                preset[key] = overrides[key]

        # The ACTIVE model's own values, above the client and below the
        # environment. This is the one place the gateway puts its own
        # configuration ahead of what the client asked for, and the reason is
        # the reason the aliases exist: OpenCode hardcodes temperature=0.55
        # for any model id containing "qwen", which is a guess keyed on a
        # NAME, while this is a decision somebody wrote down for this model.
        # Only declared keys are taken, so a model that sets temperature
        # still lets the client control top_p.
        for key, value in (self.model_sampling or {}).items():
            preset[key] = value

        # Env var overrides (highest priority)
        ENV_MAP = {
            "LOCAL_LLM_GATEWAY_TEMPERATURE":      ("temperature",       float),
            "LOCAL_LLM_GATEWAY_TOP_P":            ("top_p",             float),
            "LOCAL_LLM_GATEWAY_TOP_K":            ("top_k",             int),
            "LOCAL_LLM_GATEWAY_MIN_P":            ("min_p",             float),
            "LOCAL_LLM_GATEWAY_REPETITION_PENALTY": ("repetition_penalty", float),
            "LOCAL_LLM_GATEWAY_PRESENCE_PENALTY":   ("presence_penalty",   float),
            "LOCAL_LLM_GATEWAY_DRY_MULTIPLIER":     ("dry_multiplier",     float),
            "LOCAL_LLM_GATEWAY_DRY_BASE":           ("dry_base",           float),
            "LOCAL_LLM_GATEWAY_DRY_ALLOWED_LENGTH": ("dry_allowed_length", int),
            "LOCAL_LLM_GATEWAY_DRY_PENALTY_LAST_N": ("dry_penalty_last_n", int),
        }
        for env_key, (param_name, cast) in ENV_MAP.items():
            val = os.environ.get(env_key)
            if val is not None:
                try:
                    preset[param_name] = cast(val)
                except (ValueError, TypeError):
                    pass

        # Filter out None / 0 top_k (llama-server interprets 0 as "no limit")
        return {k: v for k, v in preset.items() if v is not None}
