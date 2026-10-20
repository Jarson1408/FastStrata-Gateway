"""
Conversation normalizer + ChatML renderer + request body builder.

Extracts and improves the following from _legacy.py:
  - format_chatml()       → ChatML prompt rendering (v2: tool grouping, no historical <think>)
  - inject_tools_to_prompt() → Tool definitions → system prompt
  - build_generate_body() → /completion request body builder
  - prune_messages()      → Context overflow guard

New: ConversationNormalizer — pre-processing pipeline:
  - Merge multiple system messages
  - Fix orphaned tool results
  - Strip historical reasoning_content (save context)
  - Stabilize system prefix (KV cache optimization)
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import uuid
from pathlib import Path

from gateway import prompts
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from gateway.config import GatewayConfig
    from gateway.upstream import UpstreamClient

_log = logging.getLogger("gateway.normalizer")


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

# The multimodal frame, and the third member of the family that
# `_neutralize_control_tokens` and `_neutralize_wrappers` belong to, further
# down. It lives up here because `_flatten_content_parts` needs it too.
#
# llama-server takes images on /completion by a marker in the prompt plus an
# ordered `multimodal_data` array in the body, and it pairs them POSITIONALLY.
#
# It does not validate the counts. Measured against a live server on
# 2026-09-13: two markers with one payload, one marker with two, and a payload
# with no marker at all ALL came back HTTP 200 -- the extra is dropped and the
# model answers about whichever image it did get. An earlier version of this
# comment said the request is refused; that was read out of a string in
# mtmd.dll and is not what this build does.
#
# Which makes the neutralization below more necessary, not less. A refusal
# would be loud. This is silent: a user, a file or a tool result containing the
# literal `<__media__>` shifts which image the model is looking at, or makes
# one vanish while the model answers confidently about another.
#
# It is broken on the way in for every role, exactly like `<|`, and the REAL
# markers are written by the renderer from `_gateway_media` rather than being
# carried in the text. That is what makes the count a property of the message
# objects instead of a property of whatever the conversation happens to say.
MEDIA_MARKER = "<__media__>"
_MEDIA_MARKER_RE = re.compile(re.escape(MEDIA_MARKER))

#: Where a message keeps its images, base64 payloads in order. Written by
#: normalization, read by the renderer and by build_generate_body, and carried
#: along by pruning -- which is the point: an image evicted with its turn must
#: take its payload with it, or the markers and the array stop matching.
MEDIA_KEY = "_gateway_media"


def _image_payload(part: object) -> str | None:
    """The base64 payload of one content part, or None when it is not an image.

    Accepts the two shapes clients actually send -- OpenAI's
    `{"type": "image_url", "image_url": {"url": ...}}` and the newer
    `{"type": "input_image", "image_url": ...}` -- and returns what
    llama-server's `multimodal_data` wants: the base64, without the data-URI
    preamble. The field is named `data`, not `url`, and stripping the prefix
    here means one place decides rather than every caller guessing.

    An `http(s)://` image is refused rather than fetched. Fetching it would
    make the gateway issue a request to an address that arrived inside a
    conversation, which is the one thing a proxy handling other people's
    content must not do on its own initiative.
    """
    if not isinstance(part, dict):
        return None
    if part.get("type") not in ("image_url", "input_image", "image"):
        return None

    raw = part.get("image_url")
    if isinstance(raw, dict):
        raw = raw.get("url")
    if not isinstance(raw, str):
        raw = part.get("data") if isinstance(part.get("data"), str) else None
    if not isinstance(raw, str) or not raw.strip():
        return None

    value = raw.strip()
    if value.startswith("data:"):
        marker = ";base64,"
        idx = value.find(marker)
        if idx == -1:
            _log.warning("Dropping an image: data URI is not base64-encoded.")
            return None
        return value[idx + len(marker):]
    if value.startswith(("http://", "https://")):
        _log.warning("Dropping an image given as a URL: the gateway does not "
                     "fetch addresses that arrive inside a conversation.")
        return None
    # Bare base64 -- some clients send it without the preamble.
    return value


def _newest_user_input(messages: list[dict]) -> list[dict]:
    """The messages the client added since the model last spoke.

    **The note a client leaves in place of a payload does not go away.** It
    replaces the attachment in the message it came with, and that message
    stays in the conversation for the rest of the session -- so a detector
    that reads the whole history reports the same single event again on every
    turn. Measured 2026-09-16 on a three-hour session: ONE pdf, THIRTY-THREE
    warnings, spread over forty-five minutes. That is precisely how a guard
    turns into something a reader scrolls past, which this file's own rules
    warn about in two other places.

    Everything after the last assistant message is what arrived for THIS
    turn. In an agentic loop that is usually nothing at all -- a turn driven
    by a tool result carries no new user message -- so the question is asked
    on the turn where the payload actually went missing, and not again.
    """
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            return messages[i + 1:]
    return messages


class MediaReport(NamedTuple):
    """One thing to say about payloads a client removed.

    `audit_error` is None when nobody did anything wrong, which is what keeps
    a correctly declined pdf out of `llmgw_errors_total`.
    """
    level: str                 # "warning" | "info"
    message: str
    audit_error: str | None


def stripped_media_reports(kinds: list[str], *, vision: bool,
                           model: str) -> list[MediaReport]:
    """What to SAY about payloads the client removed, ready to emit.

    **The wording is the part that was wrong, so it lives where a test can
    read it** rather than inside the request path where only a grep can reach
    it. Measured 2026-09-16: a pdf was correctly declined by the client and
    the gateway answered with the image message -- claiming *"'<model>' can
    serve them"* and advising a `modalities.input` key the config already
    had. Two false statements and a remedy that cannot work, about the one
    case that actually happens on this rig.

    So the kinds are split by whether THIS model could have served them:

    * **servable** -- an image, to a model that can see. The bytes existed and
      one key in the client's own config would have carried them through.
      That is the 2026-09-14 incident, it is a real defect, and it is a
      WARNING plus an audit error.
    * **declined** -- anything else. The client was right, no configuration
      changes it, and the honest advice is to convert the file. Said at INFO
      and recorded nowhere, because it is not a fault -- but said, because the
      file did not reach the model and the only other evidence is the model
      answering about something it never read.
    """
    servable = [k for k in kinds if stripped_media_is_servable(k, vision=vision)]
    declined = [k for k in kinds if not stripped_media_is_servable(k, vision=vision)]
    out: list[MediaReport] = []
    if servable:
        names = ", ".join(sorted(set(servable)))
        out.append(MediaReport(
            "warning",
            # %-formatted rather than an f-string so the prose guard can tell
            # a log line from a model-facing prompt, which is the same
            # exemption it already makes for print() and logging formats.
            ("CLIENT STRIPPED MEDIA: %d %s payload(s) were replaced with a "
             "note before this request was sent, and '%s' can serve them. The "
             "gateway never received the bytes. Check the client's model entry "
             "declares the modality (OpenCode: modalities.input must contain "
             "\"image\") — run the Configuration tab's 'Export to OpenCode'."
             ) % (len(servable), names, model),
            "client_stripped_media"))
    if declined:
        names = ", ".join(sorted(set(declined)))
        out.append(MediaReport(
            "info",
            ("CLIENT DROPPED MEDIA: %d %s payload(s) were replaced with a note "
             "before this request was sent. '%s' has no %s input, so the client "
             "was right to remove them — but the model never saw the file. "
             "Convert it first: a pdf becomes text, or one image per page."
             ) % (len(declined), names, model, names),
            None))
    return out


def stripped_media_is_servable(kind: str, *, vision: bool) -> bool:
    """Could THIS model have served the modality the client removed?

    The difference decides which of two very different things happened, and
    the message must not confuse them:

    * an **image** removed from a model that can see is a MISCONFIGURATION --
      the client's own model entry does not declare the modality, the bytes
      existed, and one key in a config file would have carried them through;
    * anything else -- a pdf, audio, video -- is the client being RIGHT. This
      model has no such input, and telling the user to declare
      `modalities.input: "image"` is advice that cannot work, about a
      capability nothing here has.

    Measured 2026-09-16: the second case is the one that actually happens on
    this rig, and the message it got was written for the first -- it claimed
    the model "can serve them" and named a key the config already had.
    """
    return vision and kind.strip().lower() == "image"


def detect_stripped_media(messages: list[dict]) -> list[str]:
    """The modalities a CLIENT removed before the request reached the gateway.

    A client that decides the model cannot see does not send a smaller
    request -- it sends the same request with the payload replaced by a
    sentence, and the gateway is then serving a turn indistinguishable from
    one that never carried a picture. That is not a hypothetical: on
    2026-09-14 it was the whole of a vision failure, and the only trace
    anywhere was an ABSENCE -- no `MULTIMODAL:` line, no image encode
    upstream, no error from anyone. See "The Model List Is Two Places".

    Three conditions, and all three matter:

    * **The RAW client messages, before normalization.** Roles here are
      exactly what the client sent. `_fix_orphaned_tool_results` can turn a
      tool result into a user turn, and a tool result is precisely where a
      `read` of a file quoting this phrase would land -- including this
      repository's own AGENTS.md.
    * **User turns only.** The note replaces a part of the message the client
      composed; the model's own history quoting the phrase is not evidence
      that anything was stripped this turn.
    * **Nothing else arrived.** A conversation that also carries a real image
      is one where the transport works and some other modality -- a PDF, say
      -- was correctly declined. Only a turn with the note and no payload at
      all describes the failure.

    Returns the client's own word for what it removed, in order, so the log
    line can say `image` rather than `media`. Empty means nothing to report,
    which is the answer for almost every request ever served.

    **Asked only of what the user added THIS turn** -- see
    `_newest_user_input`. The note does not go away, and a detector that reads
    the whole history reports one event once per turn for the rest of the
    session.
    """
    from gateway.lodestar.client_contract import stripped_media_re

    pattern = stripped_media_re()
    found: list[str] = []
    for msg in _newest_user_input(messages):
        content = msg.get("content")
        texts: list[str] = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if _image_payload(part) is not None:
                    return []          # a payload did arrive; nothing to report
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
                elif isinstance(part, str):
                    texts.append(part)
        if msg.get(MEDIA_KEY):
            return []
        if msg.get("role", "user") != "user":
            continue
        found.extend(m.group(1) for text in texts for m in pattern.finditer(text))
    return found


#: Told once per process. A gateway that cannot enforce its own image cap has
#: to say so, and saying it on every image would bury it.
_PILLOW_WARNED = False


def cap_image_pixels(payload: str, max_pixels: int) -> str:
    """Downscale a base64 image that is larger than the card can afford.

    Returns the payload unchanged when it is already small enough, when the
    cap is off, or when the image cannot be read -- **an image the gateway
    cannot decode is passed on rather than dropped**, because the model may
    well handle a format Pillow does not and losing the picture is the worse
    failure of the two.

    WHY THERE IS A CAP AT ALL
    -------------------------
    The Qwen3-VL preprocessor accepts up to **16 384 image tokens**
    (`size.longest_edge` is 16 777 216 px and the arithmetic is `pixels /
    1024` exactly). Measured 2026-09-14 against the live card, at a full
    196 608 context with the vision tower loaded:

        1440p screenshot    3 600 image tokens   1 313 MiB free
        4K screenshot       8 160 image tokens     879 MiB free
        the 16 384 ceiling                       about 1 217 MiB BELOW that

    The peak grows 97 KiB per image token and the encode transient grows
    faster still -- 148 MiB at 768 tokens, 1 808 at 16 384. Nothing on this
    card pays the ceiling, and an out-of-memory there lands in whichever
    process asks for the next allocation, quite possibly the one serving the
    user's turn. **Nobody reads a cursor readout better at sixteen
    megapixels**, so the cap costs nothing worth having.

    ASPECT RATIO IS PRESERVED, AND THAT IS NOT COSMETIC
    ---------------------------------------------------
    This rig's images are oscilloscope captures whose whole content is the
    geometry of a trace against a graticule. Squashing one to fit a pixel
    budget would keep every pixel and destroy the only thing being read.
    """
    global _PILLOW_WARNED
    if not payload or max_pixels <= 0:
        return payload

    try:
        from PIL import Image
    except ImportError:
        if not _PILLOW_WARNED:
            _PILLOW_WARNED = True
            _log.warning(
                "image_max_pixels is set to %d but Pillow is not installed, so "
                "oversized images are passed through unchanged. Install the "
                "'vision' extra to enforce it.", max_pixels)
        return payload

    import base64
    import io as _io
    import math

    try:
        raw = base64.b64decode(payload, validate=False)
        # `Image.open` reads the header and stops, so an image already under
        # the cap costs a parse and no decode at all -- which is the common
        # case and has to stay cheap.
        with Image.open(_io.BytesIO(raw)) as probe:
            width, height = probe.size
            if width * height <= max_pixels:
                return payload
            probe.load()
            scale = math.sqrt(max_pixels / float(width * height))
            new = (max(1, int(width * scale)), max(1, int(height * scale)))
            shrunk = probe.convert("RGB").resize(new, Image.Resampling.LANCZOS)
        buf = _io.BytesIO()
        shrunk.save(buf, format="PNG", optimize=False)
        out = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        _log.warning("Could not downscale an image (%s: %s); passing it "
                     "through unchanged.", type(exc).__name__, exc)
        return payload

    _log.info("IMAGE CAPPED: %dx%d (%d px) -> %dx%d (%d px), %d -> %d KB",
              width, height, width * height, new[0], new[1], new[0] * new[1],
              len(payload) // 1024, len(out) // 1024)
    return out


def media_of(msg: dict) -> list[str]:
    """The images one message carries, in order. Empty for almost every one."""
    media = msg.get(MEDIA_KEY)
    if isinstance(media, list):
        return [m for m in media if isinstance(m, str) and m]
    return []


def collect_media(messages: list[dict]) -> list[str]:
    """Every image in the conversation, in the order the renderer emits them.

    The one function both sides of the contract use. `format_chatml` writes one
    marker per item of `media_of(msg)` walking the same list in the same order,
    so the prompt's marker count and this list's length cannot drift apart
    without a test noticing -- which matters because llama-server does not
    check: a mismatch is served as HTTP 200 with an image silently dropped or
    silently shifted. See the note above MEDIA_MARKER.
    """
    out: list[str] = []
    for msg in messages:
        out.extend(media_of(msg))
    return out



# Pattern for volatile system prefix lines (KV cache stabilization)
_VOLATILE_PREFIX_RE = re.compile(
    r"^(?:Current directory:|Working directory:|Workspace:|Project root:|Directory:|Git status:|Date:|Time:|Message ID:|x-|Timestamp:)",
    re.IGNORECASE,
)

# Pattern for resolving active workspace root from system prompt
_CURRENT_DIR_RE = re.compile(
    r"^(?:Current|Working|Workspace|Project root|Directory)(?:\s+(?:working\s+)?directory)?:\s*(.+)$|"
    r"^The current working directory is\s*(.+)$",
    re.IGNORECASE,
)

# Re-export from new lodestar module for backward compatibility
from gateway.lodestar import (
    LodestarState,
    determine_state as _lodestar_determine_state,
)
from gateway.lodestar.subagent import is_subagent_request as _lodestar_is_subagent


def _append_subagent_budget(msgs: list[dict], config) -> list[dict]:
    """Tell a subagent it is over its turn budget, as the LAST message.

    Last, for the same reason the LODESTAR turn frame is last: this text
    changes every turn, and anything volatile placed ahead of the conversation
    re-prefills all of it. Everything before this stays byte-identical.

    A subagent gets none of the protocol, which is the point — but it also
    meant no limit reached it at all.
    """
    from gateway.lodestar.subagent import budget_state

    budget = getattr(config, "max_subagent_turns", 0)
    stage, turns = budget_state(
        msgs, budget, getattr(config, "subagent_turn_grace", 10))
    if stage is None:
        return msgs
    name = "subagent_budget" if stage == "soft" else "subagent_budget_exhausted"
    try:
        text = prompts.get(name, turns=turns, limit=budget)
    except KeyError:
        return msgs
    _log.warning("SUBAGENT BUDGET (%s): %d turns of %d", stage, turns, budget)
    return list(msgs) + [{"role": "system", "content": text}]


# ---------------------------------------------------------------------------
# ConversationNormalizer
# ---------------------------------------------------------------------------

class ConversationNormalizer:
    """Pre-processing pipeline for conversation messages.

    Normalizes the message history before ChatML rendering to handle
    edge cases from various clients (OpenCode, continue.dev, etc.).

    All methods return new lists (never mutate the input).
    """

    def normalize(self, messages: list[dict], config: GatewayConfig | None = None,
                  agentic: bool = True, vision: bool = False) -> list[dict]:
        """Run the full normalization pipeline.

        agentic=False (request registered no tools) skips LODESTAR context
        injection — a request that cannot call tools cannot follow the
        protocol, and graph-builder prompts must not be polluted with it.

        vision=True keeps the images the client sent, on the message rather
        than in it. It is a property of the MODEL, not of the request: a
        picture sent to a text model is not something the gateway can serve by
        trying harder.
        """
        # A shallow copy is not enough: _restore_skill_tool_calls rewrites the
        # nested tool_calls dicts, and with m.copy() those objects are still
        # shared with the caller's list — so normalization silently rewrote the
        # client's own message history, contradicting the promise above.
        msgs = [copy.deepcopy(m) if m.get("tool_calls") else m.copy()
                for m in messages]
        msgs = self._flatten_content_parts(
            msgs, vision=vision,
            image_max_pixels=int(getattr(config, "image_max_pixels", 0) or 0))
        msgs = self._scrub_assistant_special_tokens(msgs)
        msgs = self._merge_system_messages(msgs)
        if agentic:
            msgs = self._inject_context_files(msgs, config)
        msgs = self._fix_orphaned_tool_results(msgs)
        msgs = self._restore_skill_tool_calls(msgs)
        msgs = self._strip_historical_reasoning(msgs)
        msgs = self._stabilize_system_prefix(msgs)
        msgs = self._detect_and_mitigate_loops(msgs)
        msgs = self._annotate_lesson_results(msgs)
        msgs = self._neutralize_completed_nudges(msgs)
        return msgs

    def _flatten_content_parts(self, msgs: list[dict],
                               vision: bool = False,
                               image_max_pixels: int = 0) -> list[dict]:
        """Flatten OpenAI multi-part content ([{type: text, text: ...}]) to a string.

        OpenCode sends user messages as content-part lists. Without this step
        format_chatml rendered the Python repr of the list into the prompt
        (`[{'type': 'text', 'text': '<EXTREMELY_IMPORTANT>...`), wasting
        tokens and breaking string-based detection (e.g. user_requested_skills
        silently skipped non-string content).

        `vision` says whether the model that will serve this request can see.
        The images are lifted OUT of the content and onto `_gateway_media`
        rather than left inline, for two reasons that both matter downstream:
        the content stays a plain string, so every string-based check in this
        pipeline goes on working unchanged, and the prompt never carries a
        megabyte of base64 that would then be tokenized, pruned against and
        written to a full-prompt dump.

        Without vision the image is replaced by a one-line note rather than
        dropped in silence. Silence is what it used to be, and it produced the
        worst possible turn: "describe this" with nothing to describe, and the
        model apologising for something the gateway did.
        """
        result: list[dict] = []
        for m in msgs:
            content = m.get("content")
            if isinstance(content, list):
                parts: list[str] = []
                media: list[str] = []
                for p in content:
                    if isinstance(p, dict):
                        if p.get("type") == "text" and isinstance(p.get("text"), str):
                            parts.append(p["text"])
                            continue
                        payload = _image_payload(p)
                        if payload is None:
                            continue
                        if vision:
                            # Capped HERE, before anything else sees it: the
                            # prune charge, the full-prompt dump and the chat
                            # body all read `_gateway_media`, and a cap applied
                            # later would leave three readers disagreeing about
                            # how big the picture is.
                            media.append(cap_image_pixels(
                                payload, image_max_pixels))
                        else:
                            parts.append("[image omitted: this model has no "
                                         "vision. Ask for a model that has.]")
                    elif isinstance(p, str):
                        parts.append(p)
                m = m.copy()
                m["content"] = "\n\n".join(parts)
                if media:
                    m[MEDIA_KEY] = media
            result.append(m)
        return result

    _SPECIAL_TOKEN_RE = re.compile(r"<\|[A-Za-z0-9_]{1,30}\|>")

    def _scrub_assistant_special_tokens(self, msgs: list[dict]) -> list[dict]:
        """Remove leaked vocab control tokens from assistant history.

        The model occasionally emits tokens like <|mask_start|> as text
        (observed 2026-07-09 under high context pressure). Once stored by the
        client and re-sent, format_chatml would render them into the prompt
        where llama-server tokenizes them as REAL control tokens, corrupting
        the ChatML structure. Only assistant content/reasoning is scrubbed —
        user and tool messages may legitimately quote such strings.
        """
        result: list[dict] = []
        for m in msgs:
            if m.get("role") == "assistant":
                changed = False
                new_m = m
                for key in ("content", "reasoning_content", "reasoning"):
                    val = m.get(key)
                    if isinstance(val, str) and "<|" in val:
                        if not changed:
                            new_m = m.copy()
                            changed = True
                        new_m[key] = self._SPECIAL_TOKEN_RE.sub("", val)
                result.append(new_m)
            else:
                result.append(m)
        return result

    def _neutralize_completed_nudges(self, msgs: list[dict]) -> list[dict]:
        """Defuse the Phase D nudge once finalization is done.

        The nudge's echo output ("do BOTH steps... Then give your final
        answer") stays in history after the model completes Phase D. Being
        the most recent imperative in context, the model re-read it and
        REPEATED the whole finalization — duplicate lesson in the semantic
        DB and a second active-context write (observed 2026-07-14 23:39).
        Once phase_d_pending() says the cycle is closed, every marker tool
        response is rewritten to a completion notice so there is nothing
        left to obey.
        """
        from gateway.lodestar.state import PHASE_D_MARKER, phase_d_pending

        has_marker = any(
            m.get("role") == "tool" and PHASE_D_MARKER in str(m.get("content") or "")
            for m in msgs
        )
        if not has_marker or phase_d_pending(msgs):
            return msgs  # nothing to defuse, or the nudge is still applicable

        result: list[dict] = []
        for m in msgs:
            if m.get("role") == "tool" and PHASE_D_MARKER in str(m.get("content") or ""):
                m = m.copy()
                m["content"] = (
                    f"{PHASE_D_MARKER} Finalization already completed: the lesson "
                    "is saved and active-context.md is closed. Do NOT repeat "
                    "these steps. Give your final answer now."
                )
            result.append(m)
        return result

    def _annotate_lesson_results(self, msgs: list[dict]) -> list[dict]:
        """Prepend a staleness warning to any tool result that contains lesson lines.

        The semantic-memory `embed.py search` command returns HINTS from past
        sessions, not ground truth. A previous failure (2026-07-07 Capacitor
        session) showed the model trusting a lesson saying "I created X" and
        skipping the actual file creation. This annotation makes the staleness
        explicit so the model verifies before trusting.
        """
        from gateway.lodestar.lesson_annotator import annotate
        annotated = []
        for m in msgs:
            if m.get("role") == "tool" and isinstance(m.get("content"), str):
                new_m = m.copy()
                new_m["content"] = annotate(m["content"])
                annotated.append(new_m)
            else:
                annotated.append(m)
        return annotated

    def _inject_context_files(self, msgs: list[dict], config: GatewayConfig | None) -> list[dict]:
        """Single-channel LODESTAR injection (delegates to gateway.lodestar)."""
        if not config or not getattr(config, "inject_context", True) or not getattr(config, "lodestar_enabled", True):
            return msgs

        # Subagent sessions get NO LODESTAR block at all: the protocol
        # (including Phase D) belongs to the main session. The 2026-07-09
        # incident: an undetected task-subagent ran the full protocol and
        # cleared active-context.md while the main task was still running.
        if _lodestar_is_subagent(None, msgs):
            return _append_subagent_budget(msgs, config)

        ws_str = workspace_from_messages(msgs)
        if not ws_str:
            return msgs
        workspace_dir = Path(ws_str)

        state = LodestarState(determine_lodestar_state(msgs, config))
        from gateway.lodestar.context_injector import inject_lodestar_context
        return inject_lodestar_context(
            msgs, workspace_dir, state,
            max_synthesis=getattr(config, "max_synthesis_per_cycle", None),
            cycle_budget=getattr(config, "max_cycle_turns", 0),
            cycle_grace=getattr(config, "cycle_turn_grace", 0),
            context_max_chars=getattr(config, "lodestar_context_max_chars", None))

    def _merge_system_messages(self, msgs: list[dict]) -> list[dict]:
        """Merge multiple system messages into one at the beginning.

        Some clients send multiple system messages (e.g., one for the base
        system prompt, one for tool definitions). Qwen ChatML expects a single
        system block.
        """
        system_parts: list[str] = []
        non_system: list[dict] = []

        for m in msgs:
            if m.get("role") == "system":
                content = m.get("content") or ""
                if content.strip():
                    system_parts.append(content)
            else:
                non_system.append(m)

        if not system_parts:
            return non_system

        merged = {"role": "system", "content": "\n\n".join(system_parts)}
        # The gateway's own markers survive the merge. They record that THIS
        # process already injected into the system message, and a rebuilt dict
        # that drops them makes a second normalize() of the same messages
        # append the protocol block a second time.
        from gateway.lodestar.context_injector import GATEWAY_MESSAGE_KEYS

        for m in msgs:
            if m.get("role") == "system":
                for key in GATEWAY_MESSAGE_KEYS:
                    if key in m:
                        merged[key] = m[key]
        return [merged] + non_system

    def _fix_orphaned_tool_results(self, msgs: list[dict]) -> list[dict]:
        """Fix orphaned tool results and recover missing tool names from parent assistant tool_calls.

        When context pruning removes the assistant message with tool_calls but keeps
        the tool result, we reconstruct a synthetic assistant + tool_calls entry.
        """
        # Create a mapping of tool_call_id -> tool_name from assistant messages in history
        call_id_to_name = {}
        for m in msgs:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    tc_id = tc.get("id")
                    tc_name = (tc.get("function") or {}).get("name")
                    if tc_id and tc_name:
                        call_id_to_name[tc_id] = tc_name

        result: list[dict] = []
        i = 0
        while i < len(msgs):
            m = msgs[i]
            role = m.get("role")
            
            if role == "tool":
                # Collect the entire block of consecutive tool messages
                tool_block = []
                while i < len(msgs) and msgs[i].get("role") == "tool":
                    tm = msgs[i].copy()
                    
                    # Recover missing tool name from tool_call_id if possible
                    t_id = tm.get("tool_call_id")
                    if not tm.get("name") and t_id and t_id in call_id_to_name:
                        tm["name"] = call_id_to_name[t_id]
                        
                    tool_block.append(tm)
                    i += 1
                
                # Check if the message before this block was a valid assistant with tool_calls
                if result and result[-1].get("role") == "assistant" and result[-1].get("tool_calls"):
                    result.extend(tool_block)
                else:
                    # Orphaned block! Create a synthetic assistant for all tool calls in the block
                    tool_calls = []
                    for tm in tool_block:
                        t_id = tm.get("tool_call_id") or f"call_{uuid.uuid4().hex[:10]}"
                        t_name = tm.get("name") or "unknown"
                        tool_calls.append({
                            "id": t_id,
                            "type": "function",
                            "function": {
                                "name": t_name,
                                "arguments": "{}",
                            }
                        })
                    synthetic = {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": tool_calls
                    }
                    result.append(synthetic)
                    result.extend(tool_block)
            else:
                result.append(m)
                i += 1
        return result

    def _restore_skill_tool_calls(self, msgs: list[dict]) -> list[dict]:
        """Rewrite read tool calls of SKILL.md files back to skill tool calls
        in the history so the LLM's context remains consistent.
        """
        import re
        # Map to correlate tool call ID to rewritten skill name
        rewritten_calls = {}

        for m in msgs:
            role = m.get("role")
            if role == "assistant":
                tool_calls = m.get("tool_calls") or []
                for call in tool_calls:
                    func = call.get("function") or {}
                    name = func.get("name", "")
                    if name in ("read", "read_file", "view_file"):
                        args_str = func.get("arguments", "{}")
                        try:
                            args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        except Exception:
                            args = {}
                        
                        path_str = args.get("filePath") or args.get("path") or args.get("AbsolutePath") or ""
                        if path_str and path_str.lower().endswith("skill.md") and ("skills" in path_str.lower() or "config" in path_str.lower()):
                            # Extract skill name from path
                            parts = re.split(r'[\\/]', path_str)
                            if len(parts) >= 2:
                                skill_name = parts[-2]
                                func["name"] = "skill"
                                func["arguments"] = json.dumps({"skill_name": skill_name})
                                if "id" in call:
                                    rewritten_calls[call["id"]] = "skill"
            
            elif role == "tool":
                call_id = m.get("tool_call_id")
                if call_id in rewritten_calls:
                    m["name"] = "skill"

        return msgs

    def _strip_historical_reasoning(self, msgs: list[dict]) -> list[dict]:
        """Keep reasoning only for the current agentic loop (preserve_thinking).

        v2.5: assistant messages AFTER the last user-role message keep their
        reasoning_content — this is the model's working scratchpad for the
        tool loop in progress. Dropping it between tool turns is the main
        cause of the "amnesia" failure mode: the model plans steps 1-4 in
        <think>, executes step 1, and on the next turn the plan is gone.
        Reasoning from turns BEFORE the last user message is stripped to
        save context (matches Qwen3's native preserve_thinking semantics).
        """
        last_user_idx = -1
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                last_user_idx = i
                break

        result: list[dict] = []
        for i, m in enumerate(msgs):
            if (m.get("role") == "assistant"
                    and i < last_user_idx
                    and (m.get("reasoning_content") or m.get("reasoning"))):
                m = m.copy()
                m.pop("reasoning_content", None)
                m.pop("reasoning", None)
            result.append(m)
        return result

    def _stabilize_system_prefix(self, msgs: list[dict]) -> list[dict]:
        """Move volatile data from start of system message to the end.

        Volatile data (timestamps, git status, message IDs) at the beginning
        of the system prompt defeats llama-server's KV cache prefix matching.
        Moving it to the end preserves the stable prefix.
        """
        if not msgs or msgs[0].get("role") != "system":
            return msgs

        system_msg = msgs[0].copy()
        content = system_msg.get("content") or ""
        lines = content.split("\n")

        # Find volatile prefix lines
        volatile_lines: list[str] = []
        stable_start = 0
        for i, line in enumerate(lines):
            if _VOLATILE_PREFIX_RE.match(line.strip()):
                volatile_lines.append(line)
                stable_start = i + 1
            else:
                # Stop at first non-volatile, non-empty line
                if line.strip():
                    break
                # Allow blank lines between volatile lines
                if i > 0 and volatile_lines:
                    volatile_lines.append(line)
                    stable_start = i + 1

        if not volatile_lines:
            return msgs

        stable_part = "\n".join(lines[stable_start:])
        volatile_part = "\n".join(volatile_lines)
        system_msg["content"] = stable_part.strip() + "\n\n" + volatile_part.strip()

        return [system_msg] + msgs[1:]

    def _detect_and_mitigate_loops(self, msgs: list[dict]) -> list[dict]:
        """Detect retry loops of depth >= 3 and mitigate by pruning old tool results
        and injecting a system warning at the end.
        """
        from gateway.adaptive import count_retry_depth, _analyze_tool_sequence

        depth = count_retry_depth(msgs)
        if depth < 3:
            return msgs

        _log.warning("Loop mitigation triggered! Retry depth: %d", depth)

        sequence = _analyze_tool_sequence(msgs)
        if not sequence:
            return msgs

        last_call = sequence[-1]
        min_failed_turn = last_call["message_idx"]

        for i in range(len(sequence) - 2, -1, -1):
            prev_call = sequence[i]
            is_same_tool = prev_call["name"] == last_call["name"]
            was_failed = prev_call["failed"]
            
            same_resource = False
            if prev_call["file"] is not None and last_call["file"] is not None:
                same_resource = (prev_call["file"] == last_call["file"])
            else:
                norm_prev = re.sub(r"\s+", "", prev_call["args"])
                norm_last = re.sub(r"\s+", "", last_call["args"])
                same_resource = (norm_prev == norm_last)

            if is_same_tool and was_failed and same_resource:
                if prev_call["message_idx"] < min_failed_turn:
                    min_failed_turn = prev_call["message_idx"]
            else:
                break

        loop_start_idx = len(msgs)
        assistant_msg_count = 0
        for idx, m in enumerate(msgs):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                assistant_msg_count += 1
                if assistant_msg_count == min_failed_turn:
                    loop_start_idx = idx
                    break

        new_msgs = []
        pruned_count = 0
        for idx, m in enumerate(msgs):
            m_copy = m.copy()
            if idx < loop_start_idx and m_copy.get("role") == "tool":
                content = m_copy.get("content") or ""
                if len(content) > 1000:
                    pruned_count += 1
                    m_copy["content"] = (
                        f"{content[:400]}\n"
                        f"[... TRUNCATED BY GATEWAY TO PREVENT CONTEXT OVERFLOW (original length: {len(content)}) ...]\n"
                        f"{content[-400:]}"
                    )
            new_msgs.append(m_copy)

        if pruned_count > 0:
            _log.info("Loop mitigation: pruned %d historical tool response messages", pruned_count)

        warning_msg = {
            "role": "system",
            "content": prompts.get("loop_warning"),
        }
        new_msgs.append(warning_msg)
        return new_msgs


# Global singleton
_normalizer = ConversationNormalizer()


def normalize_messages(messages: list[dict], config: GatewayConfig | None = None,
                       agentic: bool = True, vision: bool = False) -> list[dict]:
    """Module-level convenience function."""
    return _normalizer.normalize(messages, config, agentic=agentic, vision=vision)


# A `_CURRENT_DIR_RE` hit proves only that some client wrote text after a
# colon. Status blocks carry lines like `Current: 5 %`, and two audit events in
# gateway_events.jsonl ended up logged under the session id `'5 %'` because of
# it. The damage is silent rather than loud: determine_lodestar_state() below
# returns READY whenever <workspace>/.context/LODESTAR.md is not a file — and
# for a bogus root it never is — so one false positive switches the whole
# protocol off for that request. Candidates are therefore validated as paths.
_ANCHORED_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\[^\\/]|/|~(?:[\\/]|$))")


def _is_directory_on_disk(val: str) -> bool:
    """True when `val` resolves to a directory reachable from this host."""
    try:
        return Path(val).is_dir()
    except (OSError, ValueError):
        return False


def _looks_like_workspace_path(val: str) -> bool:
    """True when `val` is plausibly a filesystem path rather than prose.

    Anchored forms (POSIX absolute, `C:\\...`, UNC, `~/...`) are accepted on
    shape alone — the announced root frequently does not exist on the gateway
    host. Anything else has to prove itself by actually being a directory here,
    which is what rejects `5 %`, `unknown`, `TBD` and friends.
    """
    if not val or "\n" in val:
        return False
    if _ANCHORED_PATH_RE.match(val):
        return True
    return _is_directory_on_disk(val)


def workspace_from_messages(messages: list[dict]) -> str | None:
    """The workspace root the client announced in its system prompt.

    Every candidate is path-validated before it is accepted, and a rejected one
    falls through to the next tag/line/message instead of ending the search —
    so a `Current: 5 %` status line no longer masks a real `Current directory:`
    header further down the same prompt.
    """
    for m in messages:
        content = m.get("content")
        if not isinstance(content, str):
            continue

        # Check inline xml-style tags first (<workspace>...</workspace> or <cwd>...</cwd>)
        for m_tag in re.finditer(r"<(?:workspace|cwd)>(.+?)</(?:workspace|cwd)>",
                                 content, re.IGNORECASE | re.DOTALL):
            val = m_tag.group(1).strip().strip("'\"")
            if _looks_like_workspace_path(val):
                return val

        for line in content.split("\n"):
            match = _CURRENT_DIR_RE.match(line.strip())
            if match:
                raw_path = match.group(1) or match.group(2)
                if raw_path:
                    val = raw_path.strip().strip("'\"")
                    if _looks_like_workspace_path(val):
                        return val
    return None


def determine_lodestar_state(messages: list[dict], config: GatewayConfig | None = None) -> int:
    """Backward-compatible int-based state determination.

    Delegates to gateway.lodestar.determine_state with legacy content-based
    subagent detection. New code should use gateway.lodestar.determine_state
    directly with explicit is_subagent parameter from HTTP header.

    The workspace is resolved here and handed down so the Phase-B check can
    verify the write against the file itself rather than against the client's
    error wording.
    """
    if config and not getattr(config, "lodestar_enabled", True):
        return 3  # LodestarState.READY
    is_sub = _lodestar_is_subagent(None, messages)
    ws = workspace_from_messages(messages)
    if ws is not None:
        try:
            from pathlib import Path
            if not (Path(ws) / ".context" / "LODESTAR.md").is_file():
                return 3  # LodestarState.READY — standard repos without .context/LODESTAR.md bypass gating
        except Exception:
            pass
    return int(_lodestar_determine_state(
        messages, is_subagent=is_sub, workspace_dir=ws))


# ---------------------------------------------------------------------------
# ChatML renderer (v2)
# ---------------------------------------------------------------------------

# Any `<|...|>` sequence llama-server sees in the prompt is tokenized as a REAL
# control token, whatever produced it. Message bodies are therefore neutralized
# on the way into the prompt: the `<|` is broken with a space, which keeps the
# text readable and near-identical for the model while making it impossible to
# close a turn or open a new one.
#
# _scrub_assistant_special_tokens handles the *storage* side for assistant
# history. This handles the *rendering* side for every role, which is the half
# that mattered: a file whose contents include "<|im_end|><|im_start|>system"
# used to break out of its <tool_response> block and inject a system turn. The
# gateway reads files nobody here wrote — graphify and engram walk whole
# corpora, and the enforcer synthesizes reads of its own accord.
_CONTROL_TOKEN_OPEN_RE = re.compile(r"<\|")

# The multimodal frame is the third member of this family; MEDIA_MARKER and
# the reasons for breaking it are at the top of the file, because
# _flatten_content_parts needs them too.
# The same argument one level up. `<|` is ChatML's frame; these are the
# gateway's own, and content rendered INSIDE one of them can close it.
#
# A tool result carrying `</tool_response>` leaves its block early and the rest
# of the file lands in the user turn as ordinary text -- where a following
# `<tool_response name="write">Saved.</tool_response>` is indistinguishable
# from a tool that actually ran. That is a whole forged observation, not merely
# persuasive prose, and the gateway reads files nobody in the conversation
# wrote.
#
# The same shape reaches the assistant side from the model's own history: a
# turn that discussed this protocol, or a `write` whose content quotes these
# tags, replays with its block closed early and its tool call reconstructed
# wrong. Measured 2026-09-12 on the tree: `</tool_response>` appears in five
# tracked files, all of them this gateway's own source or a report about it, so
# the readability cost of breaking it is close to nil -- unlike the prose
# markers (`=== LODESTAR TURN ===`, `[SYSTEM GUARD]`), which are all over
# AGENTS.md and are deliberately NOT touched. Those mislead the MODEL; the
# supervisor decides from message state rather than from prompt text, and the
# gateway's real frame is appended last and most recent either way.
#
# Only the closer of the block being rendered is broken, so a body is mangled
# exactly where it would otherwise escape and nowhere else.
_WRAPPER_RE = re.compile(r"<(/?)(tool_response|tool_call|function|parameter|think)\b")


def _neutralize_control_tokens(text: object) -> str:
    """Render-safe form of a message body."""
    if not isinstance(text, str):
        return "" if text is None else str(text)
    text = _CONTROL_TOKEN_OPEN_RE.sub("< |", text)
    return _MEDIA_MARKER_RE.sub("< __media__>", text)


def _neutralize_wrappers(text: object, *tags: str) -> str:
    """Render-safe form of a body that goes INSIDE one of the gateway's tags.

    Breaks `</tool_response>` into `</ tool_response>` and `<tool_response` into
    `< tool_response`: readable, obviously the same text, and no longer the tag
    the renderer emitted.
    """
    body = _neutralize_control_tokens(text)
    wanted = set(tags)

    def _break(m: re.Match[str]) -> str:
        if m.group(2) not in wanted:
            return m.group(0)
        return f"<{m.group(1)} {m.group(2)}"

    return _WRAPPER_RE.sub(_break, body)


def _media_markers(msg: dict) -> str:
    """One marker per image the message carries, in order.

    Emitted by the RENDERER out of `_gateway_media`, never carried in the
    text. That is what makes the marker count a property of the message
    objects rather than a property of whatever the conversation happens to
    say -- and llama-server does NOT check that the two agree. A mismatch is
    served, with an image dropped or shifted and no error at all.

    `collect_media` walks the same list in the same order, which is the other
    half of the same contract.
    """
    media = media_of(msg)
    if not media:
        return ""
    return "\n" + "\n".join(MEDIA_MARKER for _ in media) + "\n"


def render_tool_calls(msg: dict) -> str:
    """An assistant turn's tool calls, in the dialect the gateway asks for.

    Extracted so the two transports replay history identically. The chat
    transport hands llama.cpp a `messages` array instead of a rendered prompt,
    and if it rendered past calls differently the model would be shown one
    dialect in its history while being asked for another.
    """
    out = ""
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args = _parse_args(fn.get("arguments", "{}"))
        out += f"<tool_call>\n<function={name}>\n"
        for k, v in args.items():
            v_str = (json.dumps(v, ensure_ascii=False)
                     if isinstance(v, (dict, list)) else str(v))
            # A `write` whose content quotes these tags -- this repository's
            # own source does -- would otherwise replay with its call
            # reconstructed wrong.
            safe = _neutralize_wrappers(v_str, "parameter", "function", "tool_call")
            out += f"<parameter={k}>\n{safe}\n</parameter>\n"
        out += "</function>\n</tool_call>\n"
    return out


def render_tool_results(msgs: list[dict]) -> str:
    """Consecutive tool results, wrapped the way the model is taught to read."""
    out = ""
    for tm in msgs:
        name = tm.get("name", "")
        content = _neutralize_wrappers(tm.get("content", ""), "tool_response")
        if name:
            out += f'<tool_response name="{name}">\n{content}\n</tool_response>\n'
        else:
            out += f"<tool_response>\n{content}\n</tool_response>\n"
    return out


def format_chatml(messages: list[dict]) -> str:
    """Build a Qwen ChatML prompt from normalized messages.

    Improvements over v1 (_legacy.format_chatml):
      1. Tool response grouping — consecutive role=tool messages are merged
         into a single <|im_start|>user block with multiple <tool_response>.
      2. No <think> reconstruction from reasoning_content — Qwen's chat
         template auto-prefills <think> in the assistant turn, so
         reconstructing it causes double-thinking (§2.6 fix).
      3. Iterates with index for proper tool grouping.
    """
    prompt = ""
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")

        if role == "system":
            body = _neutralize_control_tokens(msg.get("content", ""))
            prompt += f"<|im_start|>system\n{body}{_media_markers(msg)}<|im_end|>\n"

        elif role == "user":
            body = _neutralize_control_tokens(msg.get("content", ""))
            prompt += f"<|im_start|>user\n{body}{_media_markers(msg)}<|im_end|>\n"

        elif role == "assistant":
            prompt += "<|im_start|>assistant\n"
            # v2.5 preserve_thinking: re-render the reasoning of the current
            # tool loop (normalizer keeps it only for turns after the last
            # user message). This is what keeps the model's multi-step plan
            # alive across tool calls. Historical turns have no reasoning
            # here, so no double-think is possible in old context.
            reasoning = _neutralize_wrappers(
                msg.get("reasoning_content") or msg.get("reasoning") or "",
                "think")
            if reasoning:
                prompt += f"<think>\n{reasoning}\n</think>\n"
            content = _neutralize_control_tokens(msg.get("content") or "")
            if content:
                prompt += f"{content}\n"
            prompt += _media_markers(msg)
            # Reconstruct tool_calls as XML
            prompt += render_tool_calls(msg)
            prompt += "<|im_end|>\n"

        elif role == "tool":
            # Group consecutive tool messages into a single user block
            tool_msgs = [msg]
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                tool_msgs.append(messages[j])
                j += 1
            prompt += "<|im_start|>user\n"
            for tm in tool_msgs:
                prompt += render_tool_results([tm])
                prompt += _media_markers(tm)
            prompt += "<|im_end|>\n"
            i = j
            continue

        i += 1

    prompt += "<|im_start|>assistant\n"
    return prompt


def render_prompt(messages: list[dict], prompt_format: str = "chatml",
                  thinking: bool = False, tool_dialect: str = "qwen_xml",
                  reasoning: str = "") -> str:
    """The `/completion` prompt in the active model's own chat format.

    ONE dispatch, so a reader that renders a prompt cannot pick the format
    differently from the one that built the request.
    """
    if prompt_format == "gemma4":
        from gateway.dialects.gemma4 import format_gemma4
        return format_gemma4(messages, thinking=thinking,
                             native_tools=(tool_dialect == "gemma4"))
    if prompt_format == "muse":
        from gateway.dialects.muse import format_muse
        return format_muse(messages, reasoning=reasoning or "high",
                           native_tools=(tool_dialect == "muse"))
    return format_chatml(messages)


def _parse_args(args_str: str) -> dict:
    """Safely parse tool call arguments JSON."""
    try:
        result = json.loads(args_str)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# Tool injection into system prompt
# ---------------------------------------------------------------------------

def inject_tools_to_prompt(client_body: dict, tools: list[dict],
                           dialect: str = "qwen_xml") -> None:
    """Inject tool definitions into the system prompt, removing tools array.

    Bypasses llama-server's native tool parser (which doesn't understand
    Qwen's XML format) by injecting tools as text in the system prompt.

    `dialect="gemma4"` describes them in Gemma 4's OWN protocol instead: the
    rules (without the Qwen-format bullets) go into the system text, and the
    tool list rides on the system message as `_gateway_tools`, for
    `format_gemma4` to render as `<|tool>declaration:...<tool|>` where the
    model's template puts them -- after the system text, inside its turn. A
    private key rather than text, for the reason `_gateway_media` is one: the
    renderer is the only reader that knows the frame.
    """
    if not tools:
        return

    if dialect in ("gemma4", "muse"):
        messages = client_body.get("messages") or []
        rules = prompts.get("tool_calling_rules_native")
        if messages and messages[0].get("role") == "system":
            sys_msg = messages[0]
            sys_msg["content"] = rules + "\n\n" + str(sys_msg.get("content", ""))
        else:
            sys_msg = {"role": "system", "content": rules}
            messages.insert(0, sys_msg)
        sys_msg["_gateway_tools"] = [t for t in tools if isinstance(t, dict)]
        client_body["messages"] = messages
        client_body.pop("tools", None)
        return

    lines = [
        "# Tools\n",
        "You have access to the following functions:\n",
        "<tools>"
    ]
    for t in tools:
        fn = t.get("function", {})
        lines.append(json.dumps({"type": "function", "function": fn}, ensure_ascii=False))
    lines.append("</tools>\n")
    lines.append("If you choose to call a function ONLY reply in the following format with NO suffix:\n")
    lines.append(
        "<tool_call>\n<function=example_function_name>\n"
        "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
        "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
        "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n"
    )
    lines.append(prompts.get("tool_calling_rules") + "\n\n")

    tools_str = "\n".join(lines)

    messages = client_body.get("messages") or []
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = tools_str + str(messages[0].get("content", ""))
    else:
        messages.insert(0, {"role": "system", "content": tools_str})

    client_body["messages"] = messages
    if "tools" in client_body:
        del client_body["tools"]


# ---------------------------------------------------------------------------
# Build /completion request body
# ---------------------------------------------------------------------------

def build_chat_body(client_body: dict, config: "GatewayConfig", stream: bool,
                    guard_cuts: int = 0) -> dict:
    """The `/v1/chat/completions` body, for an upstream that only sees images there.

    WHY A SECOND TRANSPORT EXISTS AT ALL

    This gateway renders its own ChatML and posts it to `/completion`, which is
    what lets it control the prompt byte for byte. Images were built on the
    same endpoint because `llama-server-impl.dll` carries `multimodal_data`
    and `LLAMA_MEDIA_MARKER` -- read out of the binary, not measured, and that
    was the mistake. Measured on 2026-09-13 against builds b10869 AND b10937,
    with an image of the digits 7431, which is the one question a model cannot
    guess:

        /completion, multimodal_data beside the prompt ... 37 prompt tokens,
                                                           answered "1"
        /completion, prompt as the documented object ..... HTTP 400
        /v1/chat/completions with image_url .............. 243 prompt tokens,
                                                           answered "7431"

    The server README documents `multimodal_data` as a field of a JSON OBJECT
    inside `prompt`; passing it beside the prompt is ignored in silence, which
    is why every earlier answer looked plausible -- the model guessed, and
    called a blue square red. The documented object form returns "Failed to
    tokenize prompt" for every variant tried, while the same object with no
    media works. So on these builds the chat endpoint is the only one that
    sees an image.

    WHAT THIS COSTS, AND WHAT IT DELIBERATELY DOES NOT

    llama.cpp applies the model's own chat template here, so the gateway gives
    up control of the exact prompt bytes for vision turns. It does NOT give up
    what the model is shown: every message is rendered through the same
    helpers as `format_chatml` -- the same `<tool_call>` dialect, the same
    `<tool_response>` wrappers, the same neutralization -- and handed over as
    message CONTENT. The model therefore reads the identical conversation; only
    the envelope differs.

    Vision models take this path and nothing else does, so a text rig keeps
    the byte-exact prompt and the warm KV prefix it has always had.
    """
    from gateway.adaptive import compute_adaptive_params

    messages = client_body.get("messages", [])
    sampling = compute_adaptive_params(
        messages, config.get_sampling_params(client_body), n_ctx=config.n_ctx,
        guard_cuts=guard_cuts, temperature_floor=config.temperature_floor)

    chat_messages: list[dict] = []
    index = 0
    while index < len(messages):
        msg = messages[index]
        role = msg.get("role", "")

        if role == "tool":
            group = [msg]
            index += 1
            while index < len(messages) and messages[index].get("role") == "tool":
                group.append(messages[index])
                index += 1
            # Tool results reach the model as a USER turn, exactly as on the
            # ChatML path: the chat template has a `tool` role of its own, and
            # letting it render one would show the model a shape its tool
            # instructions never described.
            chat_messages.append(_chat_message(
                "user", render_tool_results(group).rstrip("\n"), group))
            continue

        if role == "assistant":
            rendered = ""
            reasoning = _neutralize_wrappers(
                msg.get("reasoning_content") or msg.get("reasoning") or "", "think")
            if reasoning:
                rendered += f"<think>\n{reasoning}\n</think>\n"
            content = _neutralize_control_tokens(msg.get("content") or "")
            if content:
                rendered += f"{content}\n"
            rendered += render_tool_calls(msg)
            chat_messages.append(
                _chat_message("assistant", rendered.rstrip("\n"), [msg]))
        else:
            chat_messages.append(_chat_message(
                role or "user",
                _neutralize_control_tokens(msg.get("content", "")), [msg]))
        index += 1

    body = {
        "model": client_body.get("model", "unknown"),
        "messages": chat_messages,
        "stream": stream,
        "max_tokens": capped_n_predict(client_body, config),
        **sampling,
    }
    images = sum(1 for m in chat_messages if isinstance(m.get("content"), list))
    if images:
        _log.info("MULTIMODAL: %d message(s) carry an image (chat transport)",
                  images)
    return body


def _chat_message(role: str, text: str, sources: list[dict]) -> dict:
    """One chat message, with its images restored as OpenAI content parts.

    The payloads live on `_gateway_media`, put there by normalization so the
    rest of the pipeline could treat content as a plain string. Here they go
    back into the shape the chat endpoint reads -- the only place in the
    gateway that needs them in that form.
    """
    media = [payload for src in sources for payload in media_of(src)]
    if not media:
        return {"role": role, "content": text}
    parts: list[dict] = []
    if text:
        parts.append({"type": "text", "text": text})
    for payload in media:
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:image/png;base64,{payload}"}})
    return {"role": role, "content": parts}


def capped_n_predict(client_body: dict, config: "GatewayConfig") -> int:
    """How many tokens this turn may generate.

    The client's number is a request, not a licence. OpenCode asks for 32000;
    a turn that loops then owns the single upstream slot for the better part
    of an hour. The ceiling is generous enough that no legitimate turn reaches
    it -- measured, 49 turns in 6 834 did.

    Extracted so the two transports cannot disagree about it. They already
    disagree about the field NAME -- `/completion` calls it `n_predict` and
    `/v1/chat/completions` calls it `max_tokens` -- and a second copy of the
    capping rule behind two different names is how one of them would quietly
    stop capping.
    """
    max_tok = client_body.get("max_tokens") or client_body.get("max_completion_tokens")
    n_predict = int(max_tok) if max_tok else config.default_n_predict
    ceiling = getattr(config, "max_n_predict", 0)
    if ceiling and n_predict > ceiling:
        _log.info("n_predict capped: client asked %d, ceiling %d", n_predict, ceiling)
        n_predict = ceiling
    return n_predict


def build_generate_body(client_body: dict, config: "GatewayConfig", stream: bool,
                        guard_cuts: int = 0) -> dict:
    """Build the /completion request body for llama-server.

    v2 change: NO stop tokens — llama-server uses native EOS token detection.
    This eliminates the stop-token collision where <|im_end|> in file content
    would prematurely truncate generation.

    v2.1: Adaptive sampling — adjusts temperature based on conversation patterns
    (retry loops, edit failures, context pressure).
    """
    from gateway.adaptive import compute_adaptive_params

    messages = client_body.get("messages", [])
    sampling = config.get_sampling_params(client_body)

    # Apply adaptive sampling adjustments
    sampling = compute_adaptive_params(messages, sampling, n_ctx=config.n_ctx,
                                       guard_cuts=guard_cuts,
                                       temperature_floor=config.temperature_floor)

    body = {
        "model": client_body.get("model", "unknown"),
        "prompt": render_prompt(messages,
                                getattr(config, "prompt_format", "chatml"),
                                getattr(config, "thinking", False),
                                getattr(config, "tool_dialect", "qwen_xml"),
                                getattr(config, "reasoning", "")),
        "stream": stream,
        # v2: NO "stop" field — rely on native EOS token
        **sampling,
    }

    # Images. llama-server reads them from `multimodal_data`, positionally
    # against the `<__media__>` markers the renderer just wrote, so both come
    # from one walk over one list and cannot be assembled in different orders.
    #
    # The payloads are deliberately NOT in the prompt. In the prompt they
    # would be tokenized, counted against the prune limit, compared for KV
    # prefix stability and written to any full-prompt dump -- a megabyte of
    # base64 doing all four. A marker does none of it.
    media = collect_media(messages)
    if media:
        body["multimodal_data"] = media
        _log.info("MULTIMODAL: %d image(s) attached to this prompt", len(media))
    # `n_predict` here, `max_tokens` on the chat transport — same rule, one
    # definition, so the two names cannot drift into two policies.
    body["n_predict"] = capped_n_predict(client_body, config)

    if "repeat_penalty" in client_body:
        body["repeat_penalty"] = client_body["repeat_penalty"]

    # response_format, as the `json_schema` field BOTH engines take: llama.cpp
    # builds a grammar from it, and install/exllamav3/server.py hands it to
    # llguidance, which is already running there for tool-call syntax. One
    # body, so this function never has to learn which upstream it is talking
    # to. Until 2026-09-12 the field was read by nothing -- a client asking for
    # JSON got prose and could not tell that from a model that failed to
    # comply -- and until 2026-09-15 it was refused for every request this rig
    # serves, because one predicate answered both "takes a schema" and "takes
    # a GBNF grammar". `json_object` carries no schema of its own, so it
    # becomes the schema that accepts any object, which is what the OpenAI
    # field means.
    #
    # client_params.check() has already refused the request if the upstream
    # cannot be handed a schema, so reaching here means the field will be
    # honoured rather than ignored a second time.
    from gateway import client_params

    schema = client_params.json_schema_for(client_body)
    if schema is not None:
        body["json_schema"] = schema

    return body


# ---------------------------------------------------------------------------
# Context pruning
# ---------------------------------------------------------------------------

def _make_msg_cache_key(msg: dict) -> str:
    """Create a cache key for message token counts."""
    if not msg:
        return ""
    content = msg.get("content") or ""
    role = msg.get("role") or ""
    tool_calls = msg.get("tool_calls") or []
    tc_str = json.dumps(tool_calls, sort_keys=True)
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    # The image count is part of the identity of a message. Without it two
    # user turns whose text is identical -- "co tu jest?" twice, over two
    # different screenshots -- share a cache entry, and the second one is
    # costed from the first one's rendering.
    media = len(media_of(msg))
    return f"{role}:{content}:{tc_str}:{reasoning}:{media}"


async def estimate_message_tokens(msg: dict, upstream: "UpstreamClient",
                                  tokenizer_url: str | None = None) -> int:
    """Estimate the token count of a single message as ChatML.

    Caching lives in UpstreamClient.count_tokens, which keys a bounded LRU on
    the rendered text. This function used to keep a second, module-global dict
    on top of that — process-wide mutable state shared across every request and
    every config, wiped wholesale on reaching 5000 entries. It only saved a
    string concatenation; the HTTP call it was meant to avoid was already
    cached one layer down.
    """
    if not msg:
        return 0

    chatml = format_chatml([msg])
    suffix = "<|im_start|>assistant\n"
    if chatml.endswith(suffix):
        chatml = chatml[: -len(suffix)]
    return await upstream.count_tokens(chatml, base_url=tokenizer_url)



# Opening token of the note that stands in for evicted turns.
COMPACTION_MARKER = "[GATEWAY COMPACTION]"

# Files named in the note. Enough to recognise where the work was happening,
# short enough not to become bulk of its own.
_NOTE_MAX_FILES = 8


def _compaction_note(dropped: list[list[dict]]) -> str | None:
    """Summarize evicted turns as a fact, deterministically.

    Without this, eviction is silent: fifteen turns ago the model tried an
    `edit` on app.py, got "Could not find oldString" twice and moved on — and
    once those turns are gone it comes back and tries the identical edit,
    because nothing in the history says it failed. Today the circuit breaker
    catches that, but only after three more wasted turns.

    No model is involved. Summarizing with the LLM would occupy the single
    upstream slot in the middle of pruning and can hang; counting tool calls
    gives most of the value for none of the risk. Being a pure function of the
    dropped blocks also means the note is byte-identical whenever the same
    turns are dropped, which is what keeps it from churning the KV prefix: on
    turns where nothing new is evicted the note does not change, and on turns
    where something is evicted the prefix had already moved.
    """
    from gateway.lodestar.state import response_indicates_tool_error
    from gateway.lodestar.tool_names import get_path, parse_args

    messages = [m for block in dropped for m in block]
    if not messages:
        return None

    tool_counts: dict[str, int] = {}
    files: list[str] = []
    call_paths: dict[str, str] = {}
    errors: list[str] = []

    for m in messages:
        for call in m.get("tool_calls") or []:
            func = call.get("function") or {}
            name = str(func.get("name") or "?")
            tool_counts[name] = tool_counts.get(name, 0) + 1
            path = get_path(parse_args(func))
            if path:
                if path not in files:
                    files.append(path)
                cid = call.get("id")
                if cid:
                    call_paths[str(cid)] = path
        if m.get("role") == "tool" and response_indicates_tool_error(
                str(m.get("content") or "")):
            where = call_paths.get(str(m.get("tool_call_id") or ""))
            errors.append(where or "unknown target")

    lines = [
        f"{COMPACTION_MARKER} {len(messages)} earlier messages were removed to "
        "fit the context window.",
        "This is a record of what already happened, not an instruction.",
    ]
    if tool_counts:
        ordered = sorted(tool_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        lines.append("Tools used: " + ", ".join(f"{n}×{c}" for n, c in ordered))
    if files:
        shown = ", ".join(files[:_NOTE_MAX_FILES])
        if len(files) > _NOTE_MAX_FILES:
            shown += f", +{len(files) - _NOTE_MAX_FILES} more"
        lines.append("Files touched: " + shown)
    if errors:
        by_target: dict[str, int] = {}
        for target in errors:
            by_target[target] = by_target.get(target, 0) + 1
        ordered_err = sorted(by_target.items(), key=lambda kv: (-kv[1], kv[0]))
        lines.append("Failed tool calls: " + ", ".join(
            f"{c}× on {t}" for t, c in ordered_err))
    return "\n".join(lines)


def _pinned_block_index(blocks: list[list[dict]], messages: list[dict]) -> int | None:
    """Index of the block holding the current task's opening user message.

    "Current task" is the LODESTAR cycle: a closing write ends one, the next
    substantive user message starts the next. Pinning the request of the
    *previous* task would protect a stale instruction, so the boundary the
    state machine already computes is the right one to reuse here.

    Falls back to the first user message overall when no cycle boundary
    applies, which is also the answer for clients that never trigger LODESTAR.
    """
    from gateway.lodestar.state import effective_messages

    try:
        current = effective_messages(messages)
    except Exception:  # pragma: no cover - state derivation must never break pruning
        current = messages

    target = None
    for m in current:
        if m.get("role") == "user":
            target = m
            break
    if target is None:
        return None

    for i, block in enumerate(blocks):
        for m in block:
            if m is target:
                return i
    return None


async def prune_messages(
    messages: list[dict],
    upstream: "UpstreamClient",
    max_tokens: int = 120_000,
    audit=None,
    request_id: str = "",
    session_id: str | None = None,
    compaction: bool = True,
    image_tokens: int = 0,
    tokenizer_url: str | None = None,
) -> list[dict]:
    """Context overflow guard: prune message history to fit within max_tokens.

    Strategy: keep the system message, pin the current task's opening request,
    and keep as many recent conversational blocks as fit within max_tokens.
    Tool-call + tool-response sequences are treated as atomic blocks (never
    split mid-tool-use). Whatever is evicted leaves a factual note behind
    unless `compaction` is off.

    `image_tokens` is what one image is assumed to cost. Every other count
    here comes from rendering the message and asking /tokenize -- and after
    normalization an image renders as the thirteen characters of
    `<__media__>`, so without this a screenshot is costed at four tokens and
    really costs hundreds to thousands. That is the prune limit being wrong in
    the one direction it must never be wrong in, in the one place nothing
    would think to look. 0 keeps the old behaviour for a text-only rig, where
    there are no images to charge for anyway.
    """
    if not messages:
        return []

    system_msg = None
    start_idx = 0
    if messages[0].get("role") == "system":
        system_msg = messages[0]
        start_idx = 1

    # Group into conversational blocks
    blocks: list[list[dict]] = []
    i = start_idx
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            block = [msg]
            i += 1
            while i < len(messages) and messages[i].get("role") == "tool":
                block.append(messages[i])
                i += 1
            blocks.append(block)
        else:
            blocks.append([msg])
            i += 1

    # Gather all messages for parallel token counting
    flat_messages: list[dict] = []
    if system_msg:
        flat_messages.append(system_msg)
    for block in blocks:
        flat_messages.extend(block)

    tasks = [estimate_message_tokens(m, upstream, tokenizer_url)
             for m in flat_messages]
    token_counts = await asyncio.gather(*tasks)

    msg_tokens_map = {
        _make_msg_cache_key(m): cnt for m, cnt in zip(flat_messages, token_counts, strict=False)
    }

    def get_msg_tokens(m: dict) -> int:
        counted = msg_tokens_map.get(_make_msg_cache_key(m),
                                     len(m.get("content") or "") // 3)
        return counted + image_tokens * len(media_of(m))

    system_tokens = get_msg_tokens(system_msg) if system_msg else 0
    block_tokens = [sum(get_msg_tokens(m) for m in block) for block in blocks]

    # The user's opening request is the oldest message of the cycle, so it is
    # the first thing eviction reaches — and losing it is not a small loss.
    # The Goal in active-context survives (LODESTAR writes it to a file), but
    # the Goal is one sentence: the constraints that came with the request
    # ("don't touch gui.py", "follow the pattern in parsers/") live only here.
    # Dropping them is why a long session drifts into doing something the user
    # never asked for.
    pinned_idx = _pinned_block_index(blocks, messages)

    kept_idx: set[int] = set()
    current_tokens = system_tokens
    if pinned_idx is not None:
        kept_idx.add(pinned_idx)
        current_tokens += block_tokens[pinned_idx]

    kept_recent = False
    for i in range(len(blocks) - 1, -1, -1):
        if i in kept_idx:
            continue
        if not kept_recent or (current_tokens + block_tokens[i] <= max_tokens):
            kept_idx.add(i)
            current_tokens += block_tokens[i]
            kept_recent = True
        else:
            _log.info(
                "Context overflow guard: pruning block of %d msgs (tokens: %d). Kept: %d",
                len(blocks[i]), block_tokens[i], current_tokens,
            )
            break

    # Hard context boundary guard: ensure total prompt tokens never exceed max_tokens.
    # If the recent turn or pinned block pushed current_tokens over max_tokens,
    # shed excess tokens to prevent downstream server context exhaustion.
    if current_tokens > max_tokens:
        # Step 1: If pinned block pushed it over, unpin it first
        if pinned_idx is not None and pinned_idx in kept_idx:
            _log.warning(
                "Context overflow guard: dropping pinned block (%d tokens) to stay within limit (%d/%d)",
                block_tokens[pinned_idx], current_tokens, max_tokens,
            )
            kept_idx.remove(pinned_idx)
            current_tokens -= block_tokens[pinned_idx]
            pinned_idx = None

        # Step 2: If still over max_tokens, trim large tool results in the kept blocks
        if current_tokens > max_tokens and blocks:
            for idx in sorted(kept_idx, reverse=True):
                block = blocks[idx]
                tool_msgs = [m for m in block if m.get("role") == "tool" and isinstance(m.get("content"), str)]
                tool_msgs.sort(key=lambda m: len(m.get("content", "")), reverse=True)
                for tm in tool_msgs:
                    c = tm.get("content", "")
                    if len(c) > 1500 and current_tokens > max_tokens:
                        excess_chars = (current_tokens - max_tokens) * 3
                        trim_target = min(len(c) - 800, max(500, excess_chars))
                        keep_len = max(600, len(c) - trim_target)
                        head_len = keep_len // 2
                        tail_len = keep_len - head_len
                        trimmed_c = (
                            f"{c[:head_len]}\n\n"
                            f"[... content truncated by gateway context guard to fit token budget ...]\n\n"
                            f"{c[-tail_len:]}"
                        )
                        saved_tokens = (len(c) - len(trimmed_c)) // 3
                        tm["content"] = trimmed_c
                        current_tokens = max(0, current_tokens - saved_tokens)
                        _log.info(
                            "Context overflow guard: trimmed tool result from %d to %d chars (~%d tokens saved)",
                            len(c), len(trimmed_c), saved_tokens,
                        )
                    if current_tokens <= max_tokens:
                        break
                if current_tokens <= max_tokens:
                    break

    # What was evicted leaves a note behind. It is added after the budget has
    # been spent, so it costs a couple of hundred characters over max_tokens —
    # cheap next to the retry loop it prevents.
    note = None
    if compaction:
        dropped = [blocks[i] for i in range(len(blocks)) if i not in kept_idx]
        note = _compaction_note(dropped)
    note_msg: dict | None = {"role": "user", "content": note} if note else None

    pruned: list[dict] = []
    if system_msg:
        pruned.append(system_msg)
    if note_msg is not None and pinned_idx is None:
        # Nothing is pinned, so the note precedes everything that survived.
        pruned.append(note_msg)
        note_msg = None
    # Original order: a pinned block that survived an eviction round still
    # belongs where it was, not hoisted to the front of the conversation.
    for i in sorted(kept_idx):
        pruned.extend(blocks[i])
        if note_msg is not None and i == pinned_idx:
            # Directly after the task it belongs to, always the same place.
            pruned.append(note_msg)
            note_msg = None
    if note_msg is not None:
        pruned.append(note_msg)

    if audit is not None and len(pruned) != len(messages):
        audit.log_prune(
            request_id=request_id,
            original_count=len(messages),
            pruned_count=len(messages) - len(pruned),
            tokens_before=sum(get_msg_tokens(m) for m in flat_messages),
            tokens_after=current_tokens,
            session_id=session_id,
        )

    return pruned
