"""LODESTAR state determination - pure, side-effect-free."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from enum import IntEnum

from gateway.lodestar import client_contract

from gateway.lodestar.tool_names import (
    READ_TOOLS as _READ_TOOLS,
    EDIT_TOOLS as _EDIT_TOOLS,
    SHELL_TOOLS as _SHELL_TOOLS,
    PATH_KEYS as _PATH_KEYS,
    CMD_KEYS as _CMD_KEYS,
    ACTIVE_CONTEXT as _ACTIVE_CONTEXT_PATH,
    EMBED_SCRIPT as _EMBED_SEARCH_BIN,
    get_path as _get_path,
    get_cmd as _get_cmd,
    parse_args as _parse_args,
    is_read_active_context_call as _is_active_context_read,
    is_edit_active_context_call as _is_active_context_edit,
    is_embed_search_call as _is_embed_search,
)

_log = logging.getLogger("gateway.lodestar.state")


class LodestarState(IntEnum):
    READ = 1      # Waiting for read of active-context + semantic search
    UPDATE = 2    # Read+search done, waiting for edit of active-context
    READY = 3     # All phases complete, model free to act


def response_indicates_context_read(content: str) -> bool:
    """True if a tool response contains the actual active-context.md content.

    Strict on purpose: matching on the mere substring "active-context.md"
    false-positives on the LODESTAR.md read (its text mentions the file),
    which used to advance the state machine before the file was ever read.
    The active-context template always starts with the "# Active Context"
    heading, so require that.
    """
    return client_contract.active_context_heading() in content.lower()


def response_indicates_embed_search(content: str) -> bool:
    """True if a tool response looks like embed.py search output."""
    if "[lodestar lesson annotation]" in content.lower():
        return True
    return bool(client_contract.search_result_re().search(content))


def response_indicates_tool_error(content: str) -> bool:
    """True if a tool response is a failure wording the client is known to use.

    False does NOT mean success — it means "not a wording we recognise". Callers
    deciding whether an edit landed must prefer write_succeeded(), which checks
    the file itself; treating an unrecognised response as success is what let a
    failed edit advance the state machine (2026-07-09 13:24).
    """
    return client_contract.is_known_error_marker(content)


def write_succeeded(args: dict, content: str, workspace_dir: str | None) -> bool:
    """Did an active-context write/edit actually change the file?

    Order of evidence:
      1. A known client error wording — the client says it failed, believe it.
      2. The file on disk — does it now contain what the call claimed to write?
         This is ground truth and depends on no wording at all.
      3. Neither available — assume success, as before, but say so in the log:
         an unrecognised response is exactly how a client change would silently
         reintroduce the phantom-Goal bug, and it should be visible when it
         happens rather than after the next incident.
    """
    if response_indicates_tool_error(content):
        return False

    landed = client_contract.write_landed_on_disk(args, workspace_dir)
    if landed is True:
        return True
    if landed is False:
        _log.warning(
            "LODESTAR: active-context write reported no known error but the file "
            "does not contain what it claimed to write — treating it as failed. "
            "Response was: %r", (content or "")[:200],
        )
        return False

    _log.debug(
        "LODESTAR: cannot verify an active-context write on disk (no reachable "
        "workspace); trusting the response. Response was: %r", (content or "")[:200],
    )
    return True


# Keys that carry the NEW file content in a write/edit call. oldString must
# be ignored: an edit replacing "W toku" -> "Zakończono" has the old status
# in oldString and would mask the closure.
_NEW_CONTENT_KEYS = client_contract.new_content_keys()

# Template placeholder line kept verbatim by models: "- [W toku / Zakończono / ...]"
_PLACEHOLDER_LINE_RE = re.compile(r"^-?\s*\[.*\]\s*$")
_STATUS_SECTION_RE = re.compile(
    r"##\s*Status\s*\n(.*?)(?=\n##|\n---|\Z)", re.DOTALL | re.IGNORECASE)


def _write_indicates_closure(args: dict) -> bool:
    """True if an active-context write/edit sets Status to Zakończono.

    Careful parsing: the template's placeholder line contains the word
    "Zakończono" inside brackets, and models routinely keep it — only
    non-placeholder Status lines count. "W toku" wins over "Zakończono"
    when both appear (ambiguous → treat as still open).
    """
    parts = [str(args[k]) for k in _NEW_CONTENT_KEYS
             if isinstance(args.get(k), str)]
    text = "\n".join(parts) if parts else "\n".join(
        str(v) for v in args.values() if isinstance(v, str))
    return status_text_says_closed(text)


def status_text_says_closed(text: str) -> bool:
    """Does this text's `## Status` section report the task finished?

    Shared by the two readers that ask it -- the arguments of a write, and
    active-context.md itself -- because a second implementation would have to
    re-learn both traps this one already knows. **The template's placeholder
    line contains the closed word inside brackets**, so a fresh workspace
    would otherwise read as finished and the Phase D nudge would never fire
    on the projects that need it most. And "W toku" beats "Zakończono" when
    both appear: an ambiguous Status block is a file mid-edit, and the
    conservative answer is that the cycle is still open.
    """
    m = _STATUS_SECTION_RE.search(text)
    body = m.group(1) if m else text
    real_lines = [ln.strip() for ln in body.splitlines()
                  if ln.strip() and not _PLACEHOLDER_LINE_RE.match(ln.strip())]
    joined = " ".join(real_lines)
    if client_contract.status_open_re().search(joined):
        return False
    return bool(client_contract.status_closed_re().search(joined))


def _closure_boundary(messages: list[dict]) -> int:
    """Index of the first message AFTER the last closing active-context write."""
    boundary = 0
    for i, m in enumerate(messages):
        for call in m.get("tool_calls") or []:
            func = call.get("function") or {}
            name = func.get("name", "")
            args = _parse_args(func)
            if _is_active_context_edit(name, args) and _write_indicates_closure(args):
                boundary = i + 1
    return boundary


def _has_new_task_after(messages: list[dict], start: int) -> bool:
    """A substantive user request after `start` (short acks don't count)."""
    from gateway.lodestar.context_injector import clean_user_text
    for m in messages[start:]:
        if m.get("role") == "user":
            if len(clean_user_text(m.get("content"))) >= 20:
                return True
    return False


def effective_messages(messages: list[dict]) -> list[dict]:
    """Slice history to the current LODESTAR cycle (per-task re-arm).

    The protocol used to be one-shot per thread: once read/search/write
    appeared anywhere in history, every follow-up request ("fix the chart")
    ran ungated in state 3 — no Goal update, no memory search, no lesson.
    A write that sets Status=Zakończono closes the cycle; the next
    substantive user message starts a fresh one, so only messages after the
    closure count toward the state machine. Short acknowledgements
    ("dzięki", "ok") do not re-arm the protocol.
    """
    boundary = _closure_boundary(messages)
    if not boundary or boundary >= len(messages):
        return messages
    if _has_new_task_after(messages, boundary):
        return messages[boundary:]
    return messages


# Marker used by the Phase D nudge (app.py synthesizes the echo call, the
# normalizer neutralizes it once finalization is done). Shared here to avoid
# a normalizer→app import cycle.
PHASE_D_MARKER = "[LODESTAR PHASE D]"


def active_context_says_closed(workspace_dir: str | None) -> bool:
    """Does active-context.md itself report the task as finished?

    Ground truth for the one question the Phase D nudge asserts out loud, and
    the same evidence `write_succeeded` already prefers over any wording: the
    file on disk.

    Parsed by `status_text_says_closed`, the same function that reads the
    arguments of a write -- which is what keeps this from re-learning the two
    traps that parser already handles. **Only the `## Status` section counts.**
    The Progress Log is a list of FINISHED tasks and is full of the closed
    word; testing the whole file would call every project with any history
    closed. An unreadable or absent file answers False -- not knowing must
    never be what silences a guard.
    """
    if not workspace_dir:
        return False
    try:
        path = Path(workspace_dir) / ".context" / "memory" / "active-context.md"
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False

    return status_text_says_closed(text)


def phase_d_pending(messages: list[dict], workspace_dir: str | None = None) -> bool:
    """True if the current cycle committed a Goal but was never closed.

    **Asked of the FILE first, when the caller can reach it.** Measured
    2026-09-14: the nudge fired on a project whose active-context.md said
    `Status: Zakończono`, and its text told the model the file "still says W
    toku" -- an assertion the decision had never checked. The history-only
    answer had been fooled by ordering: the model set the status closed in one
    edit and appended its Progress Log entry in a LATER one, and an edit after
    the closing write that does not itself close anything reads, to
    `_closure_boundary`, as a cycle re-opened.

    Anything derived from the history the client sends can also be erased by
    compaction, which is a second route to the same false positive on exactly
    the long tasks that get compacted.

    `workspace_dir` is optional and the history-only answer survives without
    it. `normalizer._neutralize_completed_nudges` passes None deliberately:
    its use is the INVERSE -- it defuses a stale nudge when this returns False
    -- so a conservative "still pending" there costs one extra turn of dead
    text and can never invent a nudge.

    Used by the Phase-D nudge: when the model ends a turn with a plain-text
    final answer while active-context still says "W toku", the gateway
    reminds it to save the lesson and close the context (2026-07-09 21:57
    session: the follow-up fix ended with no lesson and an open Goal).

    "Committed" means the same thing here as it does in determine_state: an
    edit whose response reported an error changed nothing, so it does not open
    a cycle. Counting failed edits made this the only place in the module that
    skipped the 2026-07-09 13:24 fix, and it would have told the model to
    close out a task whose Goal was never written.
    """
    if active_context_says_closed(workspace_dir):
        return False  # the file itself says the task is finished

    msgs = effective_messages(messages)
    if _closure_boundary(msgs):
        return False  # current cycle already closed

    pending_edits: dict[str, dict] = {}
    for m in msgs:
        for call in m.get("tool_calls") or []:
            func = call.get("function") or {}
            args = _parse_args(func)
            if _is_active_context_edit(func.get("name", ""), args):
                cid = call.get("id")
                if not cid:
                    return True  # no id to correlate — assume it landed
                pending_edits[cid] = args
        if m.get("role") == "tool":
            cid = m.get("tool_call_id")
            if cid in pending_edits:
                args = pending_edits.pop(cid)
                if write_succeeded(args, str(m.get("content") or ""), workspace_dir):
                    return True  # a Goal really was written and never closed

    # Edits still awaiting a response: assume success, same as determine_state.
    return bool(pending_edits)


def context_supplied_by_gateway(workspace_dir: str | None) -> bool:
    """True when the gateway can hand the model active-context.md itself.

    Phase A used to spend a whole round trip — prefill, a generated tool call,
    the client executing it, another prefill with the response — on reading a
    file the gateway already reads for its own checks. The file's bytes end up
    in the context either way, so injecting them costs strictly less than
    asking for them.

    Returns False when there is no reachable workspace or no file, which is
    what keeps the eval harness and the pure-function tests on the old
    history-based path.
    """
    if not workspace_dir:
        return False
    try:
        path = Path(workspace_dir) / ".context" / "memory" / "active-context.md"
        return path.is_file() and bool(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False


def active_context_write_landed(messages: list[dict],
                                workspace_dir: str | None = None) -> bool:
    """Has a write to active-context.md SUCCEEDED in the current cycle?

    Not "was one attempted": the same `write_succeeded` ladder the state
    machine uses — a known client error wording loses, the file on disk wins,
    and an unrecognised response is assumed to have landed.

    It exists to tell two situations apart that look identical from the
    outside. A reminder the model has not acted on yet is a model that is
    still working; a reminder it has been given, and then written the very
    file the reminder is about without doing what it asked, is a model that
    has had its chance. Only the second is worth saying anything sharper
    about, and this is the question that separates them.

    Scoped to the current cycle by `effective_messages`, like everything else
    here: a write from a task that closed three turns ago says nothing about
    this one.
    """
    if not messages:
        return False
    pending: dict[str, dict] = {}
    for m in effective_messages(messages):
        for call in m.get("tool_calls") or []:
            func = call.get("function") or {}
            args = _parse_args(func)
            if not _is_active_context_edit(func.get("name", ""), args):
                continue
            cid = call.get("id")
            if not cid:
                return True          # no id to correlate — assume it landed
            pending[cid] = args
        if m.get("role") == "tool":
            cid = m.get("tool_call_id")
            if cid in pending:
                args = pending.pop(cid)
                if write_succeeded(args, str(m.get("content") or ""),
                                   workspace_dir):
                    return True
    # An edit still awaiting its response is assumed to have landed, which is
    # what `determine_state` does with the same shape.
    return bool(pending)


def determine_state(messages: list[dict], is_subagent: bool = False,
                    workspace_dir: str | None = None) -> LodestarState:
    """Current protocol state, derived from history.

    `workspace_dir` lets the Phase-B check verify the write on disk instead of
    guessing from the client's error wording. Without it the function stays
    pure — which is what the eval harness and most tests rely on — and falls
    back to matching prose.
    """
    if is_subagent:
        return LodestarState.READY
    if not messages:
        return LodestarState.READ
    messages = effective_messages(messages)

    # The gateway hands the model active-context.md in the turn frame, so the
    # read is satisfied before the model does anything.
    has_read = context_supplied_by_gateway(workspace_dir)
    has_searched = False
    has_written = False
    pending_edits: dict[str, dict] = {}

    for m in messages:
        for call in m.get("tool_calls") or []:
            func = call.get("function") or {}
            name = func.get("name", "")
            args = _parse_args(func)
            if _is_active_context_read(name, args):
                has_read = True
            elif _is_active_context_edit(name, args):
                cid = call.get("id")
                if cid:
                    # Count only after the tool response confirms success.
                    pending_edits[cid] = args
                else:
                    has_written = True
            elif _is_embed_search(name, args):
                has_searched = True

        if m.get("role") == "tool":
            content = str(m.get("content") or "")
            if response_indicates_context_read(content):
                has_read = True
            if response_indicates_embed_search(content):
                has_searched = True
            cid = m.get("tool_call_id")
            if cid in pending_edits:
                args = pending_edits.pop(cid)
                if write_succeeded(args, content, workspace_dir):
                    has_written = True

    # Edits with no response in history yet: assume success (optimistic) —
    # only an explicit failure demotes the step.
    if pending_edits:
        has_written = True

    if has_written and has_searched and has_read:
        return LodestarState.READY
    if has_searched and has_read:
        return LodestarState.UPDATE
    return LodestarState.READ


# ---------------------------------------------------------------------------
# Cycle size — the closest the gateway can get to counting agent iterations
# ---------------------------------------------------------------------------

# Synthetic call ids all start with this. Every place that fabricates a call
# tags it: syn_, syn_stub_, syn_phd_.
_SYNTHETIC_ID_PREFIX = "syn_"


def cycle_turn_count(messages: list[dict]) -> int:
    """Assistant turns spent on the current LODESTAR cycle.

    The agent loop itself lives in the client, so the gateway never sees an
    iteration counter — but it does see the whole history, and the number of
    assistant turns since the cycle opened is the same measure by another name.
    """
    return sum(1 for m in effective_messages(messages)
               if m.get("role") == "assistant")


# Stages of the cycle turn budget, mirroring the subagent one. Two, because
# the first alone is only a request and this model reads guidance and carries
# on. Unlike a subagent's, this hard stop is recoverable by the person at the
# keyboard: the turn ends with a plain answer, and their next message opens a
# new cycle, which resets the count.
CYCLE_SOFT = "soft"
CYCLE_HARD = "hard"


def cycle_budget_state(messages: list[dict], budget: int,
                       grace: int) -> tuple[str | None, int]:
    """(None | CYCLE_SOFT | CYCLE_HARD, assistant turns in this cycle).

    **"In this cycle" is doing the work in that sentence, and it depends on
    the protocol running.** The count comes from `effective_messages`, which
    slices the history at a tool call that writes a closing Status into
    active-context.md -- so with LODESTAR switched off no boundary is ever
    laid down, every assistant turn in the conversation counts, and this
    reports a session odometer rather than a task's cost. The enforcement in
    `chat_completions` therefore asks `lodestar_enabled` before it asks this;
    the function itself stays pure and answers what it is asked.
    """
    turns = cycle_turn_count(messages)
    if budget <= 0:
        return None, turns
    if turns >= budget + max(0, grace):
        return CYCLE_HARD, turns
    if turns >= budget:
        return CYCLE_SOFT, turns
    return None, turns


def synthesis_count(messages: list[dict]) -> int:
    """Tool calls the gateway has fabricated during the current cycle.

    A healthy cycle needs at most three (read, search, Goal write). A number
    climbing past that means the model is not taking the protocol steps even
    when handed the results, and continuing to synthesize just burns the single
    upstream slot.
    """
    total = 0
    for m in effective_messages(messages):
        for call in m.get("tool_calls") or []:
            if str(call.get("id") or "").startswith(_SYNTHETIC_ID_PREFIX):
                total += 1
    return total
