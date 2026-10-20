"""LODESTAR single-channel context injection.

Replaces the previous three-channel approach (full LODESTAR.md content +
[SYSTEM CONTROL: STATE X] block + hardcoded rules in inject_tools_to_prompt).

This module is the ONLY place that injects LODESTAR instructions. The
inject_tools_to_prompt function in normalizer.py must not duplicate these.

v2.3 (2026-07-07): also detects when the user asks to use skills, and in
state READY appends an explicit "load skills NOW before implementing"
reminder. The 2026-07-07 Capacitor session showed the model skipping
skill loading and going straight to Write because the reminder was too
weak.

v2.5 (2026-08-31): injects `.context/project_rules.md` when the workspace
has one. The gateway reads it rather than making the model read it: the file
is small and static, so a tool call would cost a turn and a synthesis slot
for nothing, the model could skip it, and a tool response appears once and
then disappears with the next compaction. Injected content is in every
request of the cycle instead.
"""
from __future__ import annotations

import re
import secrets
from pathlib import Path

from gateway import prompts
from gateway.lodestar import client_contract
from gateway.lodestar.state import LodestarState, active_context_write_landed


_BLOCK_HEADER = "=== LODESTAR PROTOCOL ==="
_BLOCK_FOOTER = "=== END LODESTAR PROTOCOL ==="

# The volatile half, kept OUT of the system message. Everything above lives at
# the head of the prompt and must stay byte-identical for the whole task: the
# state directive, the checklist and the transient notices change between turns,
# and ~200 tokens of them sitting in front of the conversation invalidated
# llama-server's cached prefix, forcing a full re-prefill of 16-30k tokens on
# every phase change (measured 2026-08-31 from gateway_events.jsonl). They are
# emitted as the last frame before the assistant turn instead, which is also
# where a "your only allowed actions this turn" instruction belongs.
# normalizer._stabilize_system_prefix already does this for the client's own
# volatile headers; this is the same treatment for the gateway's.
NEWLINE = chr(10)

_TURN_HEADER = "=== LODESTAR TURN ==="
_TURN_FOOTER = "=== END LODESTAR TURN ==="

# Client-injected boilerplate blocks stripped before intent detection / goal
# extraction (OpenCode superpowers preamble etc.).
_PREAMBLE_BLOCK_RE = re.compile(
    r"<(EXTREMELY_IMPORTANT|IMPORTANT|system-reminder)>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)

# Slash-command skill expansion footer lines: the user's actual prompt
# follows AFTER these when OpenCode pastes SKILL.md into the message.
_SKILL_FOOTER_MARKERS = (
    "relative paths in this skill",
    "base directory for this skill",
)


def clean_user_text(content: object) -> str:
    """Strip client-injected boilerplate from a user message.

    Handles two OpenCode injection patterns:
      1. <EXTREMELY_IMPORTANT>...</EXTREMELY_IMPORTANT> superpowers preamble.
      2. Slash-command skill expansion (/brainstorming etc.): the whole
         SKILL.md is pasted into the message, ending with the footer lines
         above -- the user's actual prompt follows them.
    """
    if isinstance(content, list):
        content = "\n\n".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    if not isinstance(content, str):
        return ""
    text = _PREAMBLE_BLOCK_RE.sub("", content)
    low = text.lower()
    for marker in _SKILL_FOOTER_MARKERS:
        pos = low.rfind(marker)
        if pos != -1:
            line_end = text.find("\n", pos)
            text = text[line_end + 1:] if line_end != -1 else ""
            break
    return text.strip()


# User-request signals that mean "use skills before implementing".
# Matched against the CLEANED task text only -- the raw message always
# contains "skill" via the superpowers preamble, which made the detection
# fire for every task and turned the directive into background noise.
# Polish inflections (skilli, skille, skillami, umiejetnosci) are covered
# by stem matching.
_SKILL_REQUEST_PATTERNS = [
    # skill/skills + Polish inflections (skilli, skille, skilla, skillami...);
    # word boundary keeps "skillful"/"skilled" from matching.
    re.compile(r"\bskill(?:s|e|i|a|u|em|om|ami|ach|owi|ów|ow)?\b", re.IGNORECASE),
    re.compile(r"\bbrain-?storm", re.IGNORECASE),   # brainstorm / brain-storm
    re.compile(r"\bumiej[eę]tno", re.IGNORECASE),   # "uzyj umiejetnosci" (Polish)
]


def user_requested_skills(messages: list[dict]) -> bool:
    """Return True if the latest user message asks to use skills.

    Scans the CLEANED text of the most recent role='user' message (client
    boilerplate stripped) for skill-related keywords.
    """
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        text = clean_user_text(m.get("content"))
        if not text:
            return False
        for pat in _SKILL_REQUEST_PATTERNS:
            if pat.search(text):
                return True
        return False  # only the latest user message matters
    return False


def _read_lodestar(workspace: Path) -> str | None:
    path = workspace / ".context" / "LODESTAR.md"
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    # Resolve relative paths for the model's convenience. Forward slashes
    # work on Windows too (Python, bash tool, OpenCode). The previous global
    # "/" -> "\\" replacement corrupted non-path text in LODESTAR.md
    # (e.g. "read/view_file" became "read\view_file").
    content = content.replace(".context/", str(workspace / ".context").replace("\\", "/") + "/")
    return content


# Rules the user accumulated for one project. Capped on purpose: unlike a
# semantic lesson, which surfaces only when a search matches it and arrives
# annotated as a hint, a rule here is injected unconditionally into every
# request. A file that only ever grows therefore degrades every request in
# the workspace, so at the cap the model must replace a rule, not append one.
MAX_PROJECT_RULES = 15

# Progress Log entries kept verbatim in active-context.md before the model is
# asked to compress the older ones. The rule existed from the beginning in
# .context/rules/project-auto.md and in a checkbox inside active-context.md
# itself -- neither of which any model ever reads, so it never once fired
# (this repo's own log reached 7 entries). It now lives in LODESTAR.md and
# AGENTS.md, and the gateway counts the file on disk so the reminder appears
# only when it actually applies, naming the real number.
MAX_PROGRESS_ENTRIES = 5

_RULE_LINE_RE = re.compile(r"^\s*[-*]\s+\S")


def _display_path(path: Path) -> str:
    """Every path the gateway shows the model, in ONE shape: forward slashes.

    This used to return backslashes on Windows, on the reasoning that a path
    should look native. Measured 2026-09-01, that cost more than it bought.
    A single prompt showed the same directory eleven times in two styles — the
    client's `<env>` block and the user's own text in backslashes, LODESTAR.md
    and synthesized calls in forward slashes, and this function contributing to
    the backslash half — and the model, having to pick one when it emitted a
    path, corrupted it four times in two hours. Three of the four landed
    exactly on the separator:

        PROJEKTY\\ACME_HW -> PROJEKT            (truncated at it)
        PROJEKTY\\ACME_HW -> PROJEKTOACME_HW    (separator lost)
        PROJEKTY\\ACME_HW -> PROJEKTYPCME_HW    (separator lost)

    Forward slashes end the ambiguity the gateway itself was creating: Windows
    accepts them in every API and in both shells, they survive JSON and XML
    without doubling, and they are what the gateway's own synthesized calls
    have always used. What the client puts in its `<env>` block is not ours to
    change — but the gateway was contributing six of those eleven occurrences,
    in two styles, from one file.
    """
    return str(path).replace("\\", "/")


def _project_rules_path(workspace: Path) -> Path:
    return workspace / ".context" / "project_rules.md"


def _read_project_rules(workspace: Path) -> str | None:
    """Contents of `.context/project_rules.md`, or None when there are none."""
    path = _project_rules_path(workspace)
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return content or None


def _rule_lines(content: str) -> list[str]:
    """The bullet lines of a rules file -- the rules themselves.

    Everything else in the file (its heading, the notes for whoever maintains
    it) is written for humans and is deliberately NOT injected: it would cost
    tokens in every request of every cycle to repeat what the protocol already
    says, and the first draft of this feature shipped the whole file, which
    made the injected block twice the size of the rules it carried.
    """
    return [line.strip() for line in content.splitlines()
            if _RULE_LINE_RE.match(line)]


def count_project_rules(content: str) -> int:
    """Bullet lines in the rules file. Headings and prose do not count."""
    return len(_rule_lines(content))


def _project_rules_block(workspace: Path) -> str:
    """The [PROJECT RULES] block, or "" when the workspace has no rules file."""
    content = _read_project_rules(workspace)
    if content is None:
        return ""
    rules = _rule_lines(content)
    if not rules:
        return ""  # a file of prose with no rules in it
    path_s = _display_path(_project_rules_path(workspace))
    cap_note = ""
    if len(rules) >= MAX_PROJECT_RULES:
        cap_note = prompts.get("project_rules_full", max_rules=MAX_PROJECT_RULES)
    return prompts.get("project_rules_block", rules_path=path_s,
                       rules="\n".join(rules), cap_note=cap_note)


# How much of active-context.md the volatile frame may carry.
#
# The frame sits AFTER the newest user message, so it is displaced by every new
# turn and re-prefilled whole — that is by design, and it is why everything in
# it has to stay small. active-context.md is the one part a project can grow
# without bound: MAX_PROGRESS_ENTRIES caps the log at five ENTRIES, not bytes.
# Measured 2026-09-12 on this repository's own file — five entries and one
# compressed paragraph — 17,782 characters, against the ~735 that
# check_prefix_stability.py calls a healthy frame. About 4,500 tokens of
# re-prefill on every single turn, for history the turn does not use.
#
# 4096 comes from the file's shape rather than from taste. The part a resumed
# cycle actually needs — Goal, Status, Blockers, Next Steps — measured 348
# characters here and 598 in LODESTAR_template, and the median Progress Log
# entry measured 3,200. So the cap buys "where we are" plus the newest "what
# happened", and the rest stays on disk, one read away in the rare turn that
# wants it. 0 disables the cap.
DEFAULT_CONTEXT_MAX_CHARS = 4096

_PROGRESS_LOG_HEADING = "## Progress Log"
_ENTRY_SEPARATOR = NEWLINE + "- **"


def _clip_active_context(content: str, limit: int) -> str:
    """Hard cut, for a file with no Progress Log to shed.

    The note counts against the limit like everything else: appending it after
    cutting to `limit` is how a cap gets exceeded by the sentence announcing
    the cap. Its width is bounded by the numbers it prints, so the worst case
    is computed first and the text is cut to what is left.
    """
    reserve = len(prompts.get("active_context_clipped",
                              shown=limit, total=len(content))) + 2
    shown = content[:max(0, limit - reserve)].rstrip()
    return shown + NEWLINE + NEWLINE + prompts.get(
        "active_context_clipped", shown=len(shown), total=len(content))


def fit_active_context(content: str, limit: int) -> str:
    """active-context.md shortened to `limit` characters, newest entries kept.

    A pure function of (content, limit): the same file always renders the same
    bytes, which is what lets the frame be compared turn to turn at all.

    The head — everything above `## Progress Log` — is never dropped while
    there is a Progress Log to drop instead. It is the protocol-critical part
    and it is small; the log is history and it is not.
    """
    if limit <= 0 or len(content) <= limit:
        return content

    head, heading, log = content.partition(_PROGRESS_LOG_HEADING)
    if not heading or len(head) + len(heading) >= limit:
        return _clip_active_context(content, limit)

    parts = log.split(_ENTRY_SEPARATOR)
    preamble, entries = parts[0], parts[1:]
    if not entries:
        return _clip_active_context(content, limit)

    # Reserve the note's own worst case rather than a round number: the note is
    # part of what has to fit, and guessing at its length is how a cap ends up
    # being exceeded by the thing announcing the cap.
    reserve = len(prompts.get("active_context_truncated", omitted=len(entries))) + 2
    budget = limit - len(head) - len(heading) - len(preamble) - reserve

    kept: list[str] = []
    for entry in entries:
        cost = len(_ENTRY_SEPARATOR) + len(entry)
        if cost > budget:
            break
        kept.append(entry)
        budget -= cost

    omitted = len(entries) - len(kept)
    if not omitted:
        return content

    body = head + heading + preamble + "".join(
        _ENTRY_SEPARATOR + e for e in kept)
    return body.rstrip() + NEWLINE + NEWLINE + prompts.get(
        "active_context_truncated", omitted=omitted)


def _active_context_block(workspace: Path, limit: int) -> str:
    """active-context.md, handed over instead of asked for.

    Phase A used to spend a whole round trip on this: a prefill, a generated
    tool call, the client executing it, a second prefill with the response.
    The file's bytes land in the context either way — the tool response would
    carry exactly these — so injecting them is strictly cheaper than asking.

    It belongs in the volatile frame because the model rewrites this file in
    Phase B; in the system message a Phase B write would invalidate the cached
    prefix of the whole conversation. Being in the volatile frame is also why
    it is capped: see DEFAULT_CONTEXT_MAX_CHARS.
    """
    path = workspace / ".context" / "memory" / "active-context.md"
    try:
        if not path.is_file():
            return ""
        content = path.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return ""
    if not content:
        return ""
    return prompts.get("active_context_supplied",
                       ctx_path=_display_path(path),
                       content=fit_active_context(content, limit))


def _progress_log_note(workspace: Path, state: LodestarState,
                       messages: list[dict] | None = None) -> str:
    """Reminder to compress the Progress Log, or "" when it does not apply.

    READY only. In states 1-2 the model is executing protocol-critical steps
    and has not necessarily read the file yet; asking it to also restructure
    active-context.md there risks the Phase B write the whole cycle depends
    on. Phase D happens in READY, which is where compressing belongs anyway.

    **Two wordings, and the difference is evidence rather than volume.** The
    note rode the volatile frame and was therefore repeated on every turn, and
    a local model went on appending anyway -- which is the failure this whole
    supervisor exists for: a model with few active parameters does not follow
    an instruction because it was given politely, and repeating it louder is
    not a second mechanism.

    So the sharper wording is spent only where there is proof it was ignored:
    the model has WRITTEN active-context in this cycle -- the very file the
    reminder is about -- and the log is still over the cap. Before that write
    it may simply not have got there yet, and shouting at a model that is
    still working is how a guard becomes noise.

    The gateway does not compress the log itself, and that is the same rule it
    follows for code left inside `<think>`: it has the text but no right to
    fabricate a write into somebody's memory file.
    """
    if state < LodestarState.READY:
        return ""
    count = client_contract.count_progress_entries(workspace)
    if count is None or count <= MAX_PROGRESS_ENTRIES:
        return ""
    if messages and active_context_write_landed(messages, str(workspace)):
        return prompts.get("progress_log_ignored", count=count,
                           limit=MAX_PROGRESS_ENTRIES)
    return prompts.get("progress_log_full", count=count,
                       limit=MAX_PROGRESS_ENTRIES)


def _state_directive(workspace: Path, state: LodestarState, user_wants_skills: bool) -> str:
    """The per-state instruction block, from gateway/prompts/."""
    ctx_s = _display_path(workspace / ".context" / "memory" / "active-context.md")
    search_s = _display_path(workspace / ".context" / "semantic-memory" / "embed.py")

    if state == LodestarState.READ:
        name = "state_1_read"
    elif state == LodestarState.UPDATE:
        name = "state_2_update"
    elif user_wants_skills:
        name = "state_3_ready_skills"
    else:
        name = "state_3_ready"
    return prompts.get(name, ctx_path=ctx_s, search_path=search_s)


# The gateway's own mark on the messages it created. Idempotency here used to
# be decided by searching every message for "=== LODESTAR TURN ===" — and this
# repository's own AGENTS.md contains that literal. So a turn in which the
# model read AGENTS.md, gateway.log or a captured prompt dump silently lost the
# WHOLE volatile frame: state directive, TODO, active-context handover,
# cycle-budget notice. Measured 2026-09-12 on a two-message conversation: clean
# -> one frame of 1,257 characters, one poisoned tool result -> none, with no
# log line on either side and nothing in the audit able to tell them apart.
#
# Decide idempotency from state the gateway owns, never from a substring the
# conversation can carry.
FRAME_MARKER = "_lodestar_turn_frame"
BLOCK_MARKER = "_lodestar_protocol_block"

# The marker's VALUE, not just its key. A client sends whatever JSON it likes,
# so a key it can set is a key it can forge, and a forged marker puts the bug
# above back in a form that is harder to see. This token is minted when the
# process starts and is never transmitted or logged — the prompt the model
# sees is rendered text, and these keys are not part of it — so only code in
# this process can produce a message that answers to it.
#
# Clearing the keys on the way in was the first attempt, and it cost a real
# property: normalize() would then no longer be idempotent over its own
# output, which TestLodestarHardening asserts on purpose. A value the
# conversation cannot guess keeps both.
GATEWAY_MESSAGE_KEYS: tuple[str, ...] = (FRAME_MARKER, BLOCK_MARKER)

_OWNER_TOKEN = secrets.token_hex(8)


def _is_ours(message: dict, key: str) -> bool:
    """True if THIS process wrote that marker."""
    return message.get(key) == _OWNER_TOKEN


def _has_turn_frame(messages: list[dict]) -> bool:
    """True if THIS gateway already appended the volatile frame."""
    return any(_is_ours(m, FRAME_MARKER) for m in messages)


def _ensure_system_message(messages: list[dict]) -> dict:
    for m in messages:
        if m.get("role") == "system":
            return m
    new_sys = {"role": "system", "content": ""}
    messages.insert(0, new_sys)
    return new_sys


def inject_lodestar_context(
    messages: list[dict],
    workspace_dir: Path | None,
    state: LodestarState,
    max_synthesis: int | None = None,
    cycle_budget: int = 0,
    cycle_grace: int = 0,
    context_max_chars: int | None = None,
) -> list[dict]:
    """Append LODESTAR instruction block to system message (idempotent)."""
    if workspace_dir is None:
        return messages

    lodestar_content = _read_lodestar(workspace_dir)
    if lodestar_content is None:
        return messages  # No LODESTAR.md -> gating disabled

    sys_msg = _ensure_system_message(messages)

    existing = sys_msg.get("content") or ""
    # Not `_BLOCK_HEADER in existing`: the system message is composed by the
    # CLIENT — OpenCode puts AGENTS.md in it — so a project whose own docs
    # quote the header would never be given the protocol at all.
    already_injected = _is_ours(sys_msg, BLOCK_MARKER)

    wants_skills = user_requested_skills(messages)
    directive = _state_directive(workspace_dir, state, wants_skills)

    # v2.4: also inject a [LODESTAR TODO] checklist so the model sees
    # the required sequence of actions and what's still pending. The
    # enforcer (separate path) actually blocks free actions while a
    # required action is pending; the TODO is the visual guidance.
    from gateway.lodestar.todo import LodestarTodo as _Todo
    if context_max_chars is None:
        context_max_chars = DEFAULT_CONTEXT_MAX_CHARS
    context_block = _active_context_block(workspace_dir, context_max_chars)
    todo = _Todo.from_messages(messages, state, user_wants_skills=wants_skills,
                               context_supplied=bool(context_block))
    todo_text = todo.todo_text()
    todo_block = f"\n\n{todo_text}" if todo_text else ""

    # Once the gateway has stopped acting for the model, say so in the prompt:
    # the model needs to know why its blocked calls stopped producing results,
    # or it will keep retrying the same one forever.
    limit_block = ""
    if max_synthesis is not None:
        from gateway.lodestar.state import synthesis_count

        done = synthesis_count(messages)
        if done >= max_synthesis:
            limit_block = prompts.get("cycle_limit_reached", synthesis_count=done)

    # The turn budget, which until 2026-09-01 only wrote a log line. Says the
    # number, because that is the part a model acts on — and because the user
    # reading the transcript needs to know why the work stopped.
    turns_block = ""
    if cycle_budget:
        from gateway.lodestar.state import CYCLE_HARD, cycle_budget_state

        stage, turns = cycle_budget_state(messages, cycle_budget, cycle_grace)
        if stage is not None:
            name = ("cycle_turns_exhausted" if stage == CYCLE_HARD
                    else "cycle_turns_reached")
            turns_block = prompts.get(name, turns=turns, limit=cycle_budget)

    # Project rules belong with the protocol text: both are the same for
    # the whole task, so they stay in the system message where the KV
    # cache can keep them.
    rules_block = _project_rules_block(workspace_dir)
    progress_note = _progress_log_note(workspace_dir, state, messages)

    if not already_injected:
        block = (f"{_BLOCK_HEADER}\n{lodestar_content}\n{rules_block}\n"
                 f"{_BLOCK_FOOTER}\n")
        sys_msg["content"] = f"{existing}\n\n{block}".strip()
        sys_msg[BLOCK_MARKER] = _OWNER_TOKEN

    # The volatile half goes out as the LAST frame instead: it changes
    # between turns, and at the head of the prompt it was invalidating the
    # cached prefix of the whole conversation to deliver ~200 tokens.
    if not _has_turn_frame(messages):
        turn = (f"{_TURN_HEADER}{context_block}{directive}{todo_block}{limit_block}"
                f"{turns_block}{progress_note}\n{_TURN_FOOTER}")
        messages.append({"role": "system", "content": turn,
                         FRAME_MARKER: _OWNER_TOKEN})

    return messages
