"""LODESTAR TODO tracker — enforces a required sequence of actions per state.

The 2026-07-07 Capacitor session showed two failure modes that a per-state
enforcer alone cannot prevent:

1. Model decided to load brainstorming skill in Phase A (before reading
   active-context). The enforcer blocked the skill call but the model
   continued with hallucinated tools (task, glob) and JSON-parse errors.
2. In a follow-up session, the model did Phase A+B correctly but went
   straight to Glob+Write without loading the brainstorming skill that
   the user explicitly asked for. The Phase C directive was passive
   permission ("you MAY now load skills"), not an enforced requirement.

This module implements a "forced order" approach for the LODESTAR
**protocol steps** (read context, run search, edit active-context). The
bramka injects a TODO checklist in the system message and blocks any
tool call that is not the next required step.

v2.4.1 (2026-07-07): user feedback — skill loading is NOT enforced.
The model may load a skill at any point in Phase C, or skip it entirely.
A separate context_injector v2.3 directive reminds the model when the
user asked for skills, but does not block. Required LODESTAR steps
remain: read → search → edit.

Required actions per state:

  READ   (1): 1. read active-context.md
              2. bash embed.py search
  UPDATE (2): 1. edit active-context.md (set Goal/Status)
  READY  (3): (no required action; model is free — skill load is OPTIONAL)

When a required action is pending and the model tries a different tool,
the enforcer returns a new INJECT_TODO verdict. The caller emits an
error_chunk: "Your next required action is: X. Do that first, then
you can call the tool you wanted."
"""
from __future__ import annotations

from gateway import prompts

from dataclasses import dataclass, field
from typing import Iterable

from gateway.lodestar.state import LodestarState
from gateway.lodestar.tool_names import (
    READ_TOOLS as _READ_TOOLS,
    EDIT_TOOLS as _EDIT_TOOLS,
    SHELL_TOOLS as _SHELL_TOOLS,
    SKILL_TOOLS as _SKILL_TOOLS,
    get_path as _path,
    get_cmd as _cmd,
    parse_args as _parse_args,
    is_active_context_path as _is_active_context_path,
    is_embed_search_call as _is_embed_search_call,
    is_read_active_context_call as _is_read_active_context_call,
    is_edit_active_context_call as _is_edit_active_context_call,
    is_skill_call as _is_skill_call,
)


# Canonical action keys. The presence of a key in `completed` means
# the corresponding tool call was seen in conversation history.
_ACTION_READ_ACTIVE_CONTEXT = "read_active_context"
_ACTION_RUN_EMBED_SEARCH = "run_embed_search"
_ACTION_EDIT_ACTIVE_CONTEXT = "edit_active_context"
_ACTION_LOAD_SKILL = "load_skill"  # tracked but NOT enforced (v2.4.1)

# Required sequences per state, in order. Each tuple is (key, human_text).
# v2.4.1: skill load is NOT in the required list — the model may load a
# skill at any time during Phase C, or skip it entirely. The user's
# v2.4 feedback: forcing the model to load a skill is too aggressive;
# the LODESTAR protocol only enforces the read → search → edit sequence.
_REQUIRED_BY_STATE: dict[int, list[tuple[str, str]]] = {
    int(LodestarState.READ): [
        (_ACTION_READ_ACTIVE_CONTEXT,
         "Read .context/memory/active-context.md (bash/read/view_file with that path)"),
        (_ACTION_RUN_EMBED_SEARCH,
         'Run: python .context/semantic-memory/embed.py search "<keywords in English>"'),
    ],
    int(LodestarState.UPDATE): [
        (_ACTION_EDIT_ACTIVE_CONTEXT,
         "Edit .context/memory/active-context.md (set Goal, Next Steps, Status=W toku)"),
    ],
    int(LodestarState.READY): [],  # No required actions; model is free
}


@dataclass
class LodestarTodo:
    """Tracks LODESTAR phase actions and exposes next-required + completion."""

    state: LodestarState
    user_wants_skills: bool
    completed: dict[str, bool] = field(default_factory=dict)
    edit_failed: bool = False  # an active-context edit errored this task

    @classmethod
    def from_messages(
        cls,
        messages: list[dict],
        state: LodestarState,
        user_wants_skills: bool,
        context_supplied: bool = False,
        workspace_dir: str | None = None,
    ) -> "LodestarTodo":
        """Build the checklist for this request.

        `context_supplied` is for the injector, which has already rendered the
        handover block. `workspace_dir` is for everyone else: the answer is
        derived here rather than passed in, because the first version left the
        two enforcement call sites in app.py computing it independently — they
        did not, so the enforcer went on demanding a read the model had already
        been handed, blocked its search and synthesized the read itself.
        """
        # Same per-task boundary as determine_state: actions from a CLOSED
        # cycle (Status=Zakończono written) must not satisfy the checklist
        # of the follow-up task.
        from gateway.lodestar.state import effective_messages
        messages = effective_messages(list(messages))
        completed = cls._analyze_history(messages)
        from gateway.lodestar.state import context_supplied_by_gateway
        if context_supplied or context_supplied_by_gateway(workspace_dir):
            # The gateway put active-context.md in the turn frame, so the read
            # is already done and must not be demanded — the enforcer would
            # otherwise block the search and synthesize a read of a file the
            # model was just handed.
            completed[_ACTION_READ_ACTIVE_CONTEXT] = True
        return cls(
            state=state,
            user_wants_skills=user_wants_skills,
            completed=completed,
            edit_failed=cls._had_failed_edit(messages),
        )

    @staticmethod
    def _had_failed_edit(messages: Iterable[dict]) -> bool:
        """True if an active-context edit errored after the last user message."""
        from gateway.lodestar.state import response_indicates_tool_error
        pending: set[str] = set()
        failed = False
        for m in messages:
            if m.get("role") == "user":
                pending.clear()
                failed = False
                continue
            for tc in m.get("tool_calls") or []:
                fn = (tc.get("function") or {}).get("name", "")
                args = _parse_args(tc.get("function") or {})
                if _is_edit_active_context_call(fn, args) and tc.get("id"):
                    pending.add(tc["id"])
            if m.get("role") == "tool" and m.get("tool_call_id") in pending:
                if response_indicates_tool_error(str(m.get("content") or "")):
                    failed = True
        return failed

    @staticmethod
    def _analyze_history(messages: Iterable[dict]) -> dict[str, bool]:
        done = {
            _ACTION_READ_ACTIVE_CONTEXT: False,
            _ACTION_RUN_EMBED_SEARCH: False,
            _ACTION_EDIT_ACTIVE_CONTEXT: False,
            _ACTION_LOAD_SKILL: False,
        }
        from gateway.lodestar.state import (
            response_indicates_context_read,
            response_indicates_embed_search,
            response_indicates_tool_error,
        )
        pending_edit_ids: set[str] = set()
        for m in messages:
            # Tool calls from assistant messages
            for tc in m.get("tool_calls") or []:
                fn = (tc.get("function") or {}).get("name", "")
                args = _parse_args(tc.get("function") or {})
                if _is_read_active_context_call(fn, args):
                    done[_ACTION_READ_ACTIVE_CONTEXT] = True
                if _is_embed_search_call(fn, args):
                    done[_ACTION_RUN_EMBED_SEARCH] = True
                if _is_edit_active_context_call(fn, args):
                    # Count only after the tool response confirms success —
                    # a "Could not find oldString" edit must keep the step
                    # pending (2026-07-09 13:24 incident).
                    cid = tc.get("id")
                    if cid:
                        pending_edit_ids.add(cid)
                    else:
                        done[_ACTION_EDIT_ACTIVE_CONTEXT] = True
                if _is_skill_call(fn):
                    done[_ACTION_LOAD_SKILL] = True
            # Tool responses (rescues pruned/orphaned histories where the
            # originating call lost its arguments) — same strict markers
            # as gateway.lodestar.state.determine_state.
            if m.get("role") == "tool" and isinstance(m.get("content"), str):
                content = m["content"]
                if response_indicates_context_read(content):
                    done[_ACTION_READ_ACTIVE_CONTEXT] = True
                if response_indicates_embed_search(content):
                    done[_ACTION_RUN_EMBED_SEARCH] = True
                cid = m.get("tool_call_id")
                if cid in pending_edit_ids:
                    pending_edit_ids.discard(cid)
                    if not response_indicates_tool_error(content):
                        done[_ACTION_EDIT_ACTIVE_CONTEXT] = True
        # Edits with no response in history yet: assume success (optimistic).
        if pending_edit_ids:
            done[_ACTION_EDIT_ACTIVE_CONTEXT] = True
        return done

    def mark_call(self, tool_name: str, tool_params: dict) -> None:
        """Record a tool call emitted DURING the current stream.

        The todo is built from history at request start, so without this a
        model emitting `read` + `bash embed.py search` in one turn had its
        (correct) search blocked: the enforcer still saw the read as pending
        and auto-executed a redundant second read (2026-07-09 13:05 log).
        """
        name = (tool_name or "").lower()
        if _is_read_active_context_call(name, tool_params):
            self.completed[_ACTION_READ_ACTIVE_CONTEXT] = True
        if _is_embed_search_call(name, tool_params):
            self.completed[_ACTION_RUN_EMBED_SEARCH] = True
        if _is_edit_active_context_call(name, tool_params):
            self.completed[_ACTION_EDIT_ACTIVE_CONTEXT] = True
        if _is_skill_call(name):
            self.completed[_ACTION_LOAD_SKILL] = True

    def next_required(self) -> tuple[str, str] | None:
        """Return (action_key, human_text) for the next pending required action,
        or None if all required actions are complete.

        v2.4.1: skill load is NOT in the required list. The model is free
        to load a skill at any point in Phase C, or skip it entirely.
        """
        # State-bound required actions (read, search, edit only)
        for key, text in _REQUIRED_BY_STATE.get(int(self.state), []):
            if not self.completed.get(key, False):
                if key == _ACTION_EDIT_ACTIVE_CONTEXT and self.edit_failed:
                    # Escalate after a failed edit: full-content write can't
                    # fail on oldString matching.
                    text = prompts.get("todo_write_after_failed_edit")
                return (key, text)
        return None

    def is_blocking(self, tool_name: str, tool_params: dict) -> bool:
        """Return True if a required action is pending and this call is not
        the one that completes it. Caller should block the call and inject
        a reminder.

        v2.4.1: skill calls are NEVER blocking. The model is free to call
        skill() at any time during Phase C, and free to skip it.
        """
        if _is_skill_call(tool_name):
            return False  # Never block skill calls
        if _is_read_active_context_call(tool_name.lower(), tool_params):
            # Re-reading active-context is always legitimate — after a failed
            # edit the model re-reads the file to build a fresh oldString
            # (2026-07-09 13:42: this recovery was blocked and forced a
            # synthetic write instead of letting the model self-correct).
            return False
        next_req = self.next_required()
        if next_req is None:
            return False
        next_key, _ = next_req
        # If this call is doing the required action, don't block
        if next_key == _ACTION_READ_ACTIVE_CONTEXT and _is_read_active_context_call(tool_name, tool_params):
            return False
        if next_key == _ACTION_RUN_EMBED_SEARCH and _is_embed_search_call(tool_name, tool_params):
            return False
        if next_key == _ACTION_EDIT_ACTIVE_CONTEXT and _is_edit_active_context_call(tool_name, tool_params):
            return False
        return True

    def todo_text(self) -> str:
        """Render a checklist for the system message.

        v2.4.1: only includes the enforced steps (read, search, edit). The
        skill load is NOT in the enforced list — see the separate
        context_injector v2.3 directive for skill-as-guidance.
        """
        required: list[tuple[str, str]] = list(_REQUIRED_BY_STATE.get(int(self.state), []))
        if not required:
            return ""
        lines = ["[LODESTAR TODO]"]
        for key, text in required:
            mark = "[x]" if self.completed.get(key, False) else "[ ]"
            lines.append(f"  {mark} {text}")
        next_req = self.next_required()
        if next_req is not None:
            lines.append(f"NEXT REQUIRED ACTION: {next_req[1]}")
            lines.append(
                "The gateway will BLOCK any other tool call until this is done."
            )
        return "\n".join(lines)
