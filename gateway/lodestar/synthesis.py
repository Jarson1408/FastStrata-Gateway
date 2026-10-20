"""
Synthetic tool calls the gateway makes on the model's behalf.

When the model will not take the next required LODESTAR step — it answers
with text, or reaches for an unrelated tool — the gateway emits the call
itself. The client executes it, the result comes back, and the session
continues instead of stalling with finish=stop and no work done.

This is the part of the protocol with real side effects: these calls read and
overwrite files in the user's workspace and run shell commands, none of which
the model asked for. It used to live inside app.py's 400-line
_handle_fsm_event, mixed in with SSE framing, which made it hard to see what
the gateway does unprompted. Keeping it in one module makes that surface
reviewable.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from gateway import prompts
from gateway.lodestar.state import PHASE_D_MARKER, phase_d_pending
from gateway.lodestar.context_injector import clean_user_text
from gateway.lodestar.subagent import is_subagent_request
from gateway.lodestar.tool_names import (
    READ_CANDIDATES, SHELL_CANDIDATES, WRITE_CANDIDATES, first_available,
)
from gateway.shell_safety import build_embed_search_command, sanitize_search_query

if TYPE_CHECKING:
    from gateway.lodestar.todo import LodestarTodo
    from gateway.openai_sse import OpenAISSEEmitter

_log = logging.getLogger("gateway.lodestar.synthesis")

def synthesize_phase_d_nudge(
    client_body: dict | None,
    tools: list[dict],
    emitter: "OpenAISSEEmitter",
    raw_tail: str,
    workspace_dir: str | None = None,
) -> list[bytes] | None:
    """Remind the model to finalize when it ends an open task with plain text.

    Phase D (save lesson + close active-context) has no hard trigger the
    state machine can gate on — the model just stops calling tools. This
    nudge converts a final-looking, text-only turn into one synthetic
    `bash echo` call carrying the reminder; the echo output returns as a
    tool response, the client re-prompts the model, and the model writes
    its OWN lesson and closes the context before answering.

    Guards: never for subagents, never when the model is asking the user a
    question (trailing "?"), at most once per cycle (marker check), and never
    when active-context.md on disk already reports the task finished --
    **the message asserts what that file says, so the decision has to have
    read it.** Without `workspace_dir` the old history-only answer stands.
    """

    msgs = (client_body or {}).get("messages") or []
    if is_subagent_request(None, msgs):
        return None
    if not phase_d_pending(msgs, workspace_dir):
        return None
    if "?" in (raw_tail or "")[-150:]:
        return None  # the model is asking the user something — not a final answer

    # At most one nudge per cycle: look for the marker after the last user msg
    last_user = -1
    for i, m in enumerate(msgs):
        if m.get("role") == "user":
            last_user = i
    for m in msgs[last_user + 1:]:
        if m.get("role") == "tool" and PHASE_D_MARKER in str(m.get("content") or ""):
            return None
        for tc in m.get("tool_calls") or []:
            if PHASE_D_MARKER in str((tc.get("function") or {}).get("arguments") or ""):
                return None

    bash_tool = first_available(tools, SHELL_CANDIDATES)
    if not bash_tool:
        return None

    msg = PHASE_D_NUDGE_TEXT
    syn_args = json.dumps({"command": f'echo "{msg}"'}, ensure_ascii=False)
    syn_call_id = f"syn_phd_{uuid.uuid4().hex[:6]}"
    return [
        emitter.tool_call_start_chunk(8890, syn_call_id, bash_tool),
        emitter.tool_call_arg_chunk(8890, syn_args),
    ]


# Appended to every delegated task prompt. Opens with subagent.SUBAGENT_MARKER,
# which is what the detection keys on — the gateway is the only thing that
# emits that tag, so a spawned session is recognised without the loose phrase
# matching that used to let an ordinary sentence disable the protocol.
SUBAGENT_TASK_MARKER = "\n\n" + prompts.get("subagent_directive")

# The Phase D reminder, kept as a module constant so the marker check and the
# snapshot test can both reach it.
PHASE_D_NUDGE_TEXT = prompts.get("phase_d_nudge", marker=PHASE_D_MARKER)


def extract_user_goal(messages: list[dict]) -> str | None:
    """Extract the user's actual task text from user messages (newest first).

    Falls back to earlier user messages when the newest one contains only
    injected boilerplate (e.g. a bare /skill invocation with no arguments).
    """
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        text = clean_user_text(m.get("content"))
        if text:
            # Collapsed to one line: this goes into active-context.md as a
            # bullet under "## Goal", and a task description containing its own
            # markdown heading used to become a second "## Status" section in
            # the file — splitting the document and silently discarding
            # everything after the heading.
            return re.sub(r"\s+", " ", text).strip()[:400]
    return None


def embed_search_query(client_body: dict | None) -> str | None:
    """Build a clean, shell-safe embed.py search query from the user's task.

    The 2026-07-09 skill run showed the synthetic search using the RAW user
    message — including OpenCode's <EXTREMELY_IMPORTANT> superpowers preamble
    and newlines — as the query, so semantic search matched boilerplate
    instead of the task.
    """
    goal = extract_user_goal((client_body or {}).get("messages") or [])
    if not goal:
        return None
    q = re.sub(
        r"^(stworz|stwórz|napisz|zrob|zrób|utworz|utwórz|create|write|build|make)\s+",
        "", goal, flags=re.IGNORECASE,
    )
    # Shell metacharacters are stripped here, not just quoted: the goal text
    # is user input and ends up inside a command the client runs unattended.
    return sanitize_search_query(q)


def set_md_section(md: str, header: str, body: str) -> str:
    """Replace the body of a `## header` section in a markdown document."""
    pattern = re.compile(
        rf"(^##\s+{re.escape(header)}\s*\n)(.*?)(?=^##\s|^---\s*$|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    if pattern.search(md):
        return pattern.sub(lambda m: m.group(1) + body + "\n\n", md, count=1)
    return md


def rewrite_preserves_document(original: str, rewritten: str) -> bool:
    """Guard a synthetic full-file write against destroying the document.

    `set_md_section` rewrites active-context.md with a regex and the result is
    sent as a `write` call the client performs without asking. A mis-matching
    pattern would therefore silently truncate the user's context file —
    Progress Log included. Two cheap invariants catch that: every `## ` heading
    the original had must still be there, and the file must not have lost most
    of its body.
    """

    original_headers = re.findall(r"^##\s+(.+?)\s*$", original, re.MULTILINE)
    new_headers = re.findall(r"^##\s+(.+?)\s*$", rewritten, re.MULTILINE)

    # Added structure is as damaging as lost structure, and the check for lost
    # headings cannot see it: a goal carrying its own "## Status" produced a
    # document with two Status sections, both of which passed the "every
    # original heading survives" test while the second shadowed the first.
    if len(new_headers) > len(original_headers):
        _log.error(
            "LODESTAR ENFORCER: refusing synthetic active-context write — "
            "it would add section(s) %s",
            sorted(set(new_headers) - set(original_headers)) or "(duplicate)",
        )
        return False

    missing = set(original_headers) - set(new_headers)
    if missing:
        _log.error(
            "LODESTAR ENFORCER: refusing synthetic active-context write — "
            "it would drop section(s) %s", sorted(missing),
        )
        return False
    if original and len(rewritten) < len(original) * 0.6:
        _log.error(
            "LODESTAR ENFORCER: refusing synthetic active-context write — "
            "content shrank from %d to %d chars", len(original), len(rewritten),
        )
        return False
    return True


def synthesize_active_context_write(
    workspace_dir: str | None,
    tools: list[dict],
    client_body: dict | None,
) -> tuple[str, str] | None:
    """Build a synthetic write of active-context.md (Goal + Status=W toku).

    Used when the model is in state 2 but never emits the required edit —
    the observed 2026-07-09 stall: the model produced only text, the
    stubborn guide had nothing to synthesize for edit_active_context, and
    the turn ended with finish=stop, silently killing the session.

    Returns (write_tool_name, args_json) or None if synthesis is impossible.
    """

    if not workspace_dir:
        return None
    ctx_path = Path(workspace_dir) / ".context" / "memory" / "active-context.md"
    if not ctx_path.is_file():
        return None
    try:
        current = ctx_path.read_text(encoding="utf-8")
    except OSError:
        return None

    goal = extract_user_goal((client_body or {}).get("messages") or [])
    if not goal:
        return None

    write_tool = first_available(tools, WRITE_CANDIDATES)
    if not write_tool:
        return None

    new_content = set_md_section(current, "Goal", f"- {goal}")
    new_content = set_md_section(new_content, "Status", "- W toku")

    if not rewrite_preserves_document(current, new_content):
        return None  # better no synthesis than a truncated context file

    # Resolve the content parameter name from the tool schema
    content_param = "content"
    for t in tools:
        fn = t.get("function") or {}
        if fn.get("name") == write_tool:
            props = (fn.get("parameters") or {}).get("properties") or {}
            for cand in ("content", "CodeContent", "text", "new_content"):
                if cand in props:
                    content_param = cand
                    break
            break

    args = json.dumps(
        {
            path_param_for(tools, write_tool): str(ctx_path).replace("\\", "/"),
            content_param: new_content,
        },
        ensure_ascii=False,
    )
    _log.warning(
        "LODESTAR ENFORCER: synthesizing a full-file write of %s "
        "(%d -> %d chars) via '%s'",
        ctx_path.name, len(current), len(new_content), write_tool,
    )
    return write_tool, args


def synthesize_stubborn_call(
    next_key: str,
    workspace_dir: str | None,
    tools: list[dict],
    emitter: "OpenAISSEEmitter",
    client_body: dict | None = None,
) -> list[bytes] | None:
    """v2.4.7: synthesize a tool call the model SHOULD have made.

    Called when the model produces a turn with no tool calls (just
    text) and there's a required LODESTAR action pending. The
    user said: 'wysylac mu bez mojej wiedzy komunikaty, ktore maja go
    do tego zmusic' (send messages without my knowledge that force
    it to do this). The bramka acts as that force.

    Returns a list of synthetic SSE chunks (TOOL_CALL_START +
    TOOL_CALL_ARG) that the OpenCode client will execute, OR None
    if no synthesis is possible for this required action (e.g.
    edit_active_context needs model-provided content).
    """

    if next_key == "read_active_context" and workspace_dir:
        ctx_path = Path(workspace_dir) / ".context" / "memory" / "active-context.md"
        if not ctx_path.is_file():
            return None
        # Find the client's read tool
        read_tool = first_available(tools, READ_CANDIDATES)
        if not read_tool:
            return None
        syn_idx = 8888
        syn_call_id = f"syn_stub_{uuid.uuid4().hex[:6]}"
        syn_args = json.dumps(
            {path_param_for(tools, read_tool): str(ctx_path).replace("\\", "/")},
            ensure_ascii=False,
        )
        return [
            emitter.tool_call_start_chunk(syn_idx, syn_call_id, read_tool),
            emitter.tool_call_arg_chunk(syn_idx, syn_args),
        ]

    if next_key == "run_embed_search" and workspace_dir:
        query = embed_search_query(client_body)
        if not query:
            return None
        embed_path = Path(workspace_dir) / ".context" / "semantic-memory" / "embed.py"
        if not embed_path.is_file():
            return None
        bash_tool = first_available(tools, SHELL_CANDIDATES)
        if not bash_tool:
            return None
        syn_cmd = build_embed_search_command(embed_path, query)
        if not syn_cmd:
            return None
        syn_idx = 8889
        syn_call_id = f"syn_stub_{uuid.uuid4().hex[:6]}"
        syn_args = json.dumps({"command": syn_cmd}, ensure_ascii=False)
        return [
            emitter.tool_call_start_chunk(syn_idx, syn_call_id, bash_tool),
            emitter.tool_call_arg_chunk(syn_idx, syn_args),
        ]

    if next_key == "edit_active_context":
        # v2.5.1: the gateway sets the Goal itself (from the user's request)
        # when the model won't emit the edit. Before this, state 2 was a
        # dead end: "no synthesized call available ... sending noop" and
        # the session stopped with finish=stop.
        synth = synthesize_active_context_write(workspace_dir, tools, client_body)
        if synth is None:
            return None
        write_tool, args = synth
        syn_idx = 8890
        syn_call_id = f"syn_stub_{uuid.uuid4().hex[:6]}"
        return [
            emitter.tool_call_start_chunk(syn_idx, syn_call_id, write_tool),
            emitter.tool_call_arg_chunk(syn_idx, args),
        ]

    return None


def path_param_for(tools: list[dict], tool_name: str) -> str:
    """Pick the file-path parameter name a read/edit tool actually declares.

    The candidate order differs from tool_names.PATH_KEYS on purpose: this
    picks what to *write* into a synthesized call, and clients that declare
    several of these expect the first one here.
    """
    for t in tools:
        fn = t.get("function") or {}
        if fn.get("name") == tool_name:
            props = (fn.get("parameters") or {}).get("properties") or {}
            for cand in ("filePath", "AbsolutePath", "path", "file_path", "TargetFile"):
                if cand in props:
                    return cand
            break
    return "filePath"


# ---------------------------------------------------------------------------
# Required-step synthesis, shared by the streaming and non-streaming paths
# ---------------------------------------------------------------------------

def query_from_model_command(cmd: str) -> str | None:
    """Recover the search term from the model's own embed.py command."""
    if "embed.py" not in cmd or "search" not in cmd:
        return None
    for pattern in (r'search\s+"([^"]+)"', r"search\s+'([^']+)'", r"search\s+(\S+)"):
        m = re.search(pattern, cmd)
        if m:
            return m.group(1).strip('"').strip("'")
    return None


def synthetic_context_read(
    tools: list[dict],
    workspace_dir: str | None,
    todo: "LodestarTodo | None" = None,
) -> tuple[str, str] | None:
    """(tool, args_json) reading active-context.md, or None."""
    if not workspace_dir:
        return None
    ctx_path = Path(workspace_dir) / ".context" / "memory" / "active-context.md"
    if not ctx_path.is_file():
        return None
    read_tool = first_available(tools, READ_CANDIDATES)
    if not read_tool:
        return None
    args = json.dumps(
        {path_param_for(tools, read_tool): str(ctx_path).replace("\\", "/")},
        ensure_ascii=False,
    )
    if todo is not None:
        todo.mark_call(read_tool, {"filePath": str(ctx_path)})
    return read_tool, args


def synthetic_embed_search(
    tools: list[dict],
    workspace_dir: str | None,
    client_body: dict | None,
    todo: "LodestarTodo | None" = None,
    model_command: str = "",
) -> tuple[str, str] | None:
    """(tool, args_json) running the semantic search, or None.

    Prefers the query the model was already reaching for; falls back to the
    user's task text.
    """
    if not workspace_dir:
        return None
    embed_path = Path(workspace_dir) / ".context" / "semantic-memory" / "embed.py"
    if not embed_path.is_file():
        return None
    bash_tool = first_available(tools, SHELL_CANDIDATES)
    if not bash_tool:
        return None
    query = query_from_model_command(model_command) or embed_search_query(client_body)
    # Sanitization belongs here, not at the call site: this query may come
    # straight out of the model's own command string.
    syn_cmd = build_embed_search_command(embed_path, query)
    if not syn_cmd:
        return None
    if todo is not None:
        todo.mark_call(bash_tool, {"command": syn_cmd})
    return bash_tool, json.dumps({"command": syn_cmd}, ensure_ascii=False)


def synthetic_goal_write(
    tools: list[dict],
    workspace_dir: str | None,
    client_body: dict | None,
    todo: "LodestarTodo | None" = None,
) -> tuple[str, str] | None:
    """(tool, args_json) committing the Goal, or None.

    v2.5.1: before this, state 2 was a dead end — the model produced text only,
    nothing could be synthesized, and the turn ended finish=stop with the
    session silently dead.
    """
    synth = synthesize_active_context_write(workspace_dir, tools, client_body)
    if synth is None:
        return None
    write_tool, syn_args = synth
    if todo is not None:
        todo.mark_call(write_tool, {"filePath": "active-context.md"})
    return write_tool, syn_args


def synthesize_required_step(
    next_key: str | None,
    *,
    tools: list[dict],
    workspace_dir: str | None,
    client_body: dict | None,
    todo: "LodestarTodo | None" = None,
    model_command: str = "",
) -> tuple[str, str] | None:
    """(tool, args_json) for the pending protocol step, or None.

    One lookup instead of the three nested if-blocks this replaced, and the
    single place both request paths ask "what would the model have had to do
    here?".
    """
    if next_key == "read_active_context":
        return synthetic_context_read(tools, workspace_dir, todo)
    if next_key == "run_embed_search":
        return synthetic_embed_search(tools, workspace_dir, client_body, todo,
                                      model_command=model_command)
    if next_key == "edit_active_context":
        return synthetic_goal_write(tools, workspace_dir, client_body, todo)
    return None
