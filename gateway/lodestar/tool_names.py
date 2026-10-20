"""
Shared vocabulary for recognising client tools by name and arguments.

Different clients name the same capability differently — `read` / `read_file`
/ `view_file`, `bash` / `run_command` / `shell` — and pass paths under
different keys. Every part of the protocol needs the same answers to "is this
a read of active-context.md?" and "is this an embed.py search?".

Those sets and helpers used to exist three times over, in state.py, enforcer.py
and todo.py, plus eight inline candidate tuples in app.py and tool_schema.py.
Eleven places to edit for one new alias, and the copies had already drifted:
state._get_cmd() did not lowercase while the other two did, so an
`EMBED.PY SEARCH` command was recognised by two of the three consumers.

One definition lives here. Consumers import it; they do not restate it.
"""
from __future__ import annotations

import json
from typing import Any, Iterable

__all__ = [
    "READ_TOOLS", "EDIT_TOOLS", "SHELL_TOOLS", "SKILL_TOOLS", "SEARCH_TOOLS",
    "PATH_KEYS", "CMD_KEYS", "WORKDIR_KEYS", "RESOURCE_KEYS",
    "ACTIVE_CONTEXT", "EMBED_SCRIPT", "get_resource",
    "READ_CANDIDATES", "EDIT_CANDIDATES", "SHELL_CANDIDATES", "WRITE_CANDIDATES",
    "get_path", "get_cmd", "get_workdir", "parse_args", "first_available",
    "is_active_context_path", "is_embed_search_call",
    "is_read_active_context_call", "is_edit_active_context_call",
    "is_skill_call",
]

# -- Tool name families ------------------------------------------------------

READ_TOOLS: frozenset[str] = frozenset({
    "read", "read_file", "view_file", "get_file_content",
})
EDIT_TOOLS: frozenset[str] = frozenset({
    "apply_patch", "edit", "write", "write_file",
    "replace_file_content", "write_to_file",
})
SHELL_TOOLS: frozenset[str] = frozenset({"bash", "run_command", "shell", "exec"})
SKILL_TOOLS: frozenset[str] = frozenset({"skill", "load_skill"})

#: Tools that LOOK for something, and whose "no match" is an answer rather
#: than a failure.
#:
#: This distinction has one consumer today -- adaptive sampling, which lowers
#: temperature when tool calls fail -- and it matters because the failure is
#: silent in the wrong direction: a `grep` that legitimately finds nothing was
#: counted as an error, so a routine search dropped the temperature of the
#: next generation. Measured 2026-09-12: three empty greps took temperature
#: from 0.3 to 0.15.
#:
#: The names this rig actually serves (`grep`, `glob`) are first; the rest are
#: other clients' spellings of the same capability, kept for the same reason
#: every set in this file keeps them.
SEARCH_TOOLS: frozenset[str] = frozenset({
    "grep", "glob", "list", "ls",
    "grep_search", "search_grep", "ripgrep", "codebase_search",
    "list_dir", "list_directory", "view_file_outline",
    "search_web", "web_search", "get_code_item",
})

# Ordered preference when the gateway has to pick a tool to synthesize a call
# with. Order matters — the first name the client declared wins.
READ_CANDIDATES: tuple[str, ...] = ("read", "view_file", "read_file", "get_file_content")
WRITE_CANDIDATES: tuple[str, ...] = ("write", "write_file", "write_to_file",
                                     "replace_file_content")
EDIT_CANDIDATES: tuple[str, ...] = ("edit", "apply_patch") + WRITE_CANDIDATES
SHELL_CANDIDATES: tuple[str, ...] = ("bash", "run_command", "shell", "exec")

# -- Argument keys -----------------------------------------------------------

PATH_KEYS: tuple[str, ...] = ("filePath", "path", "AbsolutePath", "TargetFile", "file_path")
CMD_KEYS: tuple[str, ...] = ("command", "CommandLine", "cmd", "shell_command")
# Where a shell tool is asked to run. Separate from PATH_KEYS on purpose: a
# path argument names what the call operates ON, a workdir names where it
# stands, and only the second one can be wrong in a way the OS refuses outright.
WORKDIR_KEYS: tuple[str, ...] = ("workdir", "cwd", "working_directory", "workingDirectory")

#: Every key under which a call names the THING it is working on, for asking
#: "are these two calls about the same thing?".
#:
#: A superset of PATH_KEYS, and separate from it on purpose: PATH_KEYS decides
#: what `path_repair` may rewrite and what `destructive` inspects, and those
#: must stay narrow -- widening the set that authorises a rewrite is not the
#: same decision as widening the set that recognises a repetition.
RESOURCE_KEYS: tuple[str, ...] = PATH_KEYS + (
    "SearchPath", "DirectoryPath", "file", "filename", "Target", "pattern",
)

ACTIVE_CONTEXT = "active-context.md"
EMBED_SCRIPT = "embed.py"


# -- Extraction --------------------------------------------------------------

def get_path(params: dict) -> str:
    """First declared path argument, lowercased. Empty string when absent."""
    for key in PATH_KEYS:
        value = params.get(key)
        if value:
            return str(value).lower()
    return ""


def get_resource(params: dict) -> str | None:
    """What this call is working on, verbatim, or None when it says nothing.

    None rather than "" because the caller has to tell "no resource named"
    from "the same resource" -- two calls that both name nothing are not
    thereby about the same file.
    """
    for key in RESOURCE_KEYS:
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def get_cmd(params: dict) -> str:
    """First declared command argument, lowercased. Empty string when absent.

    Lowercasing is deliberate and is what the three former copies disagreed
    about: every consumer compares against lowercase literals, so the copy
    that preserved case simply failed to recognise an uppercase command.
    """
    for key in CMD_KEYS:
        value = params.get(key)
        if value:
            return str(value).lower()
    return ""


def get_workdir(params: dict) -> str:
    """First declared working-directory argument, verbatim. Empty when absent.

    Not lowercased, unlike get_path: this value is handed back to the model in
    an error message and compared against the filesystem, so its case matters.
    """
    for key in WORKDIR_KEYS:
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def parse_args(func: dict) -> dict:
    """Tool-call arguments as a dict, whatever shape they arrived in."""
    raw: Any = func.get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def first_available(tools: list[dict] | None, candidates: Iterable[str]) -> str | None:
    """First candidate the client actually declared, or None.

    Replaces the hand-rolled `for cand in (...)` loops that were scattered
    across app.py and tool_schema.py.
    """
    declared = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            declared.add(fn["name"])
    for candidate in candidates:
        if candidate in declared:
            return candidate
    return None


# -- Predicates --------------------------------------------------------------

def is_active_context_path(params: dict) -> bool:
    return ACTIVE_CONTEXT in get_path(params)


def is_embed_search_call(name: str, params: dict) -> bool:
    if name not in SHELL_TOOLS:
        return False
    cmd = get_cmd(params)
    return EMBED_SCRIPT in cmd and " search " in f" {cmd} "


def is_read_active_context_call(name: str, params: dict) -> bool:
    return name in READ_TOOLS and is_active_context_path(params)


def is_edit_active_context_call(name: str, params: dict) -> bool:
    return name in EDIT_TOOLS and is_active_context_path(params)


def is_skill_call(name: str) -> bool:
    return name in SKILL_TOOLS
