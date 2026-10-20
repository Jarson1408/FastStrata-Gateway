"""Base definitions and protocols for tool calling parsers."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Protocol


class EventType(Enum):
    """High-level events emitted by the tool parser."""
    CONTENT = "content"                  # text content → delta.content
    REASONING = "reasoning"              # thinking text → delta.reasoning_content
    TOOL_CALL_START = "tool_call_start"  # {index, id, name}
    PARAM_START = "param_start"          # {index, name, type, is_first}
    PARAM_CHUNK = "param_chunk"          # {index, text, param_type}
    PARAM_END = "param_end"              # {index, name, param_type}
    TOOL_CALL_END = "tool_call_end"      # {index, id, name, params, auto_closed}


@dataclass(slots=True)
class FsmEvent:
    type: EventType
    data: Any = None


class BaseToolParser(Protocol):
    """Protocol for stateful tool call parsers."""

    @property
    def tool_call_count(self) -> int:
        ...

    @property
    def in_thinking(self) -> bool:
        """True while the parser sits inside an unterminated <think> block.

        The reasoning-budget guard needs this: a turn that has produced
        thousands of tokens and is STILL inside <think> has not started a
        tool call, so cutting it cannot truncate one mid-argument.
        """
        ...

    def feed(self, chunk: str) -> Iterator[FsmEvent]:
        ...

    def finish(self, finish_reason: str = "stop") -> Iterator[FsmEvent]:
        ...


def extract_json_candidate(text: str) -> str:
    """Extract JSON string from potential markdown code fences or raw text."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            if lines[-1].strip().startswith("```"):
                return "\n".join(lines[1:-1]).strip()
            return "\n".join(lines[1:]).strip()
    return s


def is_json_candidate_closed(text: str) -> bool:
    """Check whether a text starting with '{' or '```' appears closed."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        return len(lines) >= 2 and lines[-1].strip().startswith("```")
    if s.startswith("{"):
        in_str = False
        escape = False
        depth = 0
        has_open = False
        for ch in s:
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if not in_str:
                if ch == '{':
                    depth += 1
                    has_open = True
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and has_open:
                        return True
        return depth == 0 and has_open
    return False


COMMON_TOOL_KEYS = {
    "name", "tool", "function", "id", "path", "filePath", "file",
    "content", "command", "cmd", "script", "pattern", "patch", "patchText",
    "oldString", "old_string", "old_text", "agent", "prompt", "task",
    "question", "questions", "directory", "dir", "arguments", "parameters",
}


#: Keys only a CONTENT search takes. A glob lists files by NAME, so a call
#: carrying one of these was never meant for it.
_GREP_ONLY_KEYS = frozenset({
    "include", "caseSensitive", "case_sensitive", "ignoreCase", "ignore_case",
    "-i", "output_mode", "multiline", "regex",
})
_GLOB_META = re.compile(r"[*?\[\]{}]")
_REGEX_SIGNS = re.compile(r"\\[bdswBDSW]|\.\*|\.\+|\||\^|\$|[()]")
_FILE_NAME = re.compile(r"\.[A-Za-z0-9]{1,8}$")


def _schema_props(tools: list[dict] | None, name: str) -> set[str]:
    for t in tools or ():
        fn = (t or {}).get("function") or {}
        if fn.get("name") == name:
            return set(((fn.get("parameters") or {}).get("properties") or {}).keys())
    return set()


def choose_search_tool(data: dict[str, Any], tools: list[dict] | None = None) -> str:
    """`grep` or `glob` for a call that names no tool and carries `pattern`.

    Both tools take `pattern`, and until 2026-10-17 every such call was sent
    as `glob`. Measured that evening: the model reasoned "the glob tool does
    not work on files, let me use grep", emitted the same nameless JSON --
    `pattern` a regex, `path` a FILE, `caseSensitive` -- and the gateway named
    it `glob` again, ten times, until the loop breaker fired. The model was
    right every time; the gateway was the loop.

    Decided by evidence in the arguments, weighed rather than first-match,
    because each signal alone is weak: keys only one schema declares, a path
    that names a file (glob searches a directory), a pattern that reads as a
    regex or as a wildcard. A tie stays `glob`, which is what every caller
    got before and what the existing fixtures expect.
    """
    keys = set(data.keys())
    grep_score = glob_score = 0

    grep_props, glob_props = _schema_props(tools, "grep"), _schema_props(tools, "glob")
    if grep_props and glob_props:
        grep_score += 2 * len(keys & (grep_props - glob_props))
        glob_score += 2 * len(keys & (glob_props - grep_props))
    if keys & _GREP_ONLY_KEYS:
        grep_score += 2

    path = data.get("path")
    if isinstance(path, str) and path.strip():
        last = re.split(r"[\\/]", path.strip().rstrip("\\/"))[-1]
        if _FILE_NAME.search(last) and not _GLOB_META.search(last):
            grep_score += 2

    pattern = data.get("pattern")
    if isinstance(pattern, str) and pattern:
        if _REGEX_SIGNS.search(pattern):
            grep_score += 1
        elif _GLOB_META.search(pattern):
            glob_score += 2
            if "/" in pattern or "**" in pattern:
                glob_score += 1
        elif _FILE_NAME.search(pattern):
            glob_score += 1     # `README.md`: a name, not a phrase
        else:
            grep_score += 1     # `E12`: a phrase, not a name

    return "grep" if grep_score > glob_score else "glob"


def is_potential_tool_call(buf: str, tools: list[dict] | None = None) -> bool:
    """Checks whether `buf` appears to contain an unclosed or ongoing tool call."""
    if not buf:
        return False
    brace_pos = buf.find("{")
    if brace_pos == -1:
        return False

    pre_text = buf[:brace_pos]
    if re.search(r'(?:[A-Za-z0-9_]|\bcall:\s*[A-Za-z0-9_.-]*)\s*$', pre_text):
        return False

    tail = buf[brace_pos:]
    m = re.search(r'^\s*\{\s*"([^"\\]+)"', tail)
    if not m:
        # If very short (e.g. `{"` or `{\n  "`), it may be the start of a key
        return len(tail.strip()) < 40

    first_key = m.group(1).strip()
    if first_key in COMMON_TOOL_KEYS:
        return True

    if tools:
        for t in tools:
            fn = (t or {}).get("function") or {}
            fn_name = fn.get("name")
            if fn_name and first_key == fn_name:
                return True
            props = ((fn.get("parameters") or {}).get("properties") or {})
            if first_key in props:
                return True

    return False


def resolve_tool_call_dict(
    data: dict[str, Any],
    tools: list[dict] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Resolve an already-parsed JSON dictionary into a recognized tool call."""
    if not isinstance(data, dict) or not data:
        return None

    import json
    from gateway.tool_schema import TOOL_NAME_ALIASES, TOOL_FAMILIES

    registered_names: set[str] = set()
    if tools:
        for t in tools:
            fn = (t or {}).get("function") or {}
            n = fn.get("name")
            if n:
                registered_names.add(n)

    def _map_to_registered(name: str) -> str:
        if not registered_names or name in registered_names:
            return name
        if name in TOOL_NAME_ALIASES:
            aliased = TOOL_NAME_ALIASES[name]
            if aliased in registered_names:
                return aliased
        for family in TOOL_FAMILIES:
            if name in family:
                for member in family:
                    if member in registered_names:
                        return member
        return name

    # 1. Explicit tool name: {"name": "...", "arguments": ...} / {"tool": ...} / {"function": ...}
    name_key = next((k for k in ("name", "function", "tool") if k in data), None)
    if name_key and isinstance(data[name_key], str):
        fn_name = data[name_key].strip()
        resolved_name = _map_to_registered(TOOL_NAME_ALIASES.get(fn_name, fn_name))
        if not registered_names or resolved_name in registered_names or resolved_name == "skill":
            args = data.get("arguments") or data.get("parameters") or data.get("input")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = None
            if not isinstance(args, dict):
                args = {k: v for k, v in data.items() if k not in ("name", "function", "tool")}
            return resolved_name, args

    # 2. OpenCode 2.0 native skill call: {"id": "<skill_id>"}
    if "id" in data and len(data) <= 2 and isinstance(data.get("id"), str):
        if not registered_names or "skill" in registered_names:
            return "skill", {"id": data["id"]}

    # 3. Known parameter signatures for OpenCode standard tools
    data_keys = set(data.keys())
    if "pattern" in data_keys:
        choice = choose_search_tool(data, tools)
        other = "glob" if choice == "grep" else "grep"
        for candidate in (choice, other):
            mapped = _map_to_registered(candidate)
            if not registered_names or mapped in registered_names:
                return mapped, data
        return _map_to_registered(choice), data
    if "command" in data_keys or "cmd" in data_keys or "script" in data_keys:
        return _map_to_registered("bash"), data
    if "patchText" in data_keys or "patch" in data_keys:
        return _map_to_registered("patch"), data
    if "oldString" in data_keys or "old_string" in data_keys or "old_text" in data_keys:
        return _map_to_registered("edit"), data
    if "content" in data_keys and ("path" in data_keys or "filePath" in data_keys or "file" in data_keys):
        return _map_to_registered("write"), data
    if ("path" in data_keys or "filePath" in data_keys or "file" in data_keys) and not ("content" in data_keys or "pattern" in data_keys or "oldString" in data_keys):
        return _map_to_registered("read"), data
    if "agent" in data_keys and ("prompt" in data_keys or "task" in data_keys):
        return _map_to_registered("subagent"), data
    if "questions" in data_keys or "question" in data_keys:
        return _map_to_registered("question"), data
    if "directory" in data_keys or "dir" in data_keys:
        return _map_to_registered("ls"), data

    # 4. Fallback: schema scoring against registered tools
    if tools:
        best_tool = None
        max_overlap = 0
        for t in tools:
            fn = (t or {}).get("function") or {}
            fn_name = fn.get("name") or ""
            if not fn_name:
                continue
            params = fn.get("parameters") or {}
            props = set((params.get("properties") or {}).keys())
            req = set(params.get("required") or [])

            overlap = len(data_keys & props)
            if overlap == 0:
                continue
            if req and not req.issubset(data_keys):
                continue
            if overlap > max_overlap:
                max_overlap = overlap
                best_tool = fn_name

        if best_tool and max_overlap >= 1:
            return best_tool, data

    return None


def _balanced_object_end(buf: str, start: int) -> int | None:
    """Index just past the `}` that closes the `{` at `start`, or None.

    String-aware, so a brace inside a quoted value does not count. None means
    the object is still open at the end of `buf` -- the one case in which
    waiting for the next chunk can help.
    """
    depth = 0
    in_str = escape = False
    for i in range(start, len(buf)):
        ch = buf[i]
        if escape:
            escape = False
        elif in_str:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def process_speculative_json(
    buf: str,
    tools: list[dict] | None = None,
) -> tuple[list[tuple[str, dict[str, Any]]], str, str]:
    """Scans `buf` for leading text and consecutive JSON tool calls.

    Returns:
      (tool_calls, content_to_emit, remaining_unprocessed_buf)
    """
    import json
    decoder = json.JSONDecoder()
    tool_calls: list[tuple[str, dict[str, Any]]] = []
    emitted_content = ""
    idx = 0
    n = len(buf)

    while idx < n:
        # Check for markdown code fences: ```json ... ```
        stripped_tail = buf[idx:].lstrip()
        if stripped_tail.startswith("```"):
            fence_start = buf.find("```", idx)
            fence_end = buf.find("```", fence_start + 3)
            if fence_end != -1:
                block = buf[fence_start:fence_end + 3]
                lines = block.splitlines()
                inner = "\n".join(lines[1:-1]).strip() if len(lines) >= 2 else ""
                tc = None
                try:
                    data = json.loads(inner)
                    tc = resolve_tool_call_dict(data, tools)
                except Exception:
                    try:
                        from gateway.tool_schema import repair_json
                        data = repair_json(inner)
                        tc = resolve_tool_call_dict(data, tools)
                    except Exception:
                        pass
                if tc:
                    emitted_content += buf[idx:fence_start]
                    tool_calls.append(tc)
                    idx = fence_end + 3
                    continue
            else:
                # Unclosed fence at end of buffer -> wait for next chunk
                break

        # Look for next '{'
        brace_pos = buf.find("{", idx)
        if brace_pos == -1:
            emitted_content += buf[idx:]
            idx = n
            break

        pre_text = buf[idx:brace_pos]
        # An attached identifier (e.g. `foo{` or `call:name{`) indicates a function call /
        # native syntax, not an anonymous bare JSON tool call. Leave it for prose or higher layers.
        if re.search(r'(?:[A-Za-z0-9_]|\bcall:\s*[A-Za-z0-9_.-]*)\s*$', pre_text):
            emitted_content += buf[idx:brace_pos + 1]
            idx = brace_pos + 1
            continue

        try:
            data, end_pos = decoder.raw_decode(buf, brace_pos)
            if isinstance(data, dict):
                tc = resolve_tool_call_dict(data, tools)
                if tc:
                    emitted_content += pre_text
                    tool_calls.append(tc)
                    idx = end_pos
                    continue
                else:
                    # Parsed JSON is not a tool call (e.g. user config). Emit as content.
                    emitted_content += buf[idx:end_pos]
                    idx = end_pos
                    continue
            else:
                emitted_content += buf[idx:end_pos]
                idx = end_pos
                continue
        except json.JSONDecodeError:
            end = _balanced_object_end(buf, brace_pos)
            if end is None:
                # Incomplete JSON at brace_pos -> wait for more chunks
                emitted_content += pre_text
                idx = brace_pos
                break
            # Closed but not JSON: no chunk will ever complete it. Waiting
            # anyway held everything after it -- prose AND the real tool call
            # -- until the turn ended, then sent it all as text. Measured
            # 2026-10-18: a summary quoting `{"version": "1.1.0", "params":
            # {...}}` swallowed the `shell` call that followed it.
            span = buf[brace_pos:end]
            tc = None
            if is_potential_tool_call(span, tools):
                try:
                    from gateway.tool_schema import repair_json
                    repaired = repair_json(span)
                    if isinstance(repaired, dict) and repaired:
                        tc = resolve_tool_call_dict(repaired, tools)
                except Exception:
                    tc = None
            if tc:
                emitted_content += pre_text
                tool_calls.append(tc)
            else:
                emitted_content += buf[idx:end]
            idx = end
            continue

    remaining = buf[idx:]
    return tool_calls, emitted_content, remaining


def parse_json_object(text: str) -> dict[str, Any] | None:
    """A JSON object out of text a model wrote, or None.

    Strict first; `repair_json` only when that fails. Shared by both FSMs for
    a JSON body inside `<function=NAME>`, which used to accept strict JSON
    only -- so a body carrying a raw Windows path (`"C:\\x\\core.py"`, an
    invalid escape) came out as a call with NO arguments at all.
    """
    try:
        data = json.loads(text)
    except Exception:
        try:
            from gateway.tool_schema import repair_json
            data = repair_json(text)
        except Exception:
            return None
    return data if isinstance(data, dict) else None


def resolve_json_tool_call(
    text: str,
    tools: list[dict] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Attempt to parse and resolve a JSON payload into a recognized tool call."""
    import json
    cleaned = extract_json_candidate(text)
    if not (cleaned.startswith("{") and cleaned.endswith("}")):
        s = cleaned.lstrip()
        if s.startswith("{"):
            try:
                data, _ = json.JSONDecoder().raw_decode(s)
            except Exception:
                return None
        else:
            return None
    else:
        try:
            data = json.loads(cleaned)
        except Exception:
            try:
                from gateway.tool_schema import repair_json
                data = repair_json(cleaned)
            except Exception:
                return None

    if not isinstance(data, dict):
        return None

    return resolve_tool_call_dict(data, tools)

