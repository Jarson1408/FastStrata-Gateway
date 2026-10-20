"""
Tool Schema Registry — centralized schema management, coercion, aliases, validation.

Replaces the following from _legacy.py:
  - _coerce_value()
  - _placeholder_for()
  - _fill_required_in_items()
  - fill_missing_required_args()
  - _repair_json()
  - _get_parameter_type()
  - _get_missing_args_str()

New capabilities:
  - Tool name alias resolution (Qwen → OpenCode naming)
  - Argument alias resolution (cmd → command, etc.)
  - Critical field protection (never auto-fill path, content, command, etc.)
  - Tool call validation
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from gateway.lodestar.tool_names import READ_CANDIDATES
from gateway.skills import resolve_skill_path

_log = logging.getLogger("gateway.tool_schema")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Critical fields — NEVER auto-fill with placeholders
CRITICAL_FIELDS = frozenset({
    "path", "filePath", "file_path", "absolutePath",
    "command", "content", "oldString", "newString",
    "old_string", "new_string", "patch", "patchText",
    "AbsolutePath", "TargetFile", "CodeContent",
    "questions",
})

# Tool name aliases: common Qwen hallucinations → real OpenCode tool names
TOOL_NAME_ALIASES: dict[str, str] = {
    "read_file": "read",
    "write_file": "write",
    "replace_in_file": "edit",
    "run_command": "bash",
    "find_files": "glob",
    "execute_command": "bash",
    "list_directory": "ls",
    "list_dir": "ls",
    "ask_question": "question",
    "ask_user": "question",
    "apply_patch": "patch",
}

# Tool families: interchangeable names across client generations (OpenCode 1.x vs 2.0).
# If the client declared any member, a call naming another member resolves to the declared one.
TOOL_FAMILIES: tuple[tuple[str, ...], ...] = (
    ("bash", "shell", "run_command", "execute_command", "exec"),
    ("task", "subagent"),
    ("patch", "apply_patch"),
    ("read", "read_file", "view_file", "get_file_content"),
    ("write", "write_file", "write_to_file"),
    ("edit", "replace_in_file", "replace_file_content"),
    ("glob", "find_files"),
    ("ls", "list_directory", "list_dir"),
    ("question", "ask_question", "ask_user"),
)

# Per-tool argument aliases: Qwen variant names → canonical names
ARG_ALIASES: dict[str, dict[str, str]] = {
    "bash": {
        "cmd": "command", "script": "command", "shell": "command",
        "input": "command",
    },
    "shell": {
        "cmd": "command", "script": "command", "shell": "command",
        "input": "command",
    },
    "edit": {
        "old_text": "oldString", "old": "oldString", "before": "oldString",
        "new_text": "newString", "new": "newString", "after": "newString",
    },
    "read": {
        "filename": "filePath", "file": "filePath",
        "filepath": "filePath", "path": "filePath",
    },
    "write": {
        "filename": "filePath", "file": "filePath",
        "filepath": "filePath",
    },
    "skill": {
        "name": "id",
        "skill_name": "id",
        "tool_call": "id",
    },
    "subagent": {
        "subagent_type": "agent",
        "type": "agent",
        "task": "prompt",
        "instructions": "prompt",
    },
    "task": {
        "agent": "subagent_type",
        "task": "prompt",
        "instructions": "prompt",
    },
    "patch": {
        "patch": "patchText",
        "patch_text": "patchText",
    },
}

# Blocked tool names (known Qwen hallucinations)
BLOCKED_TOOL_NAMES = frozenset({
    "invalid", "none", "null", "undefined", "function",
})


# ---------------------------------------------------------------------------
# ToolSchemaRegistry
# ---------------------------------------------------------------------------

class ToolSchemaRegistry:
    """Centralized tool schema management for a single request.

    Usage::

        registry = ToolSchemaRegistry(tools_from_client)
        name = registry.resolve_tool_name("read_file")  # → "read"
        args = registry.resolve_arguments("read", {"filepath": "/x"})  # → {"filePath": "/x"}
        ok, err = registry.validate_tool_call("read", args)
        registry.fill_missing_safe_args("bash", args)
    """

    def __init__(self, tools: list[dict]):
        self._tools = tools
        self._by_name: dict[str, dict] = {}  # name → full function schema
        self._params_by_name: dict[str, dict] = {}  # name → parameters schema
        self._props_by_name: dict[str, dict] = {}  # name → properties dict
        self._profile(tools)

    def _profile(self, tools: list[dict]) -> None:
        """Build name → schema index."""
        for t in tools:
            fn = (t or {}).get("function") or {}
            name = fn.get("name")
            if name:
                self._by_name[name] = fn
                params = fn.get("parameters") or {}
                self._params_by_name[name] = params
                self._props_by_name[name] = params.get("properties") or {}

    @property
    def tool_names(self) -> set[str]:
        """Set of registered tool names."""
        return set(self._by_name.keys())

    # -- Name resolution ---------------------------------------------------

    def resolve_tool_name(self, name: str) -> str | None:
        """Resolve a tool name, supporting aliases.

        Returns:
          - The canonical name if found or aliased.
          - None if the name is blocked or not found.
        """
        if name.lower() in BLOCKED_TOOL_NAMES:
            return None
        if name in self._by_name:
            return name
        # Check families (e.g. bash <-> shell, task <-> subagent, patch <-> apply_patch)
        for family in TOOL_FAMILIES:
            if name in family:
                for member in family:
                    if member in self._by_name:
                        _log.info("TOOL FAMILY ALIAS: %s → %s", name, member)
                        return member
        # Check alias
        canonical = TOOL_NAME_ALIASES.get(name)
        if canonical and canonical in self._by_name:
            _log.info("TOOL NAME ALIAS: %s → %s", name, canonical)
            return canonical
        return None

    # -- Argument resolution -----------------------------------------------

    def _rewrite_superpowers_command(self, cmd: str) -> str:
        """Rewrite superpowers SDD bash script commands to use the Python sdd_helper.py on Windows."""
        import shlex
        cmd_clean = cmd.strip()
        
        # Check and strip shell / PowerShell execution prefixes
        while True:
            cmd_clean_lower = cmd_clean.lower()
            if cmd_clean_lower.startswith("bash "):
                cmd_clean = cmd_clean[5:].strip()
            elif cmd_clean_lower.startswith("sh "):
                cmd_clean = cmd_clean[3:].strip()
            elif cmd_clean.startswith("&"):
                cmd_clean = cmd_clean[1:].strip()
            elif cmd_clean_lower.startswith(". "):
                cmd_clean = cmd_clean[2:].strip()
            else:
                break

        try:
            tokens = shlex.split(cmd_clean, posix=False)
        except Exception:
            tokens = cmd_clean.split(None, 1)

        if not tokens:
            return cmd

        # Strip surrounding quotes from tokens (since posix=False keeps them)
        tokens = [t.strip("'\"") for t in tokens]

        first_token = tokens[0]
        first_token_norm = first_token.replace("\\", "/")
        
        matched_script = None
        for script in ("task-brief", "sdd-workspace", "review-package"):
            if first_token_norm.endswith(f"/{script}") or first_token_norm == script or first_token_norm.endswith(f"/{script}.sh") or first_token_norm == f"{script}.sh":
                matched_script = script
                break
                
        if matched_script:
            args_list = []
            for arg in tokens[1:]:
                # Put back quotes for arguments containing spaces
                if " " in arg:
                    args_list.append(f'"{arg}"')
                else:
                    args_list.append(arg)
            args_str = " ".join(args_list)
            
            helper_path = str(Path(__file__).resolve().parent / "sdd_helper.py")
            rewritten = f'python "{helper_path}" {matched_script}'
            if args_str:
                rewritten += f" {args_str}"
            _log.info("SUPERPOWERS COMMAND REWRITE: %s -> %s", cmd, rewritten)
            return rewritten
            
        return cmd

    def resolve_arguments(self, tool_name: str, raw_args: dict[str, str]) -> dict[str, Any]:
        """Resolve argument aliases and coerce types.

        1. Apply per-tool argument aliases.
        2. Coerce each value based on schema type.
        """
        props = self._props_by_name.get(tool_name, {})
        arg_aliases = ARG_ALIASES.get(tool_name, {})

        if tool_name == "question":
            raw_args = heal_question_arguments(raw_args)

        resolved: dict[str, Any] = {}
        for key, val in raw_args.items():
            # Alias resolution
            canonical_key = arg_aliases.get(key, key)
            if key in props and canonical_key not in props:
                canonical_key = key
            elif canonical_key not in props and "path" in props and key in ("filePath", "filename", "file", "filepath"):
                canonical_key = "path"
            elif canonical_key not in props and "filePath" in props and key in ("path", "filename", "file", "filepath"):
                canonical_key = "filePath"
            # Schema lookup
            prop_schema = props.get(canonical_key)
            # Rewrite superpowers commands on Windows
            if tool_name in ("bash", "shell", "run_command") and canonical_key in ("command", "CommandLine") and isinstance(val, str):
                val = self._rewrite_superpowers_command(val)
            # Coerce
            resolved[canonical_key] = coerce_value(val, prop_schema)

        return resolved

    # -- Validation --------------------------------------------------------

    def validate_tool_call(self, name: str, args: dict) -> tuple[bool, str | None]:
        """Validate a tool call.

        Returns (True, None) if valid, or (False, error_message) if invalid.
        """
        if name.lower() in BLOCKED_TOOL_NAMES:
            return False, f"tool name '{name}' is blocked (known hallucination)"

        if name not in self._by_name:
            resolved = TOOL_NAME_ALIASES.get(name)
            if not resolved or resolved not in self._by_name:
                return False, f"tool '{name}' not found in schema"

        # Check critical fields — must not be empty
        for key, val in args.items():
            if key in CRITICAL_FIELDS:
                if val is None or (isinstance(val, str) and not val.strip()):
                    return False, f"critical field '{key}' is empty"

        return True, None

    # -- Missing arg filling -----------------------------------------------

    def fill_missing_safe_args(self, tool_name: str, args: dict) -> int:
        """Fill missing required args with safe placeholders.

        NEVER fills CRITICAL_FIELDS. Only fills safe fields like
        'description', 'priority', etc.

        Returns count of filled fields.
        """
        params = self._params_by_name.get(tool_name)
        if not params:
            return 0

        required = params.get("required") or []
        properties = params.get("properties") or {}
        filled = 0

        for r in required:
            if r not in args:
                if r in CRITICAL_FIELDS:
                    _log.warning(
                        "CRITICAL FIELD MISSING: tool=%s field=%s — NOT auto-filling",
                        tool_name, r,
                    )
                    continue
                args[r] = placeholder_for(r, args, properties.get(r) or {})
                filled += 1

        # Also fill required fields in nested array items
        for prop_name, prop_schema in properties.items():
            if not isinstance(prop_schema, dict):
                continue
            if prop_schema.get("type") != "array":
                continue
            n = _fill_required_in_items(
                args.get(prop_name),
                prop_schema.get("items") or {},
            )
            if n:
                filled += n

        return filled

    # -- Parameter type lookup ---------------------------------------------

    def get_param_type(self, tool_name: str, param_name: str) -> str:
        """Look up the JSON Schema type of a parameter. Fallback: 'string'."""
        props = self._props_by_name.get(tool_name, {})
        param_def = props.get(param_name) or {}
        return _resolve_schema_type(param_def)

    def get_prop_schema(self, tool_name: str, param_name: str) -> dict | None:
        """Return the full property schema for a parameter."""
        return self._props_by_name.get(tool_name, {}).get(param_name)

    # -- Structured tool call processing -----------------------------------

    def process_tool_calls(self, tool_calls: list[dict], allow_skills: bool = True, workspace_dir: str | None = None) -> int:
        """Process a list of tool calls: resolve names, coerce args, fill missing.

        Replaces _legacy.fill_missing_required_args with added name resolution.
        Filters out hallucinated tool calls (blocked or unknown names).
        Returns total count of filled fields.
        """
        filled = 0
        to_remove: list[int] = []

        for idx, call in enumerate(tool_calls):
            fn = call.get("function") or {}
            name = fn.get("name")
            if not name:
                to_remove.append(idx)
                continue

            if name == "skill" and allow_skills:
                if "skill" in self._by_name:
                    # Client natively declares `skill` tool (OpenCode 2.0)
                    resolved_name = "skill"
                else:
                    # Intercept skill tool call and rewrite to file read (OpenCode 1.x)
                    read_tool = next(
                        (c for c in READ_CANDIDATES if c in self._by_name), None)
                    
                    if read_tool:
                        raw_args_str = fn.get("arguments") or "{}"
                        try:
                            args = repair_json(raw_args_str)
                        except Exception:
                            args = {}
                        
                        skill_name = (args.get("skill_name") or args.get("name")
                                      or args.get("id") or args.get("tool_call"))
                        resolved_path = (resolve_skill_path(workspace_dir, skill_name)
                                         if skill_name else None)
                        if resolved_path is not None:
                            skill_path = str(resolved_path)
                            props = self._props_by_name.get(read_tool, {})
                            param_name = next(
                                (c for c in ("filePath", "AbsolutePath", "path") if c in props),
                                "filePath")
                            rewritten_args = {param_name: skill_path}
                            fn["name"] = read_tool
                            fn["arguments"] = json.dumps(rewritten_args, ensure_ascii=False)
                            name = read_tool
                            resolved_name = read_tool
                            _log.info("REWRITE SKILL CALL: skill(%s) -> %s(%s=%s)", skill_name, read_tool, param_name, skill_path)
                        else:
                            resolved_name = None
                    else:
                        resolved_name = None
            else:
                # Resolve tool name alias
                resolved_name = self.resolve_tool_name(name)
            if resolved_name is None:
                # Blocked or unknown tool — filter it out
                _log.warning(
                    "HALLUCINATED TOOL CALL FILTERED: name=%r (not in schema, removing)",
                    name,
                )
                to_remove.append(idx)
                continue
            if resolved_name != name:
                fn["name"] = resolved_name
                name = resolved_name

            # Parse and coerce arguments
            raw_args_str = fn.get("arguments") or "{}"
            try:
                args = repair_json(raw_args_str)
            except Exception:
                continue

            original_args = dict(args)
            if name == "question":
                args = heal_question_arguments(args)

            # Resolve argument aliases + coerce types
            props = self._props_by_name.get(name, {})
            arg_aliases = ARG_ALIASES.get(name, {})
            resolved_args: dict[str, Any] = {}
            for key, val in args.items():
                canonical_key = arg_aliases.get(key, key)
                if key in props and canonical_key not in props:
                    canonical_key = key
                elif canonical_key not in props and "path" in props and key in ("filePath", "filename", "file", "filepath"):
                    canonical_key = "path"
                elif canonical_key not in props and "filePath" in props and key in ("path", "filename", "file", "filepath"):
                    canonical_key = "filePath"
                if isinstance(val, str):
                    prop_schema = props.get(canonical_key)
                    resolved_args[canonical_key] = coerce_value(val, prop_schema)
                else:
                    resolved_args[canonical_key] = val

            # Fill missing safe args
            n = self.fill_missing_safe_args(name, resolved_args)
            filled += n

            # Write back
            if resolved_args != original_args or n > 0:
                fn["arguments"] = json.dumps(resolved_args, ensure_ascii=False)

        # Remove hallucinated calls (reverse order to preserve indices)
        for idx in reversed(to_remove):
            removed_name = (tool_calls[idx].get("function") or {}).get("name", "?")
            _log.info("Removing hallucinated tool call at index %d: %s", idx, removed_name)
            tool_calls.pop(idx)

        return filled


# ---------------------------------------------------------------------------
# Module-level functions (backward-compatible API)
# ---------------------------------------------------------------------------

def extract_json_from_string(s: str) -> Any:
    """Search for the first JSON-like container in a string and attempt to parse it."""
    s = s.strip()
    start_curly = s.find('{')
    start_bracket = s.find('[')
    
    start = -1
    end_char = ''
    if start_curly != -1 and (start_bracket == -1 or start_curly < start_bracket):
        start = start_curly
        end_char = '}'
    elif start_bracket != -1:
        start = start_bracket
        end_char = ']'
        
    if start != -1:
        end = s.rfind(end_char)
        if end != -1 and end > start:
            json_str = s[start:end+1]
            try:
                return json.loads(json_str)
            except Exception:
                pass
    return None


def heal_question_arguments(args: dict) -> dict:
    """Normalize and adapt question tool parameters into OpenCode's schema.

    OpenCode schema expects:
      {"questions": [{"question": str, "header": str, "options": [{"label": str, "description": str}], "multiple": bool}]}

    Models (especially Qwen in XML mode) often emit:
      1. Flat parameters: {"question": "...", "header": "...", "options": [...]}
      2. Single object: {"questions": {"question": "...", "options": [...]}}
      3. Stringified JSON: {"questions": "[{...}]"}
      4. String list options: options=["Opt 1", "Opt 2"]
    """
    if not isinstance(args, dict):
        return args

    result = dict(args)

    def _normalize_options(raw_opts: Any) -> list[dict[str, str]]:
        if isinstance(raw_opts, str):
            try:
                parsed = repair_json(raw_opts)
                if isinstance(parsed, list):
                    raw_opts = parsed
                elif isinstance(parsed, dict):
                    if "items" in parsed and isinstance(parsed["items"], list):
                        raw_opts = parsed["items"]
                    else:
                        raw_opts = [parsed]
            except Exception:
                lines = [line.strip().lstrip("0123456789.-*) ") for line in raw_opts.splitlines() if line.strip()]
                if lines:
                    raw_opts = lines

        if isinstance(raw_opts, dict):
            if "items" in raw_opts and isinstance(raw_opts["items"], list):
                raw_opts = raw_opts["items"]
            else:
                raw_opts = [raw_opts]

        if not isinstance(raw_opts, list):
            return []

        norm_list: list[dict[str, str]] = []
        for opt in raw_opts:
            if isinstance(opt, dict):
                label = str(opt.get("label") or opt.get("text") or opt.get("name") or opt.get("value") or "").strip()
                desc = str(opt.get("description") or opt.get("desc") or label).strip()
                if label:
                    norm_list.append({"label": label, "description": desc or label})
            elif isinstance(opt, str) and opt.strip():
                s = opt.strip()
                norm_list.append({"label": s, "description": s})
        return norm_list

    def _normalize_question_item(item: Any) -> dict[str, Any]:
        if not isinstance(item, dict):
            q_text = str(item).strip() if item else ""
            return {"question": q_text, "header": "Question", "options": [], "multiple": False}

        q_text = str(item.get("question") or item.get("text") or item.get("prompt") or item.get("query") or "").strip()
        header = str(item.get("header") or item.get("title") or "Question").strip()
        opts = _normalize_options(item.get("options") or item.get("choices") or [])
        multiple = bool(item.get("multiple", False))

        # Check for flattened option keys on item itself (e.g. when broken JSON had:
        # "options": {"label": ...}, {"label": ...} which json_repair flattened onto item)
        extra_label = str(item.get("label") or "").strip()
        if extra_label:
            extra_desc = str(item.get("description") or extra_label).strip()
            if not any(opt.get("label") == extra_label for opt in opts):
                opts.append({"label": extra_label, "description": extra_desc or extra_label})

        return {
            "question": q_text,
            "header": header or "Question",
            "options": opts,
            "multiple": multiple,
        }

    # Case A: 'questions' is already present
    if "questions" in result:
        raw_q = result["questions"]
        if isinstance(raw_q, str):
            try:
                parsed = repair_json(raw_q)
                if isinstance(parsed, (list, dict)):
                    raw_q = parsed
            except Exception:
                pass

        if isinstance(raw_q, dict):
            if "items" in raw_q and isinstance(raw_q["items"], list):
                raw_list = raw_q["items"]
            elif "questions" in raw_q and isinstance(raw_q["questions"], list):
                raw_list = raw_q["questions"]
            else:
                raw_list = [raw_q]
            result["questions"] = [_normalize_question_item(it) for it in raw_list]
        elif isinstance(raw_q, list):
            result["questions"] = [_normalize_question_item(it) for it in raw_q]
        else:
            result["questions"] = [_normalize_question_item(raw_q)]

        # Filter out empty dummy items if there are valid items
        if len(result["questions"]) > 1:
            result["questions"] = [
                q for q in result["questions"]
                if q.get("question") or q.get("options")
            ]

        for k in ("question", "header", "options", "multiple", "choices", "title", "prompt", "text"):
            result.pop(k, None)
        return result

    # Case B: Flat parameters
    q_text = result.pop("question", None) or result.pop("text", None) or result.pop("prompt", None) or result.pop("query", None)
    header = result.pop("header", None) or result.pop("title", None) or "Question"
    raw_opts = result.pop("options", None) or result.pop("choices", None) or []
    multiple = result.pop("multiple", False)

    if q_text is not None or raw_opts:
        item = {
            "question": str(q_text or "").strip(),
            "header": str(header or "Question").strip(),
            "options": _normalize_options(raw_opts),
            "multiple": bool(multiple),
        }
        result["questions"] = [item]

    return result


def coerce_value(raw: Any, prop_schema: dict | None = None) -> Any:
    """Coerce a parameter value based on the schema's declared type.

    Replaces _legacy._coerce_value.
    """
    if not isinstance(raw, str):
        return raw

    s = raw.strip()
    schema = prop_schema or {}
    schema_type = _resolve_schema_type(schema)

    if schema_type == "string":
        # Extract string value from nested JSON if model passed an object/dictionary
        # containing XML tag noise (e.g. Qwen's nested tag tool call noise).
        # Skip if the string is already a pure, clean JSON string (starts with { or [)
        if not (s.startswith("{") or s.startswith("[")):
            extracted = extract_json_from_string(s)
            if isinstance(extracted, dict):
                # Check for common key names
                for k in ("skill_name", "name", "command", "path", "value"):
                    if k in extracted and isinstance(extracted[k], str):
                        return extracted[k]
                # Fallback to first string value in dictionary
                for _k, val in extracted.items():
                    if isinstance(val, str):
                        return val
        return raw  # preserve whitespace for strings

    if schema_type in ("integer", "number"):
        try:
            return int(s) if schema_type == "integer" else float(s)
        except ValueError:
            return raw

    if schema_type == "boolean":
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        if s.lower() == "null":
            return None
        return raw

    if schema_type in ("array", "object"):
        parsed = None
        try:
            parsed = json.loads(s)
        except Exception:
            pass

        if parsed is not None:
            if schema_type == "array":
                return parsed if isinstance(parsed, list) else [parsed]
            if schema_type == "object" and isinstance(parsed, dict):
                return parsed

        # Try repairing malformed JSON (missing quotes, trailing commas, orphan values)
        try:
            import json_repair
            repaired = json_repair.repair_json(s, return_objects=True)
            if schema_type == "array":
                return repaired if isinstance(repaired, list) else [repaired]
            if schema_type == "object" and isinstance(repaired, dict):
                return repaired
        except Exception:
            pass

        extracted = extract_json_from_string(s)
        if extracted is not None:
            if schema_type == "array":
                return extracted if isinstance(extracted, list) else [extracted]
            if schema_type == "object" and isinstance(extracted, dict):
                return extracted

        if schema_type == "array":
            try:
                res = json.loads("[" + s + "]")
                if isinstance(res, list):
                    return res
            except Exception:
                pass
        return raw

    # No schema type — heuristic
    if not s:
        return ""
    if s.lower() == "null":
        return None
    if s[0] in "[{":
        try:
            return json.loads(s)
        except Exception:
            pass

        try:
            import json_repair
            repaired = json_repair.repair_json(s, return_objects=True)
            if isinstance(repaired, (dict, list)):
                return repaired
        except Exception:
            pass

    extracted = extract_json_from_string(s)
    if extracted is not None:
        return extracted

    return raw


def should_buffer_tool(name: str, tools: list[dict] | None = None) -> bool:
    """True if tool call arguments should be buffered and validated as complete JSON before emitting.

    Avoids streaming raw malformed JSON chunks to the client for tools taking structured
    data (arrays/objects, todowrite, task, etc.) where model typos (e.g. missing colons
    or keys) would cause JSON parse errors in the client.
    """
    if not name:
        return False
    norm_name = name.lower()
    if "todo" in norm_name:
        return True
    if not tools:
        return False
    for t in tools:
        fn = t.get("function") or {}
        if (fn.get("name") or "").lower() == norm_name:
            props = ((fn.get("parameters") or {}).get("properties") or {})
            for p_schema in props.values():
                if isinstance(p_schema, dict):
                    p_type = p_schema.get("type")
                    if p_type in ("array", "object") or "items" in p_schema:
                        return True
    return False



def placeholder_for(name: str, existing_args: dict, prop_schema: dict) -> Any:
    """Generate a safe placeholder for a missing required field.

    Replaces _legacy._placeholder_for.
    """
    if name == "description" and "command" in existing_args:
        cmd = str(existing_args["command"])[:80]
        return f"Run: {cmd}"
    if name == "description":
        return "(auto-filled by proxy: model omitted required description)"
    if name == "priority":
        enum = prop_schema.get("enum")
        if enum:
            return "medium" if "medium" in enum else enum[0]
        return "medium"
    enum = prop_schema.get("enum")
    if enum:
        return enum[0]
    t = prop_schema.get("type")
    if t == "string":
        return ""
    if t in ("integer", "number"):
        return 0
    if t == "boolean":
        return False
    if t == "array":
        return []
    if t == "object":
        return {}
    return ""


_JSON_STRING_LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"', re.DOTALL)
_RAW_DRIVE_PATH = re.compile(r'^[A-Za-z]:\\(?!\\)')


def _escape_raw_windows_paths(s: str) -> str:
    """Double the backslashes of string literals that are raw Windows paths.

    A model that writes `"C:\\Users\\user\\lib\\tps.py"` with SINGLE backslashes
    has written invalid JSON (`\\U`, `\\j` are not escapes), and `json_repair`
    then reads the valid-looking ones literally: `\\t` became a TAB and `\\n` a
    newline, so `lib\\tps54331_core.py` turned into `lib<TAB>ps54331_core.py`
    -- a path that cannot exist, handed on as a well-formed call. A literal
    that STARTS with a drive and a single backslash cannot be a correctly
    escaped path, so in that one shape every backslash is meant literally.
    An escaped quote stays an escaped quote.
    """
    def _fix(m: re.Match[str]) -> str:
        body = m.group(0)[1:-1]
        if not _RAW_DRIVE_PATH.match(body):
            return m.group(0)
        out, i = [], 0
        while i < len(body):
            ch = body[i]
            if ch == "\\" and i + 1 < len(body) and body[i + 1] == '"':
                out.append('\\"')
                i += 2
                continue
            out.append("\\\\" if ch == "\\" else ch)
            i += 1
        return '"' + "".join(out) + '"'
    return _JSON_STRING_LITERAL.sub(_fix, s)


def repair_json(s: str) -> dict:
    """Attempt a lightweight recovery of common JSON errors.

    Replaces _legacy._repair_json.
    """
    s = s.strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        pass

    # Raw Windows paths first, and everything below works on the fixed text:
    # json_repair would otherwise turn the `\t` in `lib\tps.py` into a TAB.
    s = _escape_raw_windows_paths(s)
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try json_repair (highly robust path for truncated or malformed JSON)
    try:
        import json_repair
        result = json_repair.repair_json(s, return_objects=True)
        if isinstance(result, dict):
            return result
        elif isinstance(result, list):
            return {"items": result}
    except Exception as e:
        _log.warning("json_repair failed: %s", e)

    # Fallback to legacy regex cleanups
    cleaned = re.sub(r',\s*([}\]])', r'\1', s)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    if s.startswith("{") and not s.endswith("}"):
        try:
            return json.loads(s + "}")
        except Exception:
            pass
    _log.warning("JSON Auto-Recovery failed for: %r", s)
    return {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_schema_type(schema: dict) -> str:
    """Resolve the effective JSON Schema type from a property definition."""
    t = schema.get("type", "string")
    if isinstance(t, list):
        non_null = [x for x in t if x and x != "null"]
        t = non_null[0] if non_null else "string"
    if t == "string" and "items" in schema:
        return "array"
    if not schema.get("type") and "items" in schema:
        return "array"
    if not schema.get("type") and "properties" in schema:
        return "object"
    return str(t)


def _fill_required_in_items(value: Any, item_schema: dict) -> int:
    """Fill required fields in array items."""
    if not isinstance(value, list) or not isinstance(item_schema, dict):
        return 0
    required = item_schema.get("required") or []
    if not required:
        return 0
    properties = item_schema.get("properties") or {}
    filled = 0
    for elem in value:
        if not isinstance(elem, dict):
            continue
        for r in required:
            if r not in elem:
                if r in CRITICAL_FIELDS:
                    continue  # never auto-fill critical nested fields
                elem[r] = placeholder_for(r, elem, properties.get(r) or {})
                filled += 1
    return filled
