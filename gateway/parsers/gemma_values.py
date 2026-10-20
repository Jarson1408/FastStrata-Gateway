"""Gemma 4's native tool syntax, in both directions.

Gemma 4 is trained on its own tool protocol, written by its chat template:

    <|tool>declaration:read{description:<|"|>Read a file<|"|>,parameters:{...}}<tool|>
    <|tool_call>call:read{filePath:<|"|>C:/x.txt<|"|>,limit:50}<tool_call|>
    <|tool_response>response:read{value:<|"|>...file...<|"|>}<tool_response|>

It is not JSON: strings are delimited by the special token `<|"|>` (id 52)
with no escaping inside, keys inside a call are bare, and objects are sorted
by key the way Jinja's `dictsort` sorts them (case-insensitively).

The FORMATTERS here are ports of the template's macros (`format_argument`,
`format_parameters`, `format_function_declaration`) -- quirks included, such
as `,properties:{` opening with a comma whether or not anything preceded it --
because the declarations are what the model was trained to read, and
`test_scripts/check_gemma4_prompt.py` compares them byte for byte with the
template. The PARSER is the inverse, and tolerant: it also takes JSON-style
`"..."` strings and quoted keys, since a model that half-remembers JSON is a
model whose call should still be read rather than dropped.

Torch-free and dependency-free, so the round trip is unit-tested:
`parse_args(format_call_args(x)) == x`.
"""
from __future__ import annotations

from typing import Any

#: The string delimiter. One special token, three visible characters.
Q = '<|"|>'

_STANDARD_KEYS = ("description", "type", "properties", "required", "nullable")


def _dictsort(mapping: dict) -> list[tuple[Any, Any]]:
    """Jinja's `dictsort`: by key, case-insensitively for strings."""
    return sorted(mapping.items(),
                  key=lambda kv: kv[0].lower() if isinstance(kv[0], str) else kv[0])


def _upper(value: Any) -> str:
    return str(value).upper()


# ---------------------------------------------------------------------------
# Formatting (gateway -> model)
# ---------------------------------------------------------------------------

def format_argument(argument: Any, escape_keys: bool = True) -> str:
    """The template's `format_argument` macro."""
    if isinstance(argument, str):
        return Q + argument + Q
    if isinstance(argument, bool):
        return "true" if argument else "false"
    if isinstance(argument, dict):
        parts = []
        for key, value in _dictsort(argument):
            k = Q + str(key) + Q if escape_keys else str(key)
            parts.append(k + ":" + format_argument(value, escape_keys=escape_keys))
        return "{" + ",".join(parts) + "}"
    if isinstance(argument, (list, tuple)):
        return "[" + ",".join(format_argument(v, escape_keys=escape_keys)
                              for v in argument) + "]"
    # Numbers render as Jinja renders them; None as Jinja's "None".
    return str(argument)


def _format_required(items: list) -> str:
    return ",".join(Q + str(item) + Q for item in items)


def format_parameters(properties: dict, _required: list | None = None) -> str:
    """The template's `format_parameters` macro, quirks and all.

    `_required` mirrors the macro's signature, which takes it and never reads
    it -- the required list is written by the CALLER, at each nesting level.
    """
    out = []
    found_first = False
    for key, value in _dictsort(properties or {}):
        if key in _STANDARD_KEYS:
            continue
        if not isinstance(value, dict):
            value = {}
        add_comma = False
        if found_first:
            out.append(",")
        found_first = True
        out.append(f"{key}:{{")
        if value.get("description"):
            out.append("description:" + Q + str(value["description"]) + Q)
            add_comma = True
        if value.get("nullable"):
            if add_comma:
                out.append(",")
            else:
                add_comma = True
            out.append("nullable:true")
        vtype = _upper(value.get("type", ""))
        if vtype == "STRING":
            if value.get("enum"):
                if add_comma:
                    out.append(",")
                else:
                    add_comma = True
                out.append("enum:" + format_argument(value["enum"]))
        elif vtype == "OBJECT":
            out.append(",properties:{")
            props = value.get("properties")
            if isinstance(props, dict):
                out.append(format_parameters(props, value.get("required") or []))
            else:
                out.append(format_parameters(value, value.get("required") or []))
            out.append("}")
            if value.get("required"):
                out.append(",required:[" + _format_required(value["required"]) + "]")
        elif vtype == "ARRAY":
            items = value.get("items")
            if isinstance(items, dict) and items:
                out.append(",items:{")
                found_item = False
                for item_key, item_value in _dictsort(items):
                    if item_value is None:
                        continue
                    if found_item:
                        out.append(",")
                    found_item = True
                    if item_key == "properties":
                        out.append("properties:{")
                        if isinstance(item_value, dict):
                            out.append(format_parameters(item_value,
                                                         items.get("required") or []))
                        out.append("}")
                    elif item_key == "required":
                        out.append("required:[" + _format_required(item_value) + "]")
                    elif item_key == "type":
                        if isinstance(item_value, str):
                            out.append("type:" + format_argument(item_value.upper()))
                        else:
                            out.append("type:" + format_argument(
                                [str(t).upper() for t in item_value]))
                    else:
                        out.append(f"{item_key}:" + format_argument(item_value))
                out.append("}")
        if add_comma:
            out.append(",")
        out.append("type:" + Q + vtype + Q + "}")
    return "".join(out)


def format_declaration(tool: dict) -> str:
    """The template's `format_function_declaration`, already trimmed."""
    fn = tool.get("function", tool) if isinstance(tool, dict) else {}
    out = ["declaration:" + str(fn.get("name", "")) + "{description:" + Q
           + str(fn.get("description") or "") + Q]
    params = fn.get("parameters")
    if params:
        out.append(",parameters:{")
        if params.get("properties"):
            out.append("properties:{" + format_parameters(
                params["properties"], params.get("required")) + "},")
        if params.get("required"):
            out.append("required:[" + _format_required(params["required"]) + "],")
        if params.get("type"):
            out.append("type:" + Q + _upper(params["type"]) + Q + "}")
    out.append("}")
    return "".join(out).strip()


def format_call_args(arguments: Any) -> str:
    """The `{...}` of a `call:NAME{...}`: keys bare, values formatted."""
    if isinstance(arguments, dict):
        return "{" + ",".join(f"{k}:" + format_argument(v, escape_keys=False)
                              for k, v in _dictsort(arguments)) + "}"
    return "{" + str(arguments or "") + "}"


def format_response(name: str, response: Any) -> str:
    """The body of a `<|tool_response>`: `response:NAME{...}`."""
    if isinstance(response, dict):
        body = ",".join(f"{k}:" + format_argument(v, escape_keys=False)
                        for k, v in _dictsort(response))
        return f"response:{name or 'unknown'}{{{body}}}"
    return (f"response:{name or 'unknown'}{{value:"
            + format_argument(response, escape_keys=False) + "}")


# ---------------------------------------------------------------------------
# Parsing (model -> gateway)
# ---------------------------------------------------------------------------

class GemmaSyntaxError(ValueError):
    """The call body is not something the model's syntax can produce."""


class _Reader:
    def __init__(self, text: str):
        self.s = text
        self.i = 0

    def ws(self) -> None:
        while self.i < len(self.s) and self.s[self.i] in " \t\r\n":
            self.i += 1

    def peek(self, token: str) -> bool:
        return self.s.startswith(token, self.i)

    def expect(self, token: str) -> None:
        self.ws()
        if not self.peek(token):
            raise GemmaSyntaxError(
                f"expected {token!r} at {self.i}: {self.s[self.i:self.i + 20]!r}")
        self.i += len(token)

    def value(self) -> Any:
        self.ws()
        if self.peek(Q):
            return self.gemma_string()
        if self.peek('"'):
            return self.json_string()
        if self.peek("{"):
            return self.obj()
        if self.peek("["):
            return self.arr()
        return self.scalar()

    def gemma_string(self) -> str:
        self.i += len(Q)
        end = self.s.find(Q, self.i)
        if end == -1:
            raise GemmaSyntaxError("unterminated <|\"|> string")
        text = self.s[self.i:end]
        self.i = end + len(Q)
        return text

    def json_string(self) -> str:
        import json
        start = self.i
        self.i += 1
        while self.i < len(self.s):
            ch = self.s[self.i]
            if ch == "\\":
                self.i += 2
                continue
            if ch == '"':
                self.i += 1
                return json.loads(self.s[start:self.i])
            self.i += 1
        raise GemmaSyntaxError("unterminated JSON string")

    def key(self) -> str:
        self.ws()
        if self.peek(Q):
            return self.gemma_string()
        if self.peek('"'):
            return self.json_string()
        start = self.i
        while self.i < len(self.s) and self.s[self.i] not in ":,}] \t\r\n":
            self.i += 1
        if start == self.i:
            raise GemmaSyntaxError(f"empty key at {start}")
        return self.s[start:self.i]

    def obj(self) -> dict:
        self.expect("{")
        out: dict = {}
        self.ws()
        if self.peek("}"):
            self.i += 1
            return out
        while True:
            k = self.key()
            self.expect(":")
            out[k] = self.value()
            self.ws()
            if self.peek(","):
                self.i += 1
                continue
            self.expect("}")
            return out

    def arr(self) -> list:
        self.expect("[")
        out: list = []
        self.ws()
        if self.peek("]"):
            self.i += 1
            return out
        while True:
            out.append(self.value())
            self.ws()
            if self.peek(","):
                self.i += 1
                continue
            self.expect("]")
            return out

    def scalar(self) -> Any:
        start = self.i
        while self.i < len(self.s) and self.s[self.i] not in ",}]\r\n":
            self.i += 1
        word = self.s[start:self.i].strip()
        if not word:
            raise GemmaSyntaxError(f"empty value at {start}")
        low = word.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        if low in ("null", "none"):
            return None
        try:
            return int(word)
        except ValueError:
            pass
        try:
            return float(word)
        except ValueError:
            pass
        # A bare word the model did not quote. Taken as the string it is
        # rather than refused: dropping a call over a missing delimiter
        # throws away an intent the model stated plainly.
        return word


def parse_args(body: str) -> dict:
    """`{filePath:<|"|>x<|"|>,limit:5}` -> `{"filePath": "x", "limit": 5}`.

    Raises GemmaSyntaxError on anything that is not one complete object,
    so a caller can emit the text as it was rather than guess.
    """
    reader = _Reader(body)
    out = reader.obj()
    reader.ws()
    if reader.i != len(reader.s):
        raise GemmaSyntaxError(f"trailing text after the object: {reader.s[reader.i:][:40]!r}")
    return out


def split_object(text: str) -> tuple[str | None, str]:
    """(the first complete `{...}` in Gemma syntax, the rest), or (None, text).

    Brace counting that skips `<|"|>` strings, whose contents are raw -- a
    `write` of source code is full of braces, and a naive scan to the first
    `}` would truncate every one of them.
    """
    if not text.startswith("{"):
        return None, text
    depth = 0
    i = 0
    while i < len(text):
        if text.startswith(Q, i):
            end = text.find(Q, i + len(Q))
            if end == -1:
                return None, text
            i = end + len(Q)
            continue
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[:i + 1], text[i + 1:]
        i += 1
    return None, text
