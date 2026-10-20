"""
Safe construction of the shell commands the gateway synthesizes.

The gateway does not run tools itself — the client does. But it *synthesizes*
tool calls the model never asked for, including `bash`, and the client
executes those unattended. That inverts the usual trust boundary: the model
is not the decision point, the gateway is. So any text that ends up inside a
synthesized command has to be treated as hostile input.

The 2026-07-29 audit found the previous sanitization (collapse whitespace,
replace `"` with `'`) insufficient: inside double quotes bash still expands
`$(...)` and backticks, so a task description containing `$(...)` became
arbitrary command execution on the user's machine.

The defence is stripping, not quoting. Shell metacharacters are removed
outright — a semantic-search query has no legitimate use for `$`, backticks,
`;`, `|`, `&`, redirects or newlines — and only then is the result wrapped in
double quotes.

Quoting alone would not do: `shlex.quote` emits single-quoted strings, which
cmd.exe does not understand, and the client that runs these commands may be
on bash, PowerShell or cmd depending on the platform. Double quotes are the
one form all three accept, and they are safe precisely because sanitization
already guarantees no metacharacter survives to be expanded inside them.
"""
from __future__ import annotations

import re
from pathlib import Path

__all__ = ["sanitize_search_query", "build_embed_search_command"]

# Characters that give a shell control flow, substitution or redirection.
# Quotes are included because the result is embedded in a quoted argument.
_SHELL_META_RE = re.compile(r"""[$`\\;|&<>(){}\[\]!#\n\r\t'"]""")

_WHITESPACE_RE = re.compile(r"\s+")


def sanitize_search_query(raw: str | None, max_len: int = 150) -> str | None:
    """Reduce free text to something safe to pass as a search argument.

    Returns None when nothing usable survives — the caller must then skip the
    synthesis entirely rather than emit a command with an empty query.
    """
    if not raw:
        return None
    cleaned = _SHELL_META_RE.sub(" ", str(raw))
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip()
    # A query of punctuation only is not worth searching for.
    if not re.search(r"\w", cleaned):
        return None
    return cleaned or None


def build_embed_search_command(embed_path: str | Path, query: str | None) -> str | None:
    """Build `python "<embed.py>" search "<query>"`.

    Returns None when the query does not survive sanitization, so callers can
    fall back to not synthesizing a call at all rather than running a search
    for nothing.
    """
    safe_query = sanitize_search_query(query)
    if safe_query is None:
        return None
    # The path comes from the gateway (workspace dir + fixed suffix), never
    # from the model, so it needs quoting for spaces but not sanitizing.
    return f'python "{embed_path}" search "{safe_query}"'
