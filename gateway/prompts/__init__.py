"""
The prompts the gateway injects, as files rather than string literals.

Every instruction the model sees from the gateway used to be a literal inside
a .py file — nine of them, spread across normalizer.py, context_injector.py,
todo.py, synthesis.py and app.py. Changing one word therefore meant a code
commit and an application release, and there was no way to answer "which
version of the prompt produced that behaviour?".

`.context/LODESTAR.md` already showed the right shape: an external, versioned
document loaded at runtime. This module applies it to the rest.

Prompts are Markdown with `{placeholder}` fields filled by str.format. They are
read once and cached — these files do not change while the gateway runs.

VERSION is logged with every request (`prompts_version` in the audit event), so
a change in model behaviour can be tied to the prompt revision that caused it.
Bump it whenever a prompt file changes.
"""
from __future__ import annotations

import functools
from pathlib import Path

__all__ = ["get", "prompts_version", "PROMPT_DIR"]

PROMPT_DIR = Path(__file__).resolve().parent
_VERSION_FILE = PROMPT_DIR / "VERSION"


@functools.lru_cache(maxsize=None)
def _read(name: str) -> str:
    path = PROMPT_DIR / f"{name}.md"
    if not path.is_file():
        raise KeyError(f"no prompt named {name!r} in {PROMPT_DIR}")
    # Trailing newlines are an artefact of the file, not of the prompt.
    return path.read_text(encoding="utf-8").rstrip("\n")


def get(name: str, /, **params: object) -> str:
    """Render a prompt.

    Raises KeyError for an unknown prompt name and, via str.format, for a
    missing placeholder — both are programming errors worth failing loudly on
    rather than shipping a prompt with a literal `{ctx_path}` in it.

    `name` is positional-only: without that, a prompt file containing the
    placeholder `{name}` cannot be rendered at all, because the keyword
    collides with this function's own parameter and str.format never sees it
    (`TypeError: got multiple values for argument 'name'`). The trap is
    invisible until someone writes that placeholder, and the fix belongs here
    rather than in a rule about what prompts may be called.
    """
    template = _read(name)
    if not params:
        return template
    return template.format(**params)


@functools.lru_cache(maxsize=1)
def prompts_version() -> str:
    """Contents of VERSION, or "0.0.0" when it is missing."""
    if not _VERSION_FILE.is_file():
        return "0.0.0"
    return _VERSION_FILE.read_text(encoding="utf-8").strip() or "0.0.0"
