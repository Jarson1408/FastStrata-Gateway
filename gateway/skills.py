"""
Locating a skill's SKILL.md on disk.

The client has no `skill` tool — loading a skill means reading its file — so
both the streaming and the non-streaming path rewrite `skill(name)` into a
read. Until 2026-07-29 each did that with its own verbatim copy of the search,
including a hardcoded `C:/Users/user/.gemini/config/skills` fallback in four
places, which made the gateway unusable for anyone else and silently wrong if
that directory ever moved.

Search order: the workspace first (a project's own skills win over global
ones), then the user's skill directories. `LOCAL_LLM_GATEWAY_SKILL_DIRS` (a
path-separator-delimited list) prepends extra roots for anyone whose client
keeps skills elsewhere.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

__all__ = ["skill_roots", "resolve_skill_path"]

_log = logging.getLogger("gateway.skills")

SKILL_DIRS_ENV = "LOCAL_LLM_GATEWAY_SKILL_DIRS"

# Relative to the workspace the request came from.
_WORKSPACE_SUBDIRS = (
    (".agents", "skills"),
    (".context", "skills"),
    (".opencode", "skills"),
    (".claude", "skills"),
    ("skills",),
)

# Relative to the user's home directory.
_HOME_SUBDIRS = (
    (".gemini", "config", "skills"),
    (".gemini", "antigravity-ide", "skills"),
    (".claude", "skills"),
    (".config", "opencode", "skills"),
    (".opencode", "skills"),
    (".cache", "opencode", "skills"),
)


def skill_roots(workspace_dir: str | Path | None) -> list[Path]:
    """Directories to search, most specific first."""
    roots: list[Path] = []

    for raw in os.environ.get(SKILL_DIRS_ENV, "").split(os.pathsep):
        raw = raw.strip()
        if raw:
            roots.append(Path(raw).expanduser())

    if workspace_dir:
        workspace = Path(workspace_dir)
        roots.extend(workspace.joinpath(*parts) for parts in _WORKSPACE_SUBDIRS)

    home = Path.home()
    roots.extend(home.joinpath(*parts) for parts in _HOME_SUBDIRS)

    # Search in plugin skill directories: .gemini/config/plugins/*/skills
    try:
        plugins_dir = home / ".gemini" / "config" / "plugins"
        if plugins_dir.is_dir():
            for plugin in plugins_dir.iterdir():
                if plugin.is_dir():
                    candidate = plugin / "skills"
                    if candidate.is_dir():
                        roots.append(candidate)
    except OSError:
        pass

    # Search in builtin skills: .gemini/antigravity-ide/builtin/skills
    builtin_skills = home / ".gemini" / "antigravity-ide" / "builtin" / "skills"
    if builtin_skills.is_dir():
        roots.append(builtin_skills)

    return roots


def resolve_skill_path(
    workspace_dir: str | Path | None,
    skill_name: str,
    fallback: bool = True,
) -> Path | None:
    """Path to `<skill_name>/SKILL.md`, or None.

    A plugin-qualified name ("anthropic-skills:pdf") resolves on its last
    segment, which is how clients name the directory.

    With `fallback` (the default) an unfound skill still returns a path — the
    best-guess location under the first root. Emitting a read of a file that
    does not exist gives the model a plain "not found" it can act on, which is
    more useful than the silence of dropping the call. Pass fallback=False when
    the caller would rather handle the miss itself.
    """
    clean_name = str(skill_name).split(":")[-1].strip()
    if not clean_name:
        return None

    roots = skill_roots(workspace_dir)
    for root in roots:
        candidate = root / clean_name / "SKILL.md"
        if candidate.is_file():
            return candidate

    if not fallback or not roots:
        _log.warning("SKILL NOT FOUND: %r searched %d root(s)", clean_name, len(roots))
        return None

    guess = roots[0] / clean_name / "SKILL.md"
    _log.warning("SKILL NOT FOUND: %r — falling back to %s so the model sees a "
                 "readable error instead of silence", clean_name, guess)
    return guess
