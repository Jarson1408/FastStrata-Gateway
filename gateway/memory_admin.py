"""Semantic-memory administration helpers for the GUI dashboard.

The per-project semantic memory lives in `.context/semantic-memory/memory.db`
(a SQLite DB written by `embed.py`). The dashboard's Memory Manager needs to
browse, delete and export lessons WITHOUT loading the sentence-transformer
model, so these helpers talk to SQLite directly. Import (which needs fresh
embeddings) is delegated to the project's own `embed.py` by the GUI.

All functions are pure and defensive: a missing DB yields an empty result,
never an exception, so the dashboard degrades gracefully on a fresh project.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemoryRow:
    id: int
    category: str
    text: str
    session_id: str
    created_at: str


def resolve_db(project_dir: str | Path) -> Path:
    """Return the memory.db path for a workspace directory."""
    return Path(project_dir) / ".context" / "semantic-memory" / "memory.db"


def resolve_embed_script(project_dir: str | Path) -> Path:
    """Return the embed.py path for a workspace directory."""
    return Path(project_dir) / ".context" / "semantic-memory" / "embed.py"


# The schema version embed.py migrates to. Kept here as a literal rather than
# imported: memory_admin talks straight to SQLite precisely so the dashboard
# does not have to load embed.py (and with it sentence-transformers).
KNOWN_SCHEMA_VERSION = 3


#: What `check_schema_version` answers for a file that is THERE and cannot be
#: read: corrupt, truncated, locked by another process, or not a database at
#: all. A distinct value because `None` already means "no database", and
#: conflating the two told a user with a locked file to "update the gateway"
#: -- advice that can never help. Unreadable is its own outcome.
UNREADABLE = -1


def check_schema_version(db_path: str | Path) -> int | None:
    """Schema version of a memory.db, or None when there is no database.

    Two readers share this schema — embed.py writes it, this module reads it —
    and every query here swallows sqlite3.Error and returns empty. That means a
    schema the dashboard does not understand looks exactly like an empty
    database, so the GUI reports "0 lessons" for a file full of them. This lets
    it tell the difference.

    Returns 0 for a database that exists but has no schema_version row and no
    memories table, and `UNREADABLE` for a file that exists and cannot be
    opened or queried.
    """
    path = Path(db_path)
    if not path.is_file():
        return None
    try:
        conn = sqlite3.connect(str(path))
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            ).fetchone()
            if row is not None:
                version_row = conn.execute("SELECT version FROM schema_version").fetchone()
                if version_row is not None:
                    return int(version_row[0])
            # Predates schema_version. Which of the first two steps it saw is
            # readable from the columns: session_id is what migration 2 adds.
            if not _has_memories_table(conn):
                return 0
            cols = {row[1] for row in
                    conn.execute("PRAGMA table_info(memories)").fetchall()}
            return 2 if "session_id" in cols else 1
        finally:
            conn.close()
    except (sqlite3.Error, ValueError, TypeError):
        # The file is there; SQLite could not read it. That is a different
        # fact from "there is no database", and the caller needs to tell them
        # apart -- see UNREADABLE.
        return UNREADABLE


def schema_is_supported(db_path: str | Path) -> bool:
    """False when the file was written by a newer gateway than this one."""
    version = check_schema_version(db_path)
    return version is not None and version <= KNOWN_SCHEMA_VERSION


def _has_memories_table(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
    ).fetchone()
    return row is not None


def list_memories(db_path: str | Path) -> list[MemoryRow]:
    """Return all memories newest-first. Empty list if the DB is absent/empty."""
    path = Path(db_path)
    if not path.is_file():
        return []
    try:
        conn = sqlite3.connect(str(path))
        try:
            if not _has_memories_table(conn):
                return []
            rows = conn.execute(
                "SELECT id, category, text, "
                "COALESCE(session_id, 'unknown'), COALESCE(created_at, '') "
                "FROM memories ORDER BY datetime(created_at) DESC, id DESC"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return [MemoryRow(r[0], r[1], r[2], r[3], r[4]) for r in rows]


def describe_database(db_path: str | Path) -> tuple[str, list[MemoryRow]]:
    """Return (what to tell the user, the rows to show) for one memory.db.

    This exists because `schema_is_supported` did not: it was written, tested
    and called from nowhere, so the protection it provides never once ran in
    production (R-2026-01). Every query in this module swallows `sqlite3.Error`
    and returns empty, which means a database in a schema this build does not
    understand is indistinguishable from an empty one — and "the semantic
    memory is empty" is a natural and false conclusion to draw from a dashboard
    showing zero lessons.

    The decision lives here rather than in the widget for the same reason the
    bug survived: a check inside the GUI is a check nothing can test.
    """
    path = Path(db_path)
    if not path.is_file():
        return f"No memory database found at {path}", []

    version = check_schema_version(path)
    if version == UNREADABLE:
        # **Not a schema problem, and saying so mattered.** This branch used to
        # fall into the one below and report "schema version None, newer than
        # this gateway understands -- update the gateway", which is advice that
        # cannot help: the commonest cause is the file being held open by
        # embed.py mid-write, and the next is a truncated copy. A reader who
        # follows that message upgrades and sees the same thing.
        return (
            f"Cannot read {path}: it exists, SQLite could not open it. "
            f"Usually embed.py is writing to it right now — try again. "
            f"Otherwise the file is truncated or is not a database.",
            [],
        )

    if not schema_is_supported(path):
        return (
            f"Database at {path} is in schema version {version}, newer than "
            f"this gateway understands (up to {KNOWN_SCHEMA_VERSION}). "
            f"Its lessons are NOT shown — update the gateway rather than "
            f"treating this as an empty memory.",
            [],
        )

    rows = list_memories(path)
    return f"{len(rows)} lesson(s) in {path}", rows


def count_memories(db_path: str | Path) -> int:
    path = Path(db_path)
    if not path.is_file():
        return 0
    try:
        conn = sqlite3.connect(str(path))
        try:
            if not _has_memories_table(conn):
                return 0
            return int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def delete_memory(db_path: str | Path, mem_id: int) -> bool:
    """Delete one memory by id. Returns True if a row was removed."""
    path = Path(db_path)
    if not path.is_file():
        return False
    try:
        conn = sqlite3.connect(str(path))
        try:
            if not _has_memories_table(conn):
                return False
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (int(mem_id),))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
    except (sqlite3.Error, ValueError, TypeError):
        return False


def export_memories(db_path: str | Path, out_path: str | Path) -> int:
    """Export all memories (without embeddings) to a JSON file. Returns count."""
    rows = list_memories(db_path)
    payload = [
        {
            "category": r.category,
            "text": r.text,
            "session_id": r.session_id,
            "created_at": r.created_at,
        }
        for r in rows
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return len(payload)


def redact_memories(
    db_path: str | Path,
    rules: "list[tuple[str, str]]",
    embed_fn: "object | None" = None,
) -> list[int]:
    """Rewrite stored lesson text through `rules`. Returns the ids changed.

    A lesson is written while working on something, and what that something
    was called leaks into it -- a client's project codename, a document title.
    The engineering the lesson carries does not depend on either, so the label
    can go without losing anything. Measured 2026-09-12 on this repository's
    own memory: 96 lessons, of which three named a client project and none
    carried a credential, a hostname or an address.

    `rules` are (pattern, replacement) pairs applied in order, which is how
    they stay the SAME vocabulary the release redaction uses. One list of
    substitutions, so a lesson that is later published matches the prose that
    was published beside it.

    **The embedding is recomputed, or it is a lie about the row.** A vector
    derived from text that no longer exists is not a disclosure -- it is not
    readable -- but it is exactly the silent inconsistency this project keeps
    finding in other people's caches. `embed_fn` is injected rather than
    imported because this module is what the GUI uses, and it must never pull
    the sentence-transformer model onto the GUI thread. Pass None to update the
    text and leave the vectors alone, which is the honest choice only for a
    caller that knows it is about to reimport.
    """
    import re
    import struct

    path = Path(db_path)
    if not path.is_file():
        return []
    compiled = [(re.compile(p), r) for p, r in rules]

    changed: list[int] = []
    try:
        conn = sqlite3.connect(str(path))
        try:
            if not _has_memories_table(conn):
                return []
            rows = conn.execute("SELECT id, text FROM memories").fetchall()
            for mem_id, text in rows:
                new = text
                for pattern, replacement in compiled:
                    new = pattern.sub(replacement, new)
                if new == text:
                    continue
                if embed_fn is None:
                    conn.execute("UPDATE memories SET text = ? WHERE id = ?",
                                 (new, mem_id))
                else:
                    vector = embed_fn(new)  # type: ignore[operator]
                    blob = struct.pack(f"{len(vector)}f", *vector)
                    conn.execute(
                        "UPDATE memories SET text = ?, embedding = ? WHERE id = ?",
                        (new, blob, mem_id))
                changed.append(int(mem_id))
            conn.commit()
        finally:
            conn.close()
    except (sqlite3.Error, ValueError, TypeError, struct.error):
        return []
    return changed
