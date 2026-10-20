# Semantic Memory Embedder

# CPU-ready embedding generation using all-MiniLM-L6-v2
# Requires: pip install sentence-transformers

import sqlite3
import os
import re
import time
from datetime import datetime

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Similarity below which a stored lesson is not worth showing. The selection
# used to guarantee `min_k` results whatever their score, so a query with no
# related history still returned its two nearest rows -- e.g. "kv cache prefix
# stability" answering with a lesson about subagent throttling at sim=0.424.
# Those land in the model's context on every single task. Silence is the more
# useful answer: it means "no prior experience here", which is true.
MIN_SIMILARITY = 0.45
MIN_SIMILARITY_ENV = "EMBED_MIN_SIMILARITY"

# Env var that callers can set to tag new lessons with the current session id
# (e.g. the OpenCode session UUID, or a manually set value). When unset,
# lessons are tagged "unknown" so stale lessons from previous sessions can
# still be distinguished.
SESSION_ID_ENV = "LODESTAR_SESSION_ID"

# When set to "0" / "false", path verification is disabled. Default: on.
VERIFY_PATHS_ENV = "EMBED_VERIFY_PATHS"

# Pattern for tokens that look like file paths inside lesson text. Structure:
#   (?<![\w/\\.])     not preceded by word/slash/backslash/dot
#   (                  capture group
#     (?:[/\\]|[A-Za-z]:[\\])?   optional absolute-path prefix:
#                                 Unix: leading /  Windows: C:\  (or \\ UNC)
#     [\w]            one word char (start of basename)
#     [\w/\\-]*       0+ word/slash/backslash/dash chars (rest of basename)
#     \.              exactly one literal dot (the separator)
#     \w{1,10}        1-10 word chars (the extension)
#   )
#   (?![\w/\\.])      not followed by word/slash/backslash/dot
# Dots are NOT allowed in the basename part to avoid swallowing "foo.py.bak"
# as a single token — the regex matches the rightmost extension.
_PATH_TOKEN_RE = re.compile(
    r"(?<![\w/\\.])((?:[/\\]|[A-Za-z]:[\\])?[\w][\w/\\-]*\.\w{1,10})(?![\w/\\.])"
)

# Version-like patterns that should never be treated as paths, even if they
# have a single-letter prefix like "v" in semver tags.
_VERSION_RE = re.compile(r"^v?\d+(\.\d+)*$")

_model = None


def get_model():
    """Load the embedding model on first use.

    sentence-transformers pulls in torch/transformers and costs ~7 s to
    import. Doing that at module import time made every `pytest` collection
    (and every `embed.py --help`) pay for a model that most code paths —
    export, list, path verification — never touch.

    Measured 2026-08-31, per invocation: 7.2 s import + 3.4 s load online, of
    which 3.0 s is the Hub checking a model that is already on disk. Loading
    offline first cuts that to 0.4 s, and the first run on a new machine falls
    back to an online load so the model can actually be fetched. CPU only, no
    GPU assumed anywhere — the same on every machine.
    """
    global _model
    if _model is None:
        prior = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer
            try:
                _model = SentenceTransformer(MODEL_NAME, device="cpu")
            except Exception:
                # Not in the local cache yet: this machine has never run it.
                for k, v in prior.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
                _model = SentenceTransformer(MODEL_NAME, device="cpu")
        finally:
            for k, v in prior.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return _model


# A warm process to borrow the model from. The remaining cost of a cold call is
# `import sentence_transformers` (7.2 s measured), which no amount of lazy
# loading removes -- only not paying it again does. `embed.py serve` holds the
# model in one process; everything else asks it over loopback and falls back to
# loading locally, so a machine with nothing running still works exactly as
# before. CPU only, stdlib only, no GPU assumed.
EMBED_SERVICE_ENV = "EMBED_SERVICE_URL"
DEFAULT_SERVICE_URL = "http://127.0.0.1:18001"
SERVICE_TIMEOUT_S = 2.0


def service_url() -> str:
    return os.environ.get(EMBED_SERVICE_ENV, DEFAULT_SERVICE_URL).rstrip("/")


def _embed_via_service(text: str) -> list[float] | None:
    """Ask the warm process. None means "not available" — never an exception."""
    import json as _json
    import urllib.error
    import urllib.request

    payload = _json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        service_url() + "/embed", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=SERVICE_TIMEOUT_S) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        vec = data.get("embedding")
        if isinstance(vec, list) and len(vec) == EMBEDDING_DIM:
            return [float(x) for x in vec]
    except (urllib.error.URLError, OSError, ValueError, TypeError):
        return None
    return None


def get_embedding(text: str) -> list[float]:
    """Embed one string, preferring a warm process over a cold model load.

    The vectors must be identical either way — the service runs the same model
    through the same call — because they are compared against embeddings
    already stored in memory.db.
    """
    vec = _embed_via_service(text)
    if vec is not None:
        return vec
    return get_model().encode(text).tolist()


def serve(port: int) -> None:
    """Run the embedding service in the foreground until interrupted."""
    import json as _json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    model = get_model()  # pay the import once, here

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._reply(200, {"status": "ok", "model": MODEL_NAME,
                                  "dim": EMBEDDING_DIM})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/embed":
                self._reply(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = _json.loads(self.rfile.read(length).decode("utf-8"))
                text = body.get("text")
                if not isinstance(text, str):
                    raise ValueError("text must be a string")
            except (ValueError, TypeError) as exc:
                self._reply(400, {"error": str(exc)})
                return
            self._reply(200, {"embedding": model.encode(text).tolist()})

        def _reply(self, code: int, obj: dict) -> None:
            payload = _json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):  # quiet: one line per embed is noise
            pass

    httpd = HTTPServer(("127.0.0.1", port), Handler)
    print(f"embedding service on http://127.0.0.1:{port} "
          f"(model={MODEL_NAME}, CPU). Ctrl+C to stop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("stopping.")
    finally:
        httpd.server_close()


# ---------------------------------------------------------------------------
# Schema migrations
# ---------------------------------------------------------------------------
#
# The schema used to be created by CREATE TABLE IF NOT EXISTS plus one ad-hoc
# PRAGMA-then-ALTER for the session_id column. That works exactly once: a
# second change would need a second bespoke block, with nothing recording which
# changes a given file had already seen. A version number and a list of
# migrations is the same amount of code and scales.
#
# Each migration takes an open connection and moves the schema forward by one
# step. The index into MIGRATIONS is the version it produces.

SCHEMA_VERSION = 3


def _migration_1_create_memories(conn: sqlite3.Connection) -> None:
    """The original table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def _migration_2_add_session_id(conn: sqlite3.Connection) -> None:
    """Tag each lesson with the session that saved it.

    Without it, stale lessons from earlier sessions are indistinguishable from
    current ones.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
    if "session_id" not in cols:
        conn.execute(
            "ALTER TABLE memories ADD COLUMN session_id TEXT NOT NULL DEFAULT 'unknown'"
        )


def _migration_3_add_indexes(conn: sqlite3.Connection) -> None:
    """Index what the queries actually filter and sort on.

    search_memories() filters by category; the dashboard lists newest-first by
    created_at. Both were full scans.
    """
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_category_created "
        "ON memories(category, created_at DESC)"
    )


MIGRATIONS = [
    _migration_1_create_memories,
    _migration_2_add_session_id,
    _migration_3_add_indexes,
]


def _current_version(conn: sqlite3.Connection) -> int:
    """Schema version of an open database.

    A file predating schema_version but already carrying the memories table is
    reported as version 2 — it went through both of the first two steps under
    the old ad-hoc code, and re-running them would be harmless but pointless.
    """
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is not None:
        return int(row[0])

    has_memories = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
    ).fetchone() is not None
    if not has_memories:
        return 0
    # Distinguish 1 from 2 by the column migration 2 adds. Reporting every
    # pre-schema_version file as 2 would skip that migration, and a database
    # created before session_id existed would never get the column — the old
    # ad-hoc code ran the ALTER unconditionally, so this has to too.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
    return 2 if "session_id" in cols else 1


def _set_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute("DELETE FROM schema_version")
    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))


def migrate(conn: sqlite3.Connection) -> int:
    """Bring an open database up to SCHEMA_VERSION. Returns the new version."""
    version = _current_version(conn)
    if version > SCHEMA_VERSION:
        # A newer gateway wrote this file. Leave it alone rather than guess.
        print(f"Warning: memory.db schema is version {version}, this build knows "
              f"{SCHEMA_VERSION}. Not migrating.")
        return version

    for target in range(version + 1, SCHEMA_VERSION + 1):
        MIGRATIONS[target - 1](conn)
        _set_version(conn, target)
        conn.commit()
    if version < SCHEMA_VERSION:
        _set_version(conn, SCHEMA_VERSION)
        conn.commit()
    return SCHEMA_VERSION


def init_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        migrate(conn)
    finally:
        conn.close()


def find_duplicate(db_path: str, category: str, text: str) -> int | None:
    """Return the id of an existing memory with identical category+text, or None.

    Guards the store against duplicate saves: models occasionally repeat the
    Phase D finalization (observed 2026-07-14: the same lesson saved twice in
    one session), and an exact-duplicate row only dilutes search results.
    """
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT id FROM memories WHERE category = ? AND text = ?",
            (category, text),
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def save_memory(db_path: str, category: str, text: str, session_id: str | None = None) -> int:
    init_db(db_path)
    embedding = get_embedding(text)
    import struct
    embedding_array = struct.pack(f'{len(embedding)}f', *embedding)
    embedding_blob = sqlite3.Binary(embedding_array)
    if session_id is None:
        session_id = os.environ.get(SESSION_ID_ENV, "unknown")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO memories (category, text, embedding, session_id) VALUES (?, ?, ?, ?)",
        (category, text, embedding_blob, session_id),
    )
    mem_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return mem_id


def get_keyword_score(text: str, query: str) -> float:
    import re
    # Get tokens split by whitespace/punctuation, keeping dots, underscores, dashes
    raw_tokens = re.findall(r'[\w\.-]+', query.lower())
    query_tokens = [t for t in raw_tokens if len(t) > 1]
    if not query_tokens:
        return 0.0
    text_lower = text.lower()
    matches = 0
    for token in query_tokens:
        if token in text_lower:
            matches += 1
        elif '.' in token:
            # Try matching base name of dot-separated tokens (e.g. 'todo.py' matches 'todo')
            base = token.split('.')[0]
            if len(base) > 2 and base in text_lower:
                matches += 1
    return matches / len(query_tokens)


def _age_seconds(created_at: str) -> float | None:
    """Best-effort age of a memory. Returns None if created_at is unparseable.

    SQLite stores TIMESTAMP as 'YYYY-MM-DD HH:MM:SS' in UTC.
    """
    if not created_at:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(created_at, fmt)
            return max(0.0, time.time() - dt.timestamp())
        except ValueError:
            continue
    return None


def _format_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def _get_recency_score(created_at: str, half_life_days: float = 30.0) -> float:
    """Calculate a normalized recency score in [0.0, 1.0] using exponential decay.

    Default half-life is 30 days. Items from today have score ~1.0,
    30 days old ~0.5, 60 days old ~0.25, etc.
    """
    age_s = _age_seconds(created_at)
    if age_s is None:
        return 0.5  # Neutral default if timestamp is unparseable or missing
    import math
    half_life_s = max(1.0, half_life_days * 86400.0)
    return math.pow(2.0, -max(0.0, age_s) / half_life_s)


def search_memories(
    db_path: str,
    query: str,
    top_k: int = 5,
    min_k: int = 2,
    threshold: float = 0.0,
    expansion_threshold: float = 0.45,
    min_similarity: float | None = None,
) -> list[dict]:
    import math
    init_db(db_path)
    query_embedding = get_embedding(query)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, category, text, embedding, session_id, created_at FROM memories"
    )
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        mem_id, category, text, embedding_blob, session_id, created_at = row
        stored = list(memoryview(embedding_blob))
        # Reconstruct float32 array from bytes
        import struct
        stored = list(struct.unpack(f'{len(stored)//4}f', bytes(stored)))
        # Cosine similarity (semantic)
        dot = sum(a * b for a, b in zip(query_embedding, stored))
        norm_q = math.sqrt(sum(a * a for a in query_embedding))
        norm_m = math.sqrt(sum(b * b for b in stored))
        if norm_q == 0 or norm_m == 0:
            sem_sim = 0.0
        else:
            sem_sim = max(0.0, dot / (norm_q * norm_m))

        # Keyword matching score [0.0, 1.0]
        key_score = get_keyword_score(text, query)

        # Recency score [0.0, 1.0] (30-day half life decay)
        rec_score = _get_recency_score(created_at or "")

        # Hybrid score: 65% semantic, 25% exact keywords, 10% recency
        hybrid_score = 0.65 * sem_sim + 0.25 * key_score + 0.10 * rec_score

        results.append({
            "id": mem_id,
            "category": category,
            "text": text,
            "session_id": session_id or "unknown",
            "created_at": created_at or "",
            "similarity": hybrid_score,
            "semantic_similarity": sem_sim,
            "keyword_score": key_score,
            "recency_score": rec_score,
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)

    # If an explicit threshold is passed (e.g. -t in CLI or test), strictly filter
    if threshold > 0.0:
        return [r for r in results if r["similarity"] >= threshold][:top_k]

    # The floor applies to the guaranteed min_k too: "the two closest rows" is
    # not a useful answer when the closest row is unrelated.
    floor = MIN_SIMILARITY if min_similarity is None else min_similarity
    results = [r for r in results if r["similarity"] >= floor]

    # Default selection: guaranteed top min_k (capped by top_k, if available)
    # + dynamic expansion for additional items whose similarity is >= expansion_threshold, up to top_k.
    effective_min_k = min(min_k, top_k)
    selected = []
    for idx, r in enumerate(results):
        if idx < effective_min_k:
            selected.append(r)
        elif len(selected) < top_k and r["similarity"] >= expansion_threshold:
            selected.append(r)
        else:
            break
    return selected


def export_memories(db_path: str, export_path: str) -> int:
    import json
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT category, text, session_id, created_at FROM memories")
    rows = cursor.fetchall()
    conn.close()

    memories = []
    for row in rows:
        cat, txt, sid, created = row
        memories.append({
            "category": cat,
            "text": txt,
            "session_id": sid,
            "created_at": created
        })

    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)
    return len(memories)


def import_memories(db_path: str, import_path: str) -> tuple[int, int]:
    import json
    import struct
    with open(import_path, "r", encoding="utf-8") as f:
        memories = json.load(f)

    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    imported_count = 0
    skipped_count = 0

    for mem in memories:
        cat = mem.get("category", "lessons")
        txt = mem.get("text", "")
        sid = mem.get("session_id", "unknown")
        created = mem.get("created_at")

        if not txt:
            continue

        # Check for duplicates
        cursor.execute("SELECT id FROM memories WHERE category = ? AND text = ?", (cat, txt))
        if cursor.fetchone():
            skipped_count += 1
            continue

        # Generate embedding
        embedding = get_embedding(txt)
        embedding_array = struct.pack(f'{len(embedding)}f', *embedding)
        embedding_blob = sqlite3.Binary(embedding_array)

        if created:
            cursor.execute(
                "INSERT INTO memories (category, text, embedding, session_id, created_at) VALUES (?, ?, ?, ?, ?)",
                (cat, txt, embedding_blob, sid, created),
            )
        else:
            cursor.execute(
                "INSERT INTO memories (category, text, embedding, session_id) VALUES (?, ?, ?, ?)",
                (cat, txt, embedding_blob, sid),
            )
        imported_count += 1

    conn.commit()
    conn.close()
    return imported_count, skipped_count


def _looks_like_path(token: str) -> bool:
    """Filter false positives: version numbers, pure digit runs, URLs."""
    base, _, ext = token.rpartition(".")
    if not base or not ext:
        return False
    # Version-number forms: "1.2.3", "v1.2.3", "3.14"
    if _VERSION_RE.match(base):
        return False
    # Require at least one alphabetic character in the base so digit-only
    # forms like "3.14" don't sneak through.
    if not any(c.isalpha() for c in base):
        return False
    return True


def extract_paths(text: str) -> list[str]:
    """Return unique path-looking tokens found in `text`, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _PATH_TOKEN_RE.finditer(text):
        token = m.group(1)
        if not _looks_like_path(token):
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


# Directories never worth walking when looking for a bare filename.
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache",
              ".pytest_cache", "dist", "build", ".claude"}

# Upper bound on directories visited per lookup. The point is to answer a
# yes/no question quickly, not to index the repository.
_SEARCH_BUDGET = 2000


def _find_by_basename(name: str, root: str) -> str | None:
    """Locate a bare filename somewhere under `root`, or None.

    Lessons name files the way people do — "in app.py", not
    "in gateway/app.py". Joining that to the working directory and declaring
    NOT FOUND was worse than not checking at all: it stamped a false denial on
    a lesson that was accurate, which is exactly the opposite of what the
    verification is for (2026-08-31: "app.py NOT FOUND" while gateway/app.py
    existed).
    """
    if os.sep in name or "/" in name:
        return None  # already has a directory component; nothing to guess
    visited = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        visited += 1
        if visited > _SEARCH_BUDGET:
            return None
        if name in filenames:
            return os.path.relpath(os.path.join(dirpath, name), root)
    return None


def verify_paths(paths: list[str], cwd: str | None = None) -> list[str]:
    """For each path, return a marker string `verified: <path> EXISTS` or `NOT FOUND`.

    Relative paths are checked against `cwd` (defaults to process CWD).
    Absolute paths are checked directly. A bare filename that is not at `cwd`
    is searched for under it before being reported missing, and the marker then
    names where it actually is. Order matches `paths`.
    """
    if cwd is None:
        cwd = os.getcwd()
    markers: list[str] = []
    for p in paths:
        candidate = p if os.path.isabs(p) else os.path.join(cwd, p)
        if os.path.exists(candidate):
            markers.append(f"verified: {p} EXISTS")
            continue
        found = None if os.path.isabs(p) else _find_by_basename(p, cwd)
        if found:
            markers.append(f"verified: {p} EXISTS at {found}")
        else:
            markers.append(f"verified: {p} NOT FOUND")
    return markers


def _verification_enabled() -> bool:
    flag = os.environ.get(VERIFY_PATHS_ENV, "1").strip().lower()
    return flag not in ("0", "false", "no", "off")


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except AttributeError:
            pass
    db = os.path.join(os.path.dirname(__file__), "memory.db")
    if len(sys.argv) > 1:
        action = sys.argv[1]
        if action == "serve":
            port = 18001
            for i, a in enumerate(sys.argv):
                if a in ("-p", "--port") and i + 1 < len(sys.argv):
                    try:
                        port = int(sys.argv[i + 1])
                    except ValueError:
                        pass
            serve(port)
        elif action == "save" and len(sys.argv) > 2:
            cat = sys.argv[2]
            txt = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
            sid = os.environ.get(SESSION_ID_ENV)
            existing = find_duplicate(db, cat, txt)
            if existing is not None:
                print(f"Duplicate lesson - already saved as id={existing}, skipping. "
                      "Do NOT save it again.")
            else:
                mem_id = save_memory(db, cat, txt, session_id=sid)
                print(f"Saved memory id={mem_id} category={cat}")
        elif action == "search" and len(sys.argv) > 2:
            limit = 5
            threshold = 0.0
            try:
                min_sim = float(os.environ.get(MIN_SIMILARITY_ENV, MIN_SIMILARITY))
            except ValueError:
                min_sim = MIN_SIMILARITY
            query_args = []
            
            i = 2
            while i < len(sys.argv):
                arg = sys.argv[i]
                if arg in ("-n", "--limit") and i + 1 < len(sys.argv):
                    try:
                        limit = int(sys.argv[i+1])
                        i += 2
                    except ValueError:
                        query_args.append(arg)
                        i += 1
                elif arg in ("-s", "--min-sim") and i + 1 < len(sys.argv):
                    try:
                        min_sim = float(sys.argv[i+1])
                        i += 2
                    except ValueError:
                        query_args.append(arg)
                        i += 1
                elif arg in ("-t", "--threshold") and i + 1 < len(sys.argv):
                    try:
                        threshold = float(sys.argv[i+1])
                        i += 2
                    except ValueError:
                        query_args.append(arg)
                        i += 1
                else:
                    query_args.append(arg)
                    i += 1
                    
            query = " ".join(query_args)
            results = search_memories(db, query, top_k=limit, threshold=threshold,
                                      min_similarity=min_sim)
            verify = _verification_enabled()
            if not results:
                # The gateway recognises that the search RAN by this line shape
                # ("[category] sim=N"). Printing nothing on an empty result --
                # which is what a fresh project with an empty memory.db does --
                # leaves LODESTAR waiting for evidence that never comes, and it
                # synthesizes the same search again. Say "nothing matched" in a
                # form the state machine can see.
                print(f"[lessons] sim=0.000 no-match: nothing stored is within "
                      f"{min_sim:.2f} similarity of this query. Treat it as no "
                      f"prior experience, not as an error.")
            for r in results:
                age_s = _age_seconds(r["created_at"])
                age = _format_age(age_s) if age_s is not None else "age=?"
                sid = r["session_id"]
                # Must start with [category] sim=X to be matched by the gateway's lesson annotator
                line = (
                    f"[{r['category']}] sim={r['similarity']:.3f} "
                    f"session={sid} {age}: [id={r['id']}] {r['text']}"
                )
                if verify:
                    paths = extract_paths(r["text"])
                    if paths:
                        markers = verify_paths(paths)
                        line += "  [" + "; ".join(markers) + "]"
                print(line)
        elif action == "list":
            conn = sqlite3.connect(db)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, category, text, session_id, created_at FROM memories ORDER BY created_at DESC"
            )
            rows = cursor.fetchall()
            conn.close()
            if not rows:
                print("No memories stored in the database.")
            for row in rows:
                mem_id, category, text, session_id, created_at = row
                age_s = _age_seconds(created_at)
                age = _format_age(age_s) if age_s is not None else "age=?"
                print(f"id={mem_id} [{category}] session={session_id} {age}: {text}")
        elif action == "delete" and len(sys.argv) > 2:
            try:
                mem_id = int(sys.argv[2])
                conn = sqlite3.connect(db)
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM memories WHERE id = ?", (mem_id,))
                if not cursor.fetchone():
                    print(f"Memory with id={mem_id} not found.")
                else:
                    cursor.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
                    conn.commit()
                    print(f"Deleted memory id={mem_id}")
                conn.close()
            except ValueError:
                print("Error: memory ID must be an integer.")
        elif action == "export" and len(sys.argv) > 2:
            try:
                count = export_memories(db, sys.argv[2])
                print(f"Exported {count} memories to {sys.argv[2]}")
            except Exception as e:
                print(f"Error exporting memories: {e}")
        elif action == "import" and len(sys.argv) > 2:
            try:
                imported, skipped = import_memories(db, sys.argv[2])
                print(f"Import complete: imported {imported}, skipped {skipped} duplicates.")
            except Exception as e:
                print(f"Error importing memories: {e}")
        else:
            print("Usage: python embed.py save|search|list|delete|export|import ...")
            print("  save:   python embed.py save lessons <text>")
            print("  search: python embed.py search <query> [-n limit] [-t threshold]")
            print("  list:   python embed.py list")
            print("  delete: python embed.py delete <id>")
            print("  export: python embed.py export <output_file.json>")
            print("  import: python embed.py import <input_file.json>")
    else:
        print("Usage: python embed.py save|search|list|delete|export|import ...")
        print("  save:   python embed.py save lessons <text>")
        print("  search: python embed.py search <query> [-n limit] [-t threshold]")
        print("  list:   python embed.py list")
        print("  delete: python embed.py delete <id>")
        print("  export: python embed.py export <output_file.json>")
        print("  import: python embed.py import <input_file.json>")
