"""What identifiers is the semantic memory carrying, and optionally drop them.

    python test_scripts/check_memory_hygiene.py             # report only
    python test_scripts/check_memory_hygiene.py --redact    # rewrite them

A lesson is written while working on something, and what that something was
called leaks into it. The engineering the lesson carries does not depend on the
label, so the label can go without losing anything -- and `memory.db` is
tracked in git (B-2026-01, it is the only copy that exists off this machine),
so what a lesson is named matters as much as what a document is named.

Reporting is the default because this rewrites the record. `--redact` reuses
the SAME substitution table as the public release
(`install/release/policy.py`), so a lesson that is published later reads like
the prose published beside it, and recomputes each changed row's embedding: a
vector derived from text that no longer exists is not a disclosure, but it is
exactly the silent inconsistency this project keeps finding elsewhere.

The credential patterns are the release's FORBIDDEN list. Nothing has ever
matched them here; they cost nothing and a miss would cost a rotation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "install" / "release"))

import policy  # noqa: E402

from gateway import memory_admin  # noqa: E402

DB = _REPO_ROOT / ".context" / "semantic-memory" / "memory.db"
EXPORT = _REPO_ROOT / ".context" / "semantic-memory" / "lessons.json"


def _embedder():
    """embed.py's own embedding function, so the vectors stay comparable.

    Imported from LODESTAR_template, which is the copy this repository's tests
    cover and the one a user receives. It uses the warm service on :18001 when
    one is listening and loads the model in-process otherwise.
    """
    sys.path.insert(0, str(_REPO_ROOT / "LODESTAR_template" / ".context"
                           / "semantic-memory"))
    import embed

    return embed.get_embedding


def report(rows) -> int:
    hits = 0
    for pattern, what in policy.compiled_forbidden():
        for row in rows:
            for m in pattern.finditer(row.text):
                hits += 1
                lo = max(0, m.start() - 40)
                print(f"  id={row.id:<4} {what}: ...{row.text[lo:m.end() + 40]}...")
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--redact", action="store_true",
                    help="rewrite the matches and recompute their embeddings")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args(argv)

    rows = memory_admin.list_memories(args.db)
    if not rows:
        print(f"no memories in {args.db}")
        return 0

    print(f"{len(rows)} lessons in {args.db}\n")
    hits = report(rows)
    if not hits:
        print("  nothing to redact")
        return 0

    if not args.redact:
        print(f"\n{hits} match(es). Re-run with --redact to rewrite them.")
        return 1

    rules = [(p, r) for p, r, _why in policy.REDACTIONS]
    changed = memory_admin.redact_memories(args.db, rules, embed_fn=_embedder())
    print(f"\nrewrote {len(changed)} lesson(s): {changed}")

    after = report(memory_admin.list_memories(args.db))
    if after:
        print(f"\n{after} match(es) SURVIVED the rewrite", file=sys.stderr)
        return 1
    print("clean")

    if args.db == DB:
        n = memory_admin.export_memories(args.db, EXPORT)
        print(f"refreshed {EXPORT.name} ({n} lessons)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
