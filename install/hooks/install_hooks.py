"""
Install the repository's Git hooks.

    python install/hooks/install_hooks.py            install
    python install/hooks/install_hooks.py --check    report status, change nothing
    python install/hooks/install_hooks.py --remove   uninstall

Git hooks live in .git/hooks/, which is not version-controlled, so something
has to be put there. What goes there is a FORWARDER, two lines long, which
execs the tracked hook in this directory.

It used to be a copy, and that cost a gate. On 2026-09-12 a new secrets check
was added to `install/hooks/pre-commit` and the very next commit ran without
it: the installed copy was months old, nothing said so, and the commit reported
four green gates out of five. That is audit finding S-2026-01 in the one place
where a copy looked unavoidable -- and it is not unavoidable, because a
forwarder is what the root .bat files already do for exactly this reason.

`--check` now tells a forwarder apart from a stale copy, because "a hook is
installed" was the question that had a reassuring and useless answer.

Pointing core.hooksPath at this directory would also work; a forwarder is
preferred because it leaves `git commit --no-verify` and any hand-written hook
behaving normally. An existing hook this script did not write is never
overwritten.
"""
from __future__ import annotations

import argparse
import stat
import subprocess
import sys
from pathlib import Path

HOOKS_SRC = Path(__file__).resolve().parent
# What this script writes today.
FORWARDER_MARKER = "install_hooks.py forwarder"
# What it wrote before 2026-09-12: a verbatim copy of the hook body. Still
# recognised, so an existing clone can be re-installed and uninstalled rather
# than being reported as somebody else's hook.
LEGACY_COPY_MARKER = "pre-commit: unit tests..."
HOOK_NAMES = ("pre-commit",)

FORWARDER = """#!/bin/sh
# install_hooks.py forwarder -- do not edit; edit install/hooks/{name} instead.
# A copy here would go stale the moment that file changes, and a stale gate
# reports success for a check it never ran.
root=$(git rev-parse --show-toplevel) || exit 0
exec sh "$root/install/hooks/{name}" "$@"
"""


def repo_root() -> Path:
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=True)
        return Path(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise SystemExit("Not a git repository (or git is not on PATH).")


def hooks_dir() -> Path:
    d = repo_root() / ".git" / "hooks"
    if not d.is_dir():
        raise SystemExit(f"No hooks directory at {d}")
    return d


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _is_ours(path: Path) -> bool:
    if not path.is_file():
        return False
    text = _read(path)
    return FORWARDER_MARKER in text or LEGACY_COPY_MARKER in text


def _is_stale_copy(path: Path) -> bool:
    """Written by an older revision, and therefore frozen at that revision."""
    if not path.is_file():
        return False
    text = _read(path)
    return LEGACY_COPY_MARKER in text and FORWARDER_MARKER not in text


def check() -> int:
    dest_dir = hooks_dir()
    missing = 0
    for name in HOOK_NAMES:
        dest = dest_dir / name
        if _is_stale_copy(dest):
            print(f"  STALE COPY  {name}  (frozen at an older revision -- "
                  f"re-run this script without --check)")
            missing += 1
        elif _is_ours(dest):
            print(f"  installed   {name}")
        elif dest.is_file():
            print(f"  FOREIGN     {name}  (a hook this script did not write)")
            missing += 1
        else:
            print(f"  missing     {name}")
            missing += 1
    return 1 if missing else 0


def install() -> int:
    dest_dir = hooks_dir()
    for name in HOOK_NAMES:
        src, dest = HOOKS_SRC / name, dest_dir / name
        if not src.is_file():
            raise SystemExit(f"Missing hook source: {src}")
        if dest.is_file() and not _is_ours(dest):
            print(f"  SKIPPED     {name}: a different hook is already installed.\n"
                  f"              Merge it by hand: {dest}")
            continue
        dest.write_text(FORWARDER.format(name=name), encoding="utf-8",
                        newline=chr(10))
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"  installed   {name}  ->  {dest}")
    print("\nRuns `pytest -q -m \"not slow\"` and `evals/runner.py` on every commit "
          "(about six seconds).\nBypass once with `git commit --no-verify`.")
    return 0


def remove() -> int:
    dest_dir = hooks_dir()
    for name in HOOK_NAMES:
        dest = dest_dir / name
        if _is_ours(dest):
            dest.unlink()
            print(f"  removed     {name}")
        elif dest.is_file():
            print(f"  left alone  {name}: not written by this script")
        else:
            print(f"  absent      {name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="report status only")
    group.add_argument("--remove", action="store_true", help="uninstall")
    args = ap.parse_args()

    if args.check:
        return check()
    if args.remove:
        return remove()
    return install()


if __name__ == "__main__":
    sys.exit(main())
