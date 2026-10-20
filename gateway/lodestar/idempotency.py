"""LODESTAR active-context idempotency check.

Tracks a hash of `active-context.md` so the gateway can detect when a subagent
edits the file out-of-band and warn the main session to re-read before re-editing.
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path


_HASH_LEN = 16
_CACHE_TTL_MS = 100


@dataclass(frozen=True)
class IdempotencyResult:
    current_hash: str
    current_content: str
    is_modified: bool


def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_HASH_LEN]


_cache: dict[str, tuple[float, IdempotencyResult]] = {}
_cache_lock = threading.Lock()


def check_active_context_idempotency(
    file_path: Path, last_known_hash: str | None
) -> IdempotencyResult:
    key = str(file_path.resolve())
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(key)
        if cached and (now - cached[0]) * 1000 < _CACHE_TTL_MS:
            result = cached[1]
        else:
            content = file_path.read_text(encoding="utf-8")
            h = hash_content(content)
            result = IdempotencyResult(
                current_hash=h,
                current_content=content,
                is_modified=(last_known_hash is None or h != last_known_hash),
            )
            _cache[key] = (now, result)
    if last_known_hash is not None and result.current_hash == last_known_hash:
        return IdempotencyResult(
            current_hash=result.current_hash,
            current_content=result.current_content,
            is_modified=False,
        )
    return result
