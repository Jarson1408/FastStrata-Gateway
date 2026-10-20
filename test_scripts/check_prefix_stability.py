"""How much of each prompt does llama-server have to re-process?

llama-server reuses the KV cache by longest common prefix. Turn N+1 can reuse
everything up to the first character that differs from turn N; the rest is
prefilled again. Measured on the live rig 2026-08-31:

    identical prompt        prompt_n =    4   (of 6161)
    prompt + appended text  prompt_n =   13
    cold                    prompt_n = 6161   in 7.64 s  -> 806 tok/s

So the server's cache is not the problem. The question is whether the GATEWAY
keeps its own prompt append-only, and the number that answers it is not the
raw prefix percentage — a volatile block that must sit after the newest turn
caps that percentage by construction, and correctly so.

The number that matters is WASTE:

    grew       = len(cur) - len(prev)      genuinely new content
    recomputed = len(cur) - common_prefix  what the server must prefill
    waste      = recomputed - grew         re-prefilled for nothing

Waste is what a mid-prompt mutation shows up as. A healthy pipeline holds it
at roughly the size of the volatile LODESTAR turn block (~700 chars) and flat
as the conversation grows. Waste that scales with conversation length means
something in the middle is moving, and every turn pays for it.

    python test_scripts/check_prefix_stability.py
        Synthetic agentic cycle through the real pipeline. Offline, ~1 s.
        This is the regression guard.

    python test_scripts/check_prefix_stability.py --dumps [DIR]
        The real thing. Needs prompts captured from a live session:
        set LOCAL_LLM_GATEWAY_AUDIT_FULL_PROMPTS=1, restart the gateway, work
        normally, then point this at the dump directory (default: ./prompts).
        Only this mode can see what the client actually sends.

Exit 0 = every turn was effectively an append. Exit 1 = waste over budget.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway import normalizer, spill  # noqa: E402
from gateway.config import GatewayConfig  # noqa: E402
from gateway.lodestar.context_injector import (  # noqa: E402
    DEFAULT_CONTEXT_MAX_CHARS,
)

TURNS = 8

# The volatile LODESTAR turn block is re-prefilled every turn by design (it
# must sit after the newest user turn), and that is the waste we accept. The
# budget is derived from what bounds that block rather than picked: the
# active-context handover is capped at DEFAULT_CONTEXT_MAX_CHARS and the rest
# of the frame (directive, TODO, notices) measured ~1,300 characters.
#
# It used to be a flat 4,000 against a 220-character active-context.md — and
# the real file in this repository measured 17,782 on 2026-09-12, 80x the
# fixture. So the one check that exists to catch a bloated frame was sized so
# that it never could: a fixture far below the threshold it guards exercises
# none of the path it claims to cover. The fixture is now the shape of a real
# file and make_workspace asserts it crosses the cap.
WASTE_BUDGET_CHARS = DEFAULT_CONTEXT_MAX_CHARS + 2500

TOOLS = [
    {"type": "function", "function": {"name": "read", "parameters": {
        "type": "object", "properties": {"filePath": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "grep", "parameters": {
        "type": "object", "properties": {"pattern": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "bash", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "write", "parameters": {
        "type": "object", "properties": {"filePath": {"type": "string"},
                                         "content": {"type": "string"}}}}},
]

ACTIVE_CONTEXT = """# Active Context

## Goal
- zaktualizuj dokumentacje o nowe foldery

## Status
- W toku

## Active Blockers & Issues
- Brak

## Next Steps
1. przejrzec strukture

---
## Progress Log
- **Poprzednie zadanie** — zrobione.
"""


# --------------------------------------------------------------------------
# Shared measurement
# --------------------------------------------------------------------------

def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def show_divergence(prev: str, cur: str, at: int) -> None:
    lo = max(0, at - 100)
    print(f"        diverges at char {at:,} of {len(prev):,}")
    print(f"        common tail:  ...{prev[lo:at]!r}")
    print(f"        previous had: {prev[at:at + 140]!r}")
    print(f"        now has:      {cur[at:at + 140]!r}")


def report(prompts: list[tuple[str, str]]) -> int:
    """prompts = [(label, text)] in turn order. Returns an exit code."""
    print("  turn            chars        grew   recomputed       waste   verdict")
    print("  " + "-" * 72)

    over: list[tuple[str, str, str, int]] = []
    prev_label, prev = prompts[0]
    print(f"  {prev_label:<12} {len(prev):>10,}  {'—':>10}  {'—':>11}  {'—':>10}   baseline")

    for label, cur in prompts[1:]:
        shared = common_prefix_len(prev, cur)
        grew = len(cur) - len(prev)
        recomputed = len(cur) - shared
        waste = recomputed - grew
        if waste <= WASTE_BUDGET_CHARS:
            verdict = "append"
        else:
            verdict = "WASTE"
            over.append((label, prev, cur, shared))
        print(f"  {label:<12} {len(cur):>10,}  {grew:>10,}  {recomputed:>11,}  "
              f"{waste:>10,}   {verdict}")
        prev_label, prev = label, cur

    print()
    if not over:
        print(f"PASS: waste stayed under {WASTE_BUDGET_CHARS:,} chars on every turn.")
        print("      The prompt is effectively append-only, so prefill costs")
        print("      only the new tokens.")
        return 0

    label, p, c, at = over[0]
    print(f"FAIL: {len(over)} turn(s) re-prefilled more than "
          f"{WASTE_BUDGET_CHARS:,} chars of unchanged conversation.")
    print(f"      First at {label}: {len(c) - at:,} chars recomputed for "
          f"{len(c) - len(p):,} new ones.\n")
    show_divergence(p, c, at)
    print("\n      Everything after that offset is prefilled again on every")
    print("      turn. Find what moves it and make it append instead.")
    return 1


# --------------------------------------------------------------------------
# Mode A — synthetic cycle through the real pipeline
# --------------------------------------------------------------------------

def make_workspace() -> Path:
    ws = Path(tempfile.mkdtemp(prefix="prefix_probe_"))
    ctx = ws / ".context"
    (ctx / "memory").mkdir(parents=True)
    (ctx / "semantic-memory").mkdir(parents=True)
    (ctx / "LODESTAR.md").write_text(
        "# LODESTAR\n\nPhase A: read active context.\n", encoding="utf-8")
    # Shaped like a real one: five long Progress Log entries, which is what
    # MAX_PROGRESS_ENTRIES permits and what this repository's own file holds.
    log = "".join(
        f"- **Wpis numer {i} (12.09.2026)** — "
        + ("szczegoly tego wpisu " * 160)[:3200] + chr(10)
        for i in range(5))
    realistic = ACTIVE_CONTEXT + log
    if len(realistic) <= DEFAULT_CONTEXT_MAX_CHARS:
        raise SystemExit(
            "fixture bug: active-context.md is below the cap it exists to "
            f"exercise ({len(realistic)} <= {DEFAULT_CONTEXT_MAX_CHARS})")
    (ctx / "memory" / "active-context.md").write_text(realistic, encoding="utf-8")
    (ctx / "project_rules.md").write_text(
        "# Rules\n\n- Zawsze uruchamiaj testy.\n", encoding="utf-8")
    return ws


def build_history(ws: Path, turns: int) -> list[dict]:
    """A cycle past Phase A/B, doing ordinary work.

    Results are deliberately large enough to spill, and assistant turns carry
    reasoning: both rewrite history, so both are candidates for moving the
    middle of the prompt.
    """
    wsp = str(ws).replace("\\", "/")
    msgs: list[dict] = [
        {"role": "system",
         "content": f"You are a coding agent.\nWorkspace: {wsp}"},
        {"role": "user", "content": "zaktualizuj dokumentacje o nowe foldery"},
        {"role": "assistant", "content": "",
         "reasoning_content": "Najpierw czytam active-context.",
         "tool_calls": [{"id": "c1", "type": "function", "function": {
             "name": "read", "arguments": json.dumps(
                 {"filePath": f"{wsp}/.context/memory/active-context.md"})}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "read", "content": ACTIVE_CONTEXT},
        {"role": "assistant", "content": "",
         "reasoning_content": "Teraz szukam lekcji w pamieci semantycznej.",
         "tool_calls": [{"id": "c2", "type": "function", "function": {
             "name": "bash", "arguments": json.dumps({
                 "command": f'python "{wsp}/.context/semantic-memory/embed.py" '
                            f'search "docs folders"'})}}]},
        {"role": "tool", "tool_call_id": "c2", "name": "bash",
         "content": "[lessons] sim=0.61 session=abc 3d ago: [id=12] Use write for active-context."},
        {"role": "assistant", "content": "",
         "reasoning_content": "Zapisuje Goal do active-context.",
         "tool_calls": [{"id": "c3", "type": "function", "function": {
             "name": "write", "arguments": json.dumps({
                 "filePath": f"{wsp}/.context/memory/active-context.md",
                 "content": ACTIVE_CONTEXT})}}]},
        {"role": "tool", "tool_call_id": "c3", "name": "write",
         "content": "File written successfully"},
    ]
    for i in range(turns):
        msgs.append({"role": "assistant", "content": "",
                     "reasoning_content": f"Sprawdzam wzorzec {i} w zrodlach.",
                     "tool_calls": [{"id": f"g{i}", "type": "function", "function": {
                         "name": "grep",
                         "arguments": json.dumps({"pattern": f"pattern_{i}"})}}]})
        # ~38 KB, comfortably over the 16 KB spill threshold. The first
        # version of this fixture landed 4 bytes UNDER it and therefore never
        # exercised spill at all, which is the most likely home for a preview
        # that stops being a pure function of its content.
        body = "".join(
            f"src/module_{i}_{j}.py:{j}: def handler_{j}(request, timeout=30):\n"
            for j in range(600)
        )
        assert len(body.encode()) > 16384, "fixture no longer triggers spill"
        msgs.append({"role": "tool", "tool_call_id": f"g{i}", "name": "grep",
                     "content": body})
    return msgs


async def render(messages: list[dict], ws: Path, config: GatewayConfig,
                 store: spill.SpillStore) -> str:
    """The production pipeline, in production order (app.py:291-551)."""
    body = {"model": "coding-35b", "messages": [dict(m) for m in messages],
            "tools": list(TOOLS)}

    body["messages"] = normalizer.normalize_messages(
        body["messages"], config, agentic=True)

    body["messages"], _ = spill.apply_spill_policy(
        body["messages"], config.spill_max_inline_bytes, store, session_id=str(ws))

    # Local, deterministic token counting: pruning must be exercised, but a
    # network round trip per message would make this useless offline.
    upstream = MagicMock()
    upstream.get_prune_limit.return_value = config.get_prune_limit()
    upstream.count_tokens = AsyncMock(
        side_effect=lambda text, base_url=None: max(1, len(text) // 4))
    body["messages"] = await normalizer.prune_messages(
        body["messages"], upstream, max_tokens=config.get_prune_limit(),
        compaction=config.compaction_enabled)

    normalizer.inject_tools_to_prompt(body, TOOLS)
    return normalizer.build_generate_body(body, config, stream=True)["prompt"]


async def run_synthetic() -> int:
    ws = make_workspace()
    (ws / "spill").mkdir()
    try:
        config = GatewayConfig(upstream="http://localhost:8080")
        config.n_ctx = 150016
        store = spill.SpillStore(ws / "spill")

        print(f"synthetic agentic cycle, {TURNS} turns through the real pipeline")
        print(f"spill threshold {config.spill_max_inline_bytes:,} bytes, "
              f"waste budget {WASTE_BUDGET_CHARS:,} chars\n")

        prompts = []
        for turn in range(TURNS):
            prompts.append((f"turn {turn}",
                            await render(build_history(ws, turn + 1), ws, config, store)))
        code = report(prompts)
        if code == 0:
            print("\n      This covers the gateway's own pipeline only. What the")
            print("      CLIENT sends between turns is not modelled here — for")
            print("      that, capture a real session and rerun with --dumps.")
        return code
    finally:
        shutil.rmtree(ws, ignore_errors=True)


# --------------------------------------------------------------------------
# Mode B — real captured prompts
# --------------------------------------------------------------------------

def run_dumps(dump_dir: Path, events: Path) -> int:
    if not dump_dir.is_dir():
        print(f"No dump directory at {dump_dir}.")
        print("Capture one first:")
        print("  set LOCAL_LLM_GATEWAY_AUDIT_FULL_PROMPTS=1, restart the gateway,")
        print("  work a normal session, then rerun this with --dumps.")
        return 1

    # The JSONL is the only record of the order requests were made in, and of
    # which session each belongs to. Filenames are request ids, not sequence.
    order: list[tuple[float, str, str]] = []
    if events.is_file():
        with events.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("event") == "prompt" and d.get("request_id"):
                    order.append((d.get("ts") or 0.0, d["request_id"],
                                  d.get("session_id") or "?"))
    if not order:
        print(f"No prompt events in {events}; falling back to file mtime.")
        order = [(p.stat().st_mtime, p.stem, "?")
                 for p in dump_dir.glob("*.txt")]

    order.sort()
    by_session: dict[str, list[tuple[str, str]]] = {}
    for _, rid, session in order:
        path = dump_dir / f"{rid}.txt"
        if not path.is_file():
            continue
        by_session.setdefault(session, []).append(
            (rid[-8:], path.read_text(encoding="utf-8")))

    if not by_session:
        print(f"Prompt events exist but no matching .txt files in {dump_dir}.")
        return 1

    worst = 0
    for session, prompts in by_session.items():
        if len(prompts) < 2:
            continue
        print(f"\n=== session {session} — {len(prompts)} turns ===\n")
        worst = max(worst, report(prompts))
    if worst == 0:
        print("\nNo session showed waste over budget.")
    return worst


def main() -> int:
    args = sys.argv[1:]
    if "--dumps" in args:
        i = args.index("--dumps")
        rest = args[i + 1:]
        root = Path(__file__).resolve().parent.parent
        dump_dir = Path(rest[0]) if rest else root / "prompts"
        return run_dumps(dump_dir, root / "gateway_events.jsonl")
    return asyncio.run(run_synthetic())


if __name__ == "__main__":
    sys.exit(main())
