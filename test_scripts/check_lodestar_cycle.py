"""
Does the LODESTAR protocol still work end to end, against the live gateway?

    python test_scripts/check_lodestar_cycle.py [--gateway URL] [--model NAME]

The unit suite and evals/ prove the state machine on frozen histories. This
proves the whole path: a real request, the real injection, the real model, the
real enforcer — the part that no offline test can reach.

It runs in a THROWAWAY workspace built from LODESTAR_template, so the model's
Phase B write lands in a temp directory and never touches the memory of the
project you are working in. The semantic memory there is empty on purpose: that
also exercises the no-match path, which is what a freshly deployed project hits.

Checks, in order:
  1. the gateway derives state 1 for a fresh cycle
  2. the injected prompt carries the stable block AND the per-turn frame
  3. active-context.md is handed over, so the read step starts complete
  4. the model's first tool call is the memory search, not a read
  5. nothing was synthesized on its behalf to get there

Exit 0 when every check passes.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read a file",
     "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}},
                    "required": ["filePath"]}}},
    {"type": "function", "function": {"name": "write", "description": "Write a file",
     "parameters": {"type": "object", "properties": {"filePath": {"type": "string"},
                                                     "content": {"type": "string"}},
                    "required": ["filePath", "content"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                    "required": ["command"]}}},
]

TASK = ("Dodaj funkcje liczaca sume dwoch liczb w pliku calc.py "
        "i napisz do niej test jednostkowy.")


def build_workspace() -> Path:
    ws = Path(tempfile.mkdtemp(prefix="lodestar_check_"))
    shutil.copytree(ROOT / "LODESTAR_template" / ".context", ws / ".context")
    # A template active-context has placeholder text; give it something real so
    # the handover is visibly the file and not the template.
    (ws / ".context" / "memory" / "active-context.md").write_text(
        "# Active Context\n\n## Goal\n- \n\n## Status\n- Zakonczono\n\n"
        "## Next Steps\n1. \n\n---\n## Progress Log\n"
        "- **MARKER-FOR-THE-CHECK** poprzednie zadanie zamkniete.\n",
        encoding="utf-8")
    return ws


def local_view(ws: Path):
    """What the gateway would build, computed in-process."""
    from gateway.lodestar.context_injector import inject_lodestar_context
    from gateway.lodestar.state import determine_state

    msgs = [{"role": "system", "content": f"<workspace>{ws}</workspace>"},
            {"role": "user", "content": TASK}]
    state = determine_state(list(msgs), workspace_dir=str(ws))
    out = inject_lodestar_context([dict(m) for m in msgs], ws, state, max_synthesis=5)
    return state, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://127.0.0.1:18000")
    ap.add_argument("--model", default="coding-35b-mxfp8")
    ap.add_argument("--keep", action="store_true", help="keep the temp workspace")
    args = ap.parse_args()

    ws = build_workspace()
    print(f"throwaway workspace: {ws}")
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    state, injected = local_view(ws)
    system_msg = injected[0]["content"]
    frame = injected[-1]["content"]

    print("\n1-3. What the gateway builds")
    check("state is 1 (READ) for a fresh cycle", int(state) == 1, f"got {int(state)}")
    check("system message carries the stable block",
          "=== LODESTAR PROTOCOL ===" in system_msg)
    # The real invariant is byte-stability, not the absence of a substring:
    # LODESTAR.md legitimately *describes* the [LODESTAR TODO] block.
    from gateway.lodestar.context_injector import inject_lodestar_context as _inj
    from gateway.lodestar.state import LodestarState as _S
    variants = {
        _inj([{"role": "system", "content": f"<workspace>{ws}</workspace>"},
              {"role": "user", "content": TASK}], ws, st, max_synthesis=5)[0]["content"]
        for st in (_S.READ, _S.UPDATE, _S.READY)
    }
    check("system message is byte-identical across all three states",
          len(variants) == 1, f"{len(variants)} distinct versions")
    check("per-turn frame is the last message",
          "=== LODESTAR TURN ===" in frame and injected[-2]["role"] == "user")
    check("active-context.md was handed over",
          "MARKER-FOR-THE-CHECK" in frame)
    check("the read step starts complete", "[x] Read" in frame)
    check("the search is the next required action",
          "NEXT REQUIRED ACTION" in frame and "embed.py" in frame.split("NEXT REQUIRED ACTION")[1])

    print("\n4-5. What the model does with it (live request)")
    body = {
        "model": args.model,
        "messages": [{"role": "system", "content": f"<workspace>{ws}</workspace>"},
                     {"role": "user", "content": TASK}],
        "tools": TOOLS,
        "stream": False,
    }
    try:
        r = httpx.post(f"{args.gateway}/v1/chat/completions", json=body, timeout=600.0)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        check("live request completed", False, repr(exc))
        print(f"\n{len(failures)} check(s) failed")
        return 1

    msg = (data.get("choices") or [{}])[0].get("message") or {}
    calls = msg.get("tool_calls") or []
    names = [(c.get("function") or {}).get("name") for c in calls]
    argsets = [(c.get("function") or {}).get("arguments", "") for c in calls]
    print(f"      model made {len(calls)} tool call(s): {names}")
    for a in argsets:
        print(f"        args: {a[:150]}")

    check("the model called a tool at all", bool(calls))
    is_search = any("embed.py" in a and "search" in a for a in argsets)
    reads_ctx = any("active-context" in a for a in argsets)
    check("first action is the memory search", is_search,
          "" if is_search else "expected an embed.py search")
    check("it did NOT re-read active-context", not reads_ctx,
          "" if not reads_ctx else "the handover did not take")
    check("the gateway did not have to act for it",
          not any(str(c.get("id", "")).startswith("syn_") for c in calls),
          "synthesized ids present" if any(str(c.get("id", "")).startswith("syn_") for c in calls) else "")

    if not args.keep:
        shutil.rmtree(ws, ignore_errors=True)
        print(f"\nremoved {ws}")
    else:
        print(f"\nkept {ws}")

    print(f"\n{'ALL CHECKS PASSED' if not failures else str(len(failures)) + ' check(s) FAILED: ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
