"""What dialect does a live Gemma 4 actually speak?

Needs a live gateway AND a Gemma model in the active preset. Asking for it is
what triggers the engine hand-over, so the first run pays the swap.

    python test_scripts/check_gemma_dialect.py
    python test_scripts/check_gemma_dialect.py --model gemma-4-e4b

WHY THIS EXISTS

`gateway/parsers/gemma.py` was written before any Gemma 4 had been downloaded.
It said so honestly -- "they are a guess until somebody downloads the weights
and checks" -- and it named THIS script as the thing that would turn the
sentence into a measurement. The script did not exist. A promise of a
measurement reads exactly like a measurement in a file nobody re-reads.

When it was finally run, on 2026-09-13, the guess was wrong: the parser looked
for `<thought> ... </thought>` and the model emits

    <|channel>thought\\n ... reasoning ... <channel|> ... answer

Nothing crashed. The reasoning simply arrived as `content`, markup and all,
instead of as `reasoning_content` -- and the end-to-end vision check PASSED
anyway, because the digits it grepped for were somewhere inside the unparsed
text. **A check that looks for a substring cannot see a turn whose structure is
wrong**, which is why this script asserts on structure and prints the raw shape
either way.

WHAT IT MEASURES

    thinking   reasoning_content is populated and content carries no markup
    tools      a registered tool comes back as a PARSED tool_call

Both go through the gateway rather than the engine, because the parser is the
thing under test and the engine does not run it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

DEFAULT_GATEWAY = "http://127.0.0.1:18000"

# Every marker either parser rewrites. Any of these reaching `content` means
# the turn was not parsed, whatever else the answer looked like.
LEAKED_MARKERS = ("<|channel>", "<channel|>", "<thought>", "</thought>",
                  "<|tool_call>", "<tool_call|>", "<think>", "</think>")

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read a file from disk.",
        "parameters": {
            "type": "object",
            "properties": {"filePath": {"type": "string"}},
            "required": ["filePath"],
        },
    },
}


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _pick_gemma(gateway: str, timeout: float) -> str | None:
    """A Gemma the gateway can actually serve.

    `installed` is checked as well as the name: this preset declares models
    before their weights exist, and asking for one of those measures the
    engine supervisor's refusal rather than the model's dialect.
    """
    try:
        listing = _get(f"{gateway}/v1/models", timeout)
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    for entry in listing.get("data", []):
        name = str(entry.get("id") or "")
        if "gemma" in name.lower() and entry.get("installed") is not False:
            return name
    return None


def _shape(text: str) -> str:
    """The markers present in a string, in the order they appear."""
    found = []
    for i, _ch in enumerate(text):
        for marker in LEAKED_MARKERS:
            if text.startswith(marker, i):
                found.append(marker)
    return " ".join(found) if found else "(none)"


def check_thinking(gateway: str, model: str, timeout: float) -> bool:
    print("\n--- thinking ---")
    body = {
        "model": model,
        "max_tokens": 1024,
        "temperature": 0.0,
        "messages": [{"role": "user",
                      "content": "In one sentence: what is 17 times 3?"}],
    }
    out = _post(f"{gateway}/v1/chat/completions", body, timeout)
    message = (out.get("choices") or [{}])[0].get("message") or {}
    content = (message.get("content") or "").strip()
    reasoning = (message.get("reasoning_content") or "").strip()

    print(f"  content            : {content[:160]!r}")
    print(f"  reasoning_content  : {len(reasoning)} chars")
    print(f"  markers in content : {_shape(content)}")

    ok = True
    leaked = [m for m in LEAKED_MARKERS if m in content]
    if leaked:
        print(f"  [FAIL] unparsed markup reached content: {leaked}")
        print("         The parser is looking for a shape this model does not")
        print("         emit. Read the raw text above and fix _OPEN_REWRITES /")
        print("         _CLOSE_REWRITES in gateway/parsers/gemma.py.")
        ok = False
    if not reasoning:
        print("  [WARN] no reasoning_content at all. Either the model did not")
        print("         think this turn, or its thinking was never recognised.")
    if ok:
        print("  [PASS] the thinking channel was split out of content.")
    return ok


def check_tools(gateway: str, model: str, timeout: float) -> bool:
    print("\n--- tool calls ---")
    # A throwaway workspace with no .context/LODESTAR.md: a live check that
    # registers tools and does NOT declare one inherits the gateway's own
    # repository, lands in LODESTAR state 1, and measures the supervisor's
    # gating while believing it is measuring the model.
    with tempfile.TemporaryDirectory() as workspace:
        target = os.path.join(workspace, "notes.txt")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("nothing here\n")
        body = {
            "model": model,
            "max_tokens": 2048,
            "temperature": 0.0,
            "tools": [READ_TOOL],
            "messages": [
                {"role": "system",
                 "content": f"<workspace>{workspace}</workspace>"},
                {"role": "user",
                 "content": f"Read the file {target.replace(os.sep, '/')} "
                            "using the read tool. Do not answer from memory."},
            ],
        }
        out = _post(f"{gateway}/v1/chat/completions", body, timeout)

    choice = (out.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = (message.get("content") or "").strip()
    calls = message.get("tool_calls") or []

    print(f"  finish_reason      : {choice.get('finish_reason')}")
    print(f"  tool_calls         : {len(calls)}")
    for call in calls:
        fn = call.get("function") or {}
        print(f"      {fn.get('name')}({fn.get('arguments')})")
    print(f"  content            : {content[:160]!r}")
    print(f"  markers in content : {_shape(content)}")

    if not calls:
        print("  [FAIL] no parsed tool call.")
        _show_raw_syntax(gateway, model, timeout)
        return False

    # **Valid JSON is not the bar, and the first version of this check set it
    # there.** A `read({})` parses perfectly and cannot read anything: the
    # gateway logged `CRITICAL FIELD MISSING: tool=read field=filePath` for
    # that exact turn while this script printed PASS. A call is only a call if
    # it carries what the schema says it requires.
    required = (READ_TOOL["function"]["parameters"].get("required") or [])
    for call in calls:
        arguments = (call.get("function") or {}).get("arguments")
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            print(f"  [FAIL] arguments are not JSON: {arguments!r}")
            _show_raw_syntax(gateway, model, timeout)
            return False
        missing = [key for key in required if not parsed.get(key)]
        if missing:
            print(f"  [FAIL] the call is missing {missing}: {parsed!r}")
            print("         A call without its required arguments cannot run.")
            _show_raw_syntax(gateway, model, timeout)
            return False
    print("  [PASS] a registered tool came back parsed, with its arguments.")
    return True


def _show_raw_syntax(gateway: str, model: str, timeout: float) -> None:
    """What the model writes in answer to THE GATEWAY'S OWN prompt.

    **The first version of this asked a prompt of its own**, with the tool
    described in the user turn, because registering no tools is the easy way
    to stop anything parsing the answer. It measured the wrong thing: with no
    tools registered the gateway injects nothing, so the model had no reason
    to use the XML dialect and replied with a JSON object in a fenced block.
    That reading sent me looking for a JSON call shape that the model does not
    produce when it is actually asked properly.

    So this builds the real thing -- `normalize_messages` then
    `inject_tools_to_prompt` then `build_chat_body`, the same three calls the
    request path makes -- and posts it straight to the ENGINE, where no FSM is
    waiting. The bytes that come back are the bytes the parser would have had.

    Requires the gateway package on the path, which a `test_scripts` runner
    already assumes.
    """
    print("\n  --- what the model writes in answer to the gateway's own prompt ---")
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from gateway import config_file, normalizer          # noqa: PLC0415
        from gateway.config import GatewayConfig             # noqa: PLC0415
        from gateway.model_registry import ModelRegistry     # noqa: PLC0415
    except ImportError as exc:
        print(f"  (cannot import the gateway package: {exc})")
        return

    registry = ModelRegistry.from_preset({"models": config_file.active_models()})
    spec = registry.resolve(model)
    if spec is None or not spec.upstream:
        print(f"  (no registry entry for {model!r}; nothing to ask directly)")
        return

    config = GatewayConfig(upstream=spec.upstream)
    config.lodestar_enabled = False
    messages = normalizer.normalize_messages(
        [{"role": "user",
          "content": "Read the file C:/tmp/notes.txt using the read tool. "
                     "Do not answer from memory."}],
        config, agentic=True, vision=spec.vision)
    client_body = {"model": model, "messages": messages, "tools": [READ_TOOL],
                   "max_tokens": 1024, "temperature": 0.0}
    normalizer.inject_tools_to_prompt(client_body, [READ_TOOL])
    body = normalizer.build_chat_body(client_body, config, stream=False)
    body["stream"] = False

    try:
        out = _post(f"{spec.upstream}/v1/chat/completions", body, timeout)
    except (urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print(f"  (could not ask the engine directly: {type(exc).__name__}: {exc})")
        return
    raw = ((out.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    print("  " + "\n  ".join(raw.strip().splitlines()[:40]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gateway", default=DEFAULT_GATEWAY)
    ap.add_argument("--model", default=None,
                    help="default: the first installed Gemma in /v1/models")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per turn; the engine hand-over is inside the first")
    ap.add_argument("--skip-tools", action="store_true")
    args = ap.parse_args()

    gateway = args.gateway.rstrip("/")
    model = args.model or _pick_gemma(gateway, 15.0)
    if not model:
        print("[SKIP] no installed Gemma in /v1/models, and none named.")
        print("       Nothing to measure -- this is an outcome, not a pass.")
        return 2

    print(f"model   : {model}")
    print(f"waiting : up to {args.timeout:.0f}s per turn")

    try:
        ok = check_thinking(gateway, model, args.timeout)
        if not args.skip_tools:
            ok = check_tools(gateway, model, args.timeout) and ok
    except urllib.error.HTTPError as exc:
        print(f"[FAIL] HTTP {exc.code}: "
              f"{exc.read().decode('utf-8', 'replace')[:400]}")
        return 1
    except (TimeoutError, OSError) as exc:
        print(f"[TIMEOUT] {type(exc).__name__}: {exc}")
        print("          Nothing was verified. Check the rig is idle.")
        return 1

    print()
    print("[PASS] the parser matches what this model emits." if ok else
          "[FAIL] the parser and the model disagree; the raw shapes are above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
