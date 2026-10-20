"""Does `tool_choice: {"type": "function"}` really force that function?

    python test_scripts/check_forced_function.py

Needs a live gateway AND a live llama-server upstream. It is the check that
turns the paragraph at the bottom of `gateway/grammar.py` into a measurement:
the GBNF that pins the function name was written against llama.cpp's
documented syntax, not against a running one, because no llama-server was
reachable on 2026-09-12.

Two outcomes are useful and they are not the same:

  FAIL     the server accepted the grammar and the model still called
           something else, or the grammar was rejected outright -- the field
           does not do what `client_params` promises a client it does.
  SKIP     the upstream is the ExLlamaV3 server, which has no grammar field
           and which `client_params.check()` refuses this request against.
           Nothing was measured; say so rather than printing a green line.

The prompt deliberately invites a DIFFERENT tool from the one it forces. A
prompt that asks for the tool it pins would pass with the grammar deleted.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

TOOLS = [
    {"type": "function", "function": {
        "name": "read", "description": "Read a file",
        "parameters": {"type": "object",
                       "properties": {"filePath": {"type": "string"}},
                       "required": ["filePath"]}}},
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
]

# Asks for a shell command; forces `read`. If the grammar is doing nothing the
# model picks bash, which is what makes this check able to go red.
PROMPT = "Run `ls -la` in the current directory and tell me what is there."
FORCED = "read"


def post(url: str, body: dict, token: str | None, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    headers = {"content-type": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gateway", default="http://127.0.0.1:18000")
    ap.add_argument("--model", default="coding-35b-mxfp8")
    ap.add_argument("--token", default=None)
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="this rig serves one request at a time; a client turn "
                         "in flight delays this one")
    args = ap.parse_args(argv)

    try:
        urllib.request.urlopen(f"{args.gateway}/health", timeout=10).read()
    except Exception as e:  # noqa: BLE001 - any failure here means "no gateway"
        print(f"[FAIL] no gateway at {args.gateway}: {e!r}")
        return 1

    # The gateway's own refusal is the SKIP signal, and asking it beats
    # guessing. The first version read the upstream out of /health and sniffed
    # it for a port -- but /health reports the upstream as a nested STATUS
    # object with no url in it, so the guess silently produced "" and the
    # request went to an upstream that cannot constrain. It then reported FAIL
    # for a rig that was never able to pass, which is the outcome this script
    # exists to distinguish. Measured 2026-09-12, on the first run.
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": TOOLS,
        "tool_choice": {"type": "function", "function": {"name": FORCED}},
        "stream": False,
        "max_tokens": 256,
    }

    try:
        out = post(f"{args.gateway}/v1/chat/completions", body, args.token,
                   args.timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:400]
        if e.code == 400 and "tool_choice" in detail and "grammar" in detail:
            print("[SKIP] the gateway refused the request: this upstream has "
                  "no grammar field, so a named tool_choice cannot be honoured "
                  "and is not pretended to be. Nothing was measured -- point "
                  f"the gateway at a llama-server and run this again.\n  {detail}")
            return 2
        print(f"[FAIL] gateway returned HTTP {e.code}: {detail}")
        return 1
    except TimeoutError:
        print(f"[TIMEOUT] no answer in {args.timeout:.0f}s. The rig serves one "
              "request at a time -- check it is idle. Nothing was measured.")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {e!r}")
        return 1

    calls = ((out.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
    if not calls:
        content = ((out.get("choices") or [{}])[0].get("message") or {}).get("content")
        print(f"[FAIL] no tool call at all. content={str(content)[:200]!r}")
        return 1

    names = [(c.get("function") or {}).get("name") for c in calls]
    if names[0] != FORCED:
        print(f"[FAIL] forced {FORCED!r}, model called {names!r}. The grammar "
              "field was accepted but did not constrain, or was ignored.")
        return 1

    try:
        json.loads((calls[0].get("function") or {}).get("arguments") or "")
    except (TypeError, ValueError) as e:
        print(f"[FAIL] {FORCED!r} was forced but its arguments do not parse: {e}")
        return 1

    print(f"[PASS] forced {FORCED!r} against a prompt asking for a shell "
          f"command; model called {names!r} with parseable arguments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
