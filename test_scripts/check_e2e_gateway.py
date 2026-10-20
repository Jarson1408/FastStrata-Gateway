"""
E2E Gateway Tests — run against a live gateway instance.

Usage:
    python test_scripts/check_e2e_gateway.py [--gateway URL] [--timeout S]

Default gateway: http://localhost:18000

These tests verify the full pipeline: request → normalize → prune → inject →
stream/non-stream → FSM parse → SSE emit → response.

Requires: a running gateway AND a running model. Four of the seven checks
generate tokens; the header used to claim otherwise ("Does NOT require:
llama-server"), which is how a run against a sleeping rig reads as a pipeline
defect.

THE RIG SERVES ONE REQUEST AT A TIME. If you are working in the client while
this runs, every live check queues behind that turn. Measured 2026-09-10: a
check with a 30 s limit waited 4.5 minutes behind an OpenCode turn and was
reported as FAIL, indistinguishable from a broken gateway. Hence the generous
default below, and hence TIMEOUT being its own verdict rather than a failure.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import tempfile
import time

import httpx

GATEWAY = "http://localhost:18000"

# Every check that generates tokens shares this budget. 600 s is not padding:
# on this rig the empty-messages check alone took 89.8 s with the slot FREE,
# and a queued request waited 4.5 minutes behind one client turn. The old
# values (120 s for three checks, 30 s for the fourth) were inconsistent with
# each other and both below what the machine actually does.
LIVE_TIMEOUT = 600.0

# The offline checks talk only to the gateway process and must stay fast: if
# /health needs more than this, that IS the finding.
OFFLINE_TIMEOUT = 5.0

PASS, FAIL, TIMEOUT, WARN = "PASS", "FAIL", "TIMEOUT", "WARN"


def _verdict(exc: Exception) -> str:
    """A timeout proves nothing; a failure proves something. Say which."""
    return TIMEOUT if isinstance(exc, httpx.TimeoutException) else FAIL


async def check_health(client: httpx.AsyncClient) -> str:
    """Test 1: Health endpoint returns OK with upstream info."""
    print("Test 1: /health endpoint")
    try:
        r = await client.get(f"{GATEWAY}/health", timeout=OFFLINE_TIMEOUT)
        data = r.json()
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        assert data.get("status") == "ok", f"Expected status=ok, got {data}"
        assert "version" in data, "Missing version field"
        assert "config" in data, "Missing config field"
        print(f"  PASS — version={data['version']}, config={data['config']}")
        return PASS
    except httpx.ConnectError:
        print(f"  SKIP — Gateway not reachable at {GATEWAY}")
        return FAIL
    except Exception as e:
        print(f"  {_verdict(e)} — {e!r}")
        return _verdict(e)


async def check_models(client: httpx.AsyncClient) -> str:
    """Test 2: /v1/models returns model list."""
    print("\nTest 2: /v1/models endpoint")
    try:
        r = await client.get(f"{GATEWAY}/v1/models", timeout=OFFLINE_TIMEOUT)
        data = r.json()
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        assert "data" in data, f"Expected 'data' key, got {list(data.keys())}"
        print(f"  PASS — {len(data['data'])} models listed")
        for m in data["data"][:5]:
            print(f"    - {m.get('id', 'unknown')}")
        return PASS
    except Exception as e:
        print(f"  {_verdict(e)} — {e!r}")
        return _verdict(e)


async def check_non_stream_completion(client: httpx.AsyncClient) -> str:
    """Test 3: Non-streaming chat completion."""
    print("\nTest 3: Non-streaming chat completion")
    try:
        body = {
            "model": "coding-35b-a3b",
            "stream": False,
            "max_tokens": 100,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Be very brief."},
                {"role": "user", "content": "Say 'hello world' and nothing else."},
            ],
        }
        r = await client.post(
            f"{GATEWAY}/v1/chat/completions",
            json=body,
            timeout=LIVE_TIMEOUT,
        )
        data = r.json()
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {data}"
        choices = data.get("choices", [])
        assert len(choices) > 0, "No choices in response"
        msg = choices[0].get("message", {})
        content = msg.get("content") or ""
        finish = choices[0].get("finish_reason")
        usage = data.get("usage", {})
        print(f"  PASS — finish={finish}, content={content[:100]!r}")
        print(f"         usage: prompt={usage.get('prompt_tokens')}, completion={usage.get('completion_tokens')}")
        if msg.get("reasoning_content"):
            print(f"         reasoning: {msg['reasoning_content'][:80]}...")
        return PASS
    except Exception as e:
        print(f"  {_verdict(e)} — {e!r}")
        return _verdict(e)


async def check_stream_completion(client: httpx.AsyncClient) -> str:
    """Test 4: Streaming chat completion — verify SSE format."""
    print("\nTest 4: Streaming chat completion")
    try:
        body = {
            "model": "coding-35b-a3b",
            "stream": True,
            "max_tokens": 100,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Be very brief."},
                {"role": "user", "content": "Count from 1 to 5."},
            ],
        }
        chunks = []
        finish_reason = None
        t_start = time.time()

        async with client.stream(
            "POST",
            f"{GATEWAY}/v1/chat/completions",
            json=body,
            timeout=LIVE_TIMEOUT,
        ) as resp:
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    chunks.append(chunk)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        fr = choices[0].get("finish_reason")
                        if fr:
                            finish_reason = fr
                except json.JSONDecodeError:
                    pass

        t_elapsed = time.time() - t_start
        content_parts = []
        for c in chunks:
            for ch in c.get("choices", []):
                delta = ch.get("delta", {})
                if "content" in delta and delta["content"]:
                    content_parts.append(delta["content"])

        full_content = "".join(content_parts)
        print(f"  PASS — {len(chunks)} chunks in {t_elapsed:.1f}s, finish={finish_reason}")
        print(f"         content: {full_content[:120]!r}")
        return PASS
    except Exception as e:
        print(f"  {_verdict(e)} — {e!r}")
        return _verdict(e)


async def check_tool_call_stream(client: httpx.AsyncClient) -> str:
    """Test 5: Streaming with tools — a tool call MUST reach the client.

    This check could not go red until 2026-09-10. It reported WARN on a miss
    and returned success anyway, explaining that the model "may not have used
    tool — this is model-dependent, not a gateway bug". That was exactly
    backwards, and measurable: the model emitted a tool call on 3 runs out of
    3, every one of them recorded in gateway.log as `tool_calls=1`, and the
    LODESTAR ENFORCER dropped each one —

        LODESTAR ENFORCER: force order (silent drop) — 'bash' blocked,
        next required: Read .context/memory/active-context.md

    — because the request carried no workspace, so it inherited the gateway's
    OWN repository, which is a LODESTAR project. The check was measuring the
    supervisor's gating and calling it the model's mood.

    Hence the throwaway workspace below: a directory with no
    `.context/LODESTAR.md` disables gating, which is what lets this check see
    the pipeline it claims to test. `check_lodestar_cycle.py` uses the same
    `<workspace>` convention for the opposite purpose.

    **The prompt is unchanged on purpose.** With the workspace declared it
    produced a tool call on 3 runs out of 3, so rewriting it would have been a
    fix aimed at the wrong thing — and would have hidden that the gating was
    the cause.

    The arguments are now parsed, not merely counted. A tool call that arrives
    as unparseable JSON is the failure this whole gateway exists to prevent,
    and counting deltas cannot see it.
    """
    print("\nTest 5: Streaming with tools (tool call expected)")

    # No `.context/LODESTAR.md` here, so the supervisor stays out of the way.
    workspace = pathlib.Path(tempfile.mkdtemp(prefix="gw_smoke_")).as_posix()

    body = {
        "model": "coding-35b-a3b",
        "stream": True,
        "max_tokens": 500,
        "messages": [
            {"role": "system", "content": f"<workspace>{workspace}</workspace>"},
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "List the files in /tmp using the bash tool."},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The command to run"},
                        "description": {"type": "string", "description": "What the command does"},
                    },
                    "required": ["command", "description"],
                },
            },
        }],
    }

    # One retry, and only on a miss. Sampling is stochastic, so a single red run
    # is not evidence; a check that reddens at random is a check people learn to
    # ignore, which is how this one ended up returning success unconditionally.
    # Two misses in a row is a finding.
    last_error: Exception | None = None
    for attempt in (1, 2):
        try:
            name, args, chunk_count, finish = await _collect_tool_call(client, body)
        except Exception as e:  # noqa: BLE001 — classified by _verdict below
            last_error = e
            break

        if name is None:
            print(f"  attempt {attempt}: {chunk_count} chunks, finish={finish}, no tool call")
            continue

        if name != "bash":
            print(f"  FAIL — tool call named {name!r}, expected 'bash'")
            return FAIL
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError as e:
            print(f"  FAIL — tool arguments are not valid JSON ({e}): {args[:120]!r}")
            return FAIL
        if "command" not in parsed:
            print(f"  FAIL — tool arguments carry no 'command': {sorted(parsed)}")
            return FAIL

        print(f"  PASS — {chunk_count} chunks, finish={finish}")
        print(f"         bash(command={parsed['command'][:60]!r})")
        return PASS

    if last_error is not None:
        print(f"  {_verdict(last_error)} — {last_error!r}")
        return _verdict(last_error)

    print("  FAIL — no tool call reached the client in 2 attempts.")
    print("         The model is not the first suspect: check gateway.log for")
    print("         'LODESTAR ENFORCER' or 'DEFERRED EMIT' on these chat ids.")
    return FAIL


async def _collect_tool_call(client: httpx.AsyncClient, body: dict):
    """Stream one request; return (tool name, argument string, chunks, finish)."""
    name: str | None = None
    args = ""
    finish = None
    chunk_count = 0

    async with client.stream(
        "POST",
        f"{GATEWAY}/v1/chat/completions",
        json=body,
        timeout=LIVE_TIMEOUT,
    ) as resp:
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            chunk_count += 1
            for choice in chunk.get("choices", []):
                finish = choice.get("finish_reason") or finish
                for tc in choice.get("delta", {}).get("tool_calls", []) or []:
                    fn = tc.get("function", {})
                    name = fn.get("name") or name
                    args += fn.get("arguments") or ""

    return name, args, chunk_count, finish


async def check_bad_json(client: httpx.AsyncClient) -> str:
    """Test 6: Bad JSON body → 400 error."""
    print("\nTest 6: Bad JSON request body")
    try:
        r = await client.post(
            f"{GATEWAY}/v1/chat/completions",
            content=b"this is not json",
            headers={"content-type": "application/json"},
            timeout=OFFLINE_TIMEOUT,
        )
        assert r.status_code == 400, f"Expected 400, got {r.status_code}"
        data = r.json()
        assert "error" in data, "Expected error in response"
        print(f"  PASS — status=400, error={data['error']}")
        return PASS
    except Exception as e:
        print(f"  {_verdict(e)} — {e!r}")
        return _verdict(e)


async def check_empty_messages(client: httpx.AsyncClient) -> str:
    """Test 7: Empty messages array → still responds (no crash)."""
    print("\nTest 7: Empty messages array")
    try:
        body = {
            "model": "coding-35b-a3b",
            "stream": False,
            "max_tokens": 10,
            "messages": [],
        }
        r = await client.post(
            f"{GATEWAY}/v1/chat/completions",
            json=body,
            timeout=LIVE_TIMEOUT,
        )
        # Should not crash — may return error or empty response
        assert r.status_code in (200, 400, 502), f"Unexpected status: {r.status_code}"
        print(f"  PASS — status={r.status_code}")
        return PASS
    except Exception as e:
        print(f"  {_verdict(e)} — {e!r}")
        return _verdict(e)


async def main():
    global GATEWAY, LIVE_TIMEOUT
    parser = argparse.ArgumentParser(description="E2E Gateway Tests")
    parser.add_argument("--gateway", default=GATEWAY, help="Gateway URL")
    parser.add_argument("--timeout", type=float, default=LIVE_TIMEOUT,
                        help="seconds allowed per live check (default 600); "
                             "raise it when the slot is busy, lower it to fail fast")
    args = parser.parse_args()

    GATEWAY = args.gateway.rstrip("/")
    LIVE_TIMEOUT = args.timeout

    print(f"    live checks: {LIVE_TIMEOUT:g}s each; this rig serves ONE at a time")
    print(f"=== E2E Gateway Tests — {GATEWAY} ===\n")

    async with httpx.AsyncClient() as client:
        results = []

        # Always-runnable tests (no upstream needed)
        ok = await check_health(client)
        results.append(("health", ok))

        if ok != PASS:
            print("\n⚠️  Gateway not reachable. Skipping live tests.")
            print("    Start the gateway: python -m gateway")
            sys.exit(1)

        results.append(("models", await check_models(client)))
        results.append(("bad_json", await check_bad_json(client)))
        results.append(("empty_messages", await check_empty_messages(client)))

        # Live tests (require upstream llama-server)
        print("\n--- Live tests (require upstream llama-server) ---")
        results.append(("non_stream", await check_non_stream_completion(client)))
        results.append(("stream", await check_stream_completion(client)))
        results.append(("tool_call", await check_tool_call_stream(client)))

    print("\n=== Summary ===")
    total = len(results)
    passed = sum(1 for _, v in results if v in (PASS, WARN))
    timed_out = [name for name, v in results if v == TIMEOUT]
    failed = [name for name, v in results if v == FAIL]
    for name, verdict in results:
        print(f"  [{verdict}] {name}")
    print(f"\n{passed}/{total} passed")

    # A TIMEOUT is not a defect and must not be reported as one -- but it also
    # verified nothing, so it still exits non-zero. Saying WHICH of the two
    # happened is the whole point: on 2026-09-10 two checks that had merely
    # queued behind a live client turn were read as a broken pipeline, and only
    # the gateway log could tell them apart.
    if timed_out:
        print(f"\n  TIMED OUT: {', '.join(timed_out)}")
        print("  Nothing is proven broken. This rig serves ONE request at a time,")
        print("  so a client turn in flight delays every live check. Re-run when")
        print(f"  the rig is idle, or raise --timeout above {LIVE_TIMEOUT:g}s.")
    if failed:
        print(f"\n  FAILED: {', '.join(failed)}   <- these are real")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
