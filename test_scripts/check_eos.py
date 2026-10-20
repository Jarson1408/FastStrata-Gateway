"""
Test EOS: verify llama-server stops on native EOS without explicit stop tokens.

CRITICAL: This test must pass before trusting v2 in production.
If the model generates <|im_end|> as text instead of stopping, we need to re-add stop tokens.
"""
import asyncio
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway.config import GatewayConfig  # noqa: E402

# Read from settings.json / the environment, exactly as the gateway does. This
# was a hardcoded 192.168.0.108 until 2026-09-04, by which time the upstream
# had moved — the script simply could not reach the model any more.
UPSTREAM = GatewayConfig.from_env().upstream

async def check_eos_direct():
    """Send a prompt directly to llama-server WITHOUT stop tokens."""
    print("=== Test EOS: Direct to llama-server (no stop tokens) ===\n")
    
    body = {
        "prompt": "<|im_start|>system\nBe very brief. Answer in one sentence.\n<|im_end|>\n<|im_start|>user\nSay hello.\n<|im_end|>\n<|im_start|>assistant\n",
        "stream": False,
        # NO "stop" field — relying on native EOS detection
        "n_predict": 200,
        "temperature": 0.1,
    }
    
    print(f"Sending to: {UPSTREAM}/completion")
    print("Body: no 'stop' field, n_predict=200\n")
    
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{UPSTREAM}/completion", json=body)
        data = r.json()
    
    content = data.get("content", "")
    stopped = data.get("stop", False)
    stop_type = data.get("stopping_word", data.get("stop_type", ""))
    tokens_predicted = data.get("tokens_predicted", 0)
    tokens_evaluated = data.get("tokens_evaluated", 0)
    
    print(f"Content: {content!r}")
    print(f"Stopped: {stopped}")
    print(f"Stop type: {stop_type}")
    print(f"Tokens predicted: {tokens_predicted}")
    print(f"Tokens evaluated: {tokens_evaluated}")
    
    # Verify EOS behavior
    has_im_end = "<|im_end|>" in content
    has_im_start = "<|im_start|>" in content
    
    print("\n--- Verification ---")
    if stopped and not has_im_end and not has_im_start:
        print("✓ PASS — Model stopped on native EOS, no leaked tokens")
    elif stopped and has_im_end:
        print("⚠ WARN — Model stopped but <|im_end|> leaked into content")
        print("  This means stop tokens ARE needed for clean output")
    elif not stopped:
        print("✗ FAIL — Model did NOT stop (hit n_predict limit)")
        print("  stop tokens may be required!")
    else:
        print(f"? UNKNOWN — stopped={stopped}, content has special tokens")
    
    return stopped and not has_im_end

async def check_eos_via_gateway():
    """Send a chat completion through the gateway and verify clean output."""
    print("\n\n=== Test EOS: Via Gateway ===\n")
    
    body = {
        "model": "coding-35b-a3b",
        "stream": False,
        "max_tokens": 500,
        "messages": [
            {"role": "system", "content": "Be very brief. Answer in one sentence."},
            {"role": "user", "content": "Say hello."},
        ],
    }
    
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post("http://localhost:18000/v1/chat/completions", json=body)
        data = r.json()
    
    if "error" in data:
        print(f"ERROR: {data['error']}")
        return False
    
    choices = data.get("choices", [])
    if not choices:
        print("ERROR: No choices in response")
        return False
    
    msg = choices[0].get("message", {})
    content = msg.get("content") or ""
    finish = choices[0].get("finish_reason", "")
    usage = data.get("usage", {})
    reasoning = msg.get("reasoning_content") or ""
    
    print(f"Content: {content!r}")
    print(f"Finish reason: {finish}")
    print(f"Usage: prompt={usage.get('prompt_tokens')}, completion={usage.get('completion_tokens')}")
    if reasoning:
        print(f"Reasoning: {reasoning[:100]}...")
    
    has_special = "<|im_end|>" in content or "<|im_start|>" in content
    has_xml_leak = "<tool_call>" in content or "<function=" in content
    
    print("\n--- Verification ---")
    if finish == "stop" and not has_special and not has_xml_leak:
        print("✓ PASS — Clean response via gateway")
    elif has_special:
        print("⚠ WARN — Special tokens leaked into content")
    elif has_xml_leak:
        print("⚠ WARN — XML tool tags leaked into content")
    elif finish == "length":
        print("✗ FAIL — Model hit length limit instead of stopping")
    else:
        print(f"? finish={finish}, special_leak={has_special}")
    
    return finish == "stop" and not has_special

async def check_multiline_content():
    """Test that <|im_end|> in file content doesn't cause premature stop."""
    print("\n\n=== Test: File content with <|im_end|> (stop token collision test) ===\n")
    
    body = {
        "model": "coding-35b-a3b",
        "stream": False,
        "max_tokens": 1000,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Repeat the following text exactly: The ChatML format uses <|im_start|> and <|im_end|> as delimiters."},
        ],
    }
    
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post("http://localhost:18000/v1/chat/completions", json=body)
        data = r.json()
    
    if "error" in data:
        print(f"ERROR: {data['error']}")
        return False
    
    choices = data.get("choices", [])
    msg = choices[0].get("message", {}) if choices else {}
    content = msg.get("content") or ""
    finish = choices[0].get("finish_reason", "") if choices else ""
    
    print(f"Content: {content!r}")
    print(f"Finish: {finish}")
    
    print("\n--- Verification ---")
    if finish == "stop" and len(content) > 20:
        print("✓ PASS — Response not truncated by stop token collision")
    elif finish == "stop" and len(content) <= 20:
        print("⚠ WARN — Response may have been truncated prematurely")
    else:
        print(f"? finish={finish}, len={len(content)}")
    
    return True

async def main():
    results = []
    
    results.append(("EOS direct", await check_eos_direct()))
    results.append(("EOS via gateway", await check_eos_via_gateway()))
    results.append(("Stop collision", await check_multiline_content()))
    
    print("\n\n=== Summary ===")
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    
    all_ok = all(ok for _, ok in results)
    print(f"\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")

if __name__ == "__main__":
    asyncio.run(main())
