"""Test streaming with local-coder model (has Modelfile with <think> prefill)."""
import httpx
import json

chunks = []
with httpx.stream(
    "POST",
    "http://localhost:18000/v1/chat/completions",
    json={
        "model": "coding-35b-mxfp8",
        "messages": [{"role": "user", "content": "Napisz krotki wiersz o programowaniu"}],
        "stream": True,
    },
    timeout=120,
) as resp:
    for line in resp.iter_lines():
        if line.startswith("data: "):
            payload = line[6:]
            if payload == "[DONE]":
                chunks.append("[DONE]")
                break
            try:
                chunks.append(json.loads(payload))
            except Exception:
                chunks.append(payload)

# Analyze: count reasoning vs content chunks
reasoning_chunks = [c for c in chunks if isinstance(c, dict) and c.get("choices", [{}])[0].get("delta", {}).get("reasoning_content")]
content_chunks = [c for c in chunks if isinstance(c, dict) and c.get("choices", [{}])[0].get("delta", {}).get("content")]

with open("test_stream.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"Total chunks: {len(chunks)}")
print(f"Reasoning chunks: {len(reasoning_chunks)}")
print(f"Content chunks: {len(content_chunks)}")
if content_chunks:
    text = "".join(c["choices"][0]["delta"]["content"] for c in content_chunks)
    print(f"Content text: {text[:200]}")
if reasoning_chunks:
    reason = "".join(c["choices"][0]["delta"]["reasoning_content"] for c in reasoning_chunks)
    print(f"Reasoning (first 100): {reason[:100]}")
