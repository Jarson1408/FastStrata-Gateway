"""
Does cancelling a request actually stop the model?

The offline suite cannot answer this. It mocks the upstream, so it can prove
the gateway calls the shutdown path — but not that llama-server on the other
end of a real socket notices and abandons the slot. The bug this script exists
for was invisible to every offline test:

    2026-08-31 20:46:52  STREAM START
    2026-08-31 20:56:02  STREAM CANCELLED     <- client gave up
    2026-08-31 21:09:59  n_decoded = 17,231   <- still generating, 13 min later

The slot is the whole machine. A generation nobody will ever read holds it for
as long as n_predict allows, and the user's next message queues behind it.

Needs a live gateway AND the model. Asks the UPSTREAM directly — /slots on
llama-server, `active_requests` from /health on the ExLlamaV3 server — because
that is the only place the truth lives: the gateway cannot report a generation
it has already stopped listening to.

The same defect was then measured on the other upstream, 2026-09-11:

    07:46:07  STREAM START
    07:50:38  reasoning budget exceeded -> gateway abandons, retries
    07:57:50  still generating at 95% GPU, holding the only slot

There the gateway abandoned the turn ITSELF, so the retry it had just issued
queued behind the generation it was replacing, and the user's turn stopped
dead with no error at all.

    python test_scripts/check_cancel_releases_slot.py

Exit 0 = the slot was released. Exit 1 = it was not.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway.config import GatewayConfig  # noqa: E402


def _upstream() -> str:
    """The address the RUNNING gateway uses, not one written down here.

    This was a hardcoded 192.168.0.108 until 2026-09-04, and by then the
    upstream had moved to another host — so the script could not read /slots,
    printed a warning, and went on to report "the model finished before the
    cut", which reads as a fixture problem rather than a broken script. Same
    class as S-2026-01: an address that stopped being true, in a file nobody
    re-read.

    GatewayConfig.from_env() is the gateway's own resolution order —
    settings.json, overridden by the environment — so the two cannot disagree.
    """
    try:
        return GatewayConfig.from_env().upstream
    except Exception:
        return os.environ.get("LOCAL_LLM_GATEWAY_UPSTREAM", "http://127.0.0.1:8080")


GATEWAY = os.environ.get("GATEWAY_URL", "http://127.0.0.1:18000")
UPSTREAM = _upstream()
MODEL = os.environ.get("GATEWAY_MODEL", "local-coder:35b-mxfp8")

# How long to let the model run before pulling the plug, and how long to give
# the gateway to propagate that. The grace is upstream_close_grace (2 s) plus
# room for llama-server to notice on its next write.
RUN_SECONDS = 8.0
RELEASE_TIMEOUT = 15.0
POLL_SECONDS = 0.5


def _busy_endpoint() -> str:
    """Which endpoint on THIS upstream reports whether it is generating.

    llama-server has /slots. The ExLlamaV3 server does not — it reports
    `active_requests` from /health. The same failure exists on both and was
    measured on both (llama-server 2026-08-31, ExLlamaV3 2026-09-11), so one
    runner covers them rather than two that drift apart.
    """
    try:
        with urllib.request.urlopen(f"{UPSTREAM}/slots", timeout=5):
            return "slots"
    except (urllib.error.URLError, OSError, ValueError):
        pass
    try:
        with urllib.request.urlopen(f"{UPSTREAM}/health", timeout=5) as r:
            if "active_requests" in json.load(r):
                return "health"
    except (urllib.error.URLError, OSError, ValueError):
        pass
    return ""


BUSY_VIA = _busy_endpoint()


def slot_state() -> tuple[bool, int]:
    """(is_processing, n_decoded) for slot 0, or (False, -1) if unreadable.

    On an upstream that only reports `active_requests`, the token count is not
    available and comes back as 0. That costs the progress line its numbers and
    nothing else: the question this script asks is whether the slot was
    RELEASED, and a count is decoration.
    """
    if BUSY_VIA == "health":
        try:
            with urllib.request.urlopen(f"{UPSTREAM}/health", timeout=5) as r:
                return int(json.load(r).get("active_requests") or 0) > 0, 0
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(f"  ! cannot read {UPSTREAM}/health: {exc}")
            return False, -1
    try:
        with urllib.request.urlopen(f"{UPSTREAM}/slots", timeout=5) as r:
            slots = json.load(r)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"  ! cannot read {UPSTREAM}/slots: {exc}")
        return False, -1
    if not slots:
        return False, -1
    slot = slots[0]
    nxt = (slot.get("next_token") or [{}])[0]
    return bool(slot.get("is_processing")), int(nxt.get("n_decoded") or 0)


def _upstream_readable() -> bool:
    return bool(BUSY_VIA)


def main() -> int:
    print(f"   upstream: {UPSTREAM} (from settings.json / environment)")
    if not _upstream_readable():
        print(f"ABORT: neither {UPSTREAM}/slots nor /health reports whether the")
        print("       upstream is generating — this check needs the")
        print("       upstream directly, because the gateway cannot report a")
        print("       generation it has already stopped listening to. Without")
        print("       it every outcome below is meaningless.")
        return 1

    busy, _ = slot_state()
    if busy:
        print("ABORT: the upstream slot is already busy. Wait for it to finish —")
        print("       this check cannot tell your generation from someone else's.")
        return 1

    body = json.dumps({
        "model": MODEL,
        "stream": True,
        # No tools: this must exercise the transport, not LODESTAR.
        "messages": [{
            "role": "user",
            "content": "Count slowly from 1 to 500, one number per line, "
                       "with a short remark after each. Do not stop early.",
        }],
        "max_tokens": 4096,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{GATEWAY}/v1/chat/completions", data=body,
        headers={"content-type": "application/json"})

    print(f"1. opening a stream to {GATEWAY} …")
    resp = urllib.request.urlopen(req, timeout=30)

    deadline = time.time() + RUN_SECONDS
    while time.time() < deadline:
        if not resp.readline():
            break
    busy, decoded_at_cut = slot_state()
    if not busy:
        print("   ! the model finished before the cut — nothing to cancel.")
        print("     Raise RUN_SECONDS or ask for a longer answer.")
        resp.close()
        return 1
    print(f"2. cutting the connection at n_decoded={decoded_at_cut}")
    resp.close()

    print(f"3. watching the slot for up to {RELEASE_TIMEOUT:.0f}s …")
    deadline = time.time() + RELEASE_TIMEOUT
    last = decoded_at_cut
    while time.time() < deadline:
        busy, decoded = slot_state()
        if decoded == -1:
            # Unreadable is NOT released. The ExLlamaV3 server answers /health
            # from the same event loop that runs the generation, so a
            # non-streaming request blocks it entirely and the probe times out
            # — and a timeout returning "not busy" would print PASS for a rig
            # that is wedged. Measured 2026-09-11 while testing this very
            # script. Keep waiting and let the deadline decide.
            print("   … upstream not answering; that is unknown, not released")
            time.sleep(POLL_SECONDS)
            continue
        if not busy:
            print(f"\nPASS: slot released {time.time() - (deadline - RELEASE_TIMEOUT):.1f}s "
                  f"after the cut, at n_decoded={decoded}")
            return 0
        if decoded != last:
            print(f"   … still decoding: {decoded} (+{decoded - decoded_at_cut} since the cut)")
            last = decoded
        time.sleep(0.5)

    _, decoded = slot_state()
    print(f"\nFAIL: {RELEASE_TIMEOUT:.0f}s after the client went away the slot is "
          f"still decoding — {decoded - decoded_at_cut} tokens past the cut.")
    print("      The upstream connection is not being closed. See")
    print("      _shutdown_upstream_worker in gateway/app.py.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
