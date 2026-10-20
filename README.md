# FastStrata Gateway (v7.0.0)

The OpenAI-compatible gateway that comes with FastStrata: it sits between an agentic client — [OpenCode](https://opencode.ai),
Aider, anything that speaks the Chat Completions API — and FastStrata's server, and makes a local mixture-of-experts
model behave like a tool-using agent instead of a chat box.

```
OpenCode ──OpenAI API──▶ gateway (:18000) ──/completion──▶ FastStrata serve/server.py (:8080) ──▶ strata-poly
```

Point your client at `http://localhost:18000/v1` and give it a model name. Everything below happens without the
client knowing.

It is the FastStrata build of Local-LLM-Gateway: the same
request pipeline, parsers, guards and LODESTAR supervisor, with one engine — FastStrata — in place of that gateway's
ExLlamaV3, llama-server and Ollama.

## What it does for you

**It starts and swaps the models for you.** Each model is a FastStrata run config (`strata-qwen36-ud-q4_k_m.json`,
written by FastStrata's setup). Ask for a model and the gateway starts FastStrata's server with it; ask for another
and it stops the first — every byte of VRAM and RAM it held — and starts the second, sending keep-alives so the
client's turn does not time out during the load. A model whose run config or GGUF is not on disk is refused *before*
anything is stopped, as a sentence in the answer, so a typo in the client's model list never empties the card.
Closing the desktop app stops FastStrata.

**It parses the tool calls your model really emits.** Qwen3-style models write XML —
`<tool_call><function=read><parameter=filePath>…` — which an OpenAI-compatibility layer cannot turn into `tool_calls`
JSON. A streaming FSM does it here, tolerantly: a `<` in prose does not eat the call, a missing closer does not swallow
the next parameter, and a call that arrives truncated is repaired rather than handed to the client as a JSON parse
error.

**It separates thinking from answering.** `<think>` becomes `reasoning_content`, streamed in real time.

**It stops a turn that has gone wrong** instead of letting it hold your GPU for half an hour. A model writing finished
code inside `<think>`, a reasoning block that has stopped saying anything new, a turn past its budget, the same tool
call for the third time with identical arguments — each is caught, and the model is told why, as content it can read
on the next turn rather than as an error that ends the session.

**It manages the context window for you.** Prompts are pruned to `n_ctx - reserve` with the opening request pinned,
oversized tool results spill to disk with a preview, and the prompt stays append-only so FastStrata's conversation
cache keeps working.

**It supervises a memory protocol, LODESTAR** — on by default, with a switch in the desktop app. A deliberately simple
way to hold a project's memory: an active-context file the model reads before it starts, a searchable store of lessons
it wrote down before, and a rule that it updates both when it finishes. Off for any project without a
`.context/LODESTAR.md`, and nothing else in the gateway depends on it.

**Why it is enforced rather than asked for.** A model with few ACTIVE parameters drifts out of a procedure within a
session and starts the next one having forgotten there was one — and Qwen3.6-35B-A3B and Ornith 1.5 activate about
three billion parameters per token, which is exactly what instruction-following across a long session gets least of.
So the gateway does not ask: it gates the first turns on the reads actually happening, checks the tool RESULT rather
than the model's claim about it, and when the model calls the wrong tool or none, it makes the required call itself.

**What it reads lives in your project, not here.** `LODESTAR_template/` is a starting point you can deploy from the
desktop app and then rewrite; these are the three files the gateway looks at:

| file | what the gateway does with it |
|---|---|
| `.context/LODESTAR.md` | its EXISTENCE turns the protocol on for that workspace, and its text is injected into the system message as the procedure the model must follow |
| `.context/memory/active-context.md` | read and handed to the model in the turn frame instead of costing a tool call; `## Progress Log` and `Status:` are what it reads back |
| `.context/semantic-memory/embed.py` | shelled out to for lesson search and saving (`embed.py search "<keywords>"`; a hit starts with `[lessons] sim=`) |

The strings the protocol reads back are in `gateway/lodestar/client_contract.py`, and each is overridable from
`settings.json`.

## The models

FastStrata's setup offers four, and the gateway's preset "FastStrata (this PC)" lists all four. The ones not
downloaded show as *not installed* in the desktop app and are never swapped to.

| model id (what the client asks for) | run config | quantization |
|---|---|---|
| `qwen3.6-35b-a3b` | `strata-qwen36-ud-q4_k_m.json` | Qwen3.6-35B-A3B UD-Q4_K_M |
| `qwen3.6-35b-a3b-q3` | `strata-qwen36-ud-q3_k_m.json` | Qwen3.6-35B-A3B UD-Q3_K_M |
| `ornith-1.5-35b-a3b` | `strata-ornith-apex-i-quality.json` | Ornith 1.5 35B-A3B APEX-I-QUALITY |
| `ornith-1.5-35b-a3b-compact` | `strata-ornith-apex-i-compact.json` | Ornith 1.5 35B-A3B APEX-I-COMPACT |

Speed, quality and memory of each, measured, are in FastStrata's README. The context window of a model is the one
FastStrata's setup chose for the PC: the gateway's `n_ctx`, the run config's `--max-context` and OpenCode's
`limit.context` must agree.

A `remote` model is possible too: an OpenAI-compatible server somebody else runs, which the gateway reaches but never
starts or stops.

## Requirements

- Python **3.12+**
- FastStrata, installed by its setup. The gateway finds it in the folder above its own, or at `polystrata_dir`.
- Windows is the developed-and-run platform. The code is portable, but the launchers under `install/windows/` are
  Windows-only.

## Install

FastStrata's setup installs this gateway into `<FastStrata>\gateway\`. By hand:

```bash
pip install -e ".[gui]"          # add ,memory for LODESTAR's semantic memory
python -m gateway.gui            # the desktop app: tray, logs, models, settings
```

or `python -m gateway` for the gateway alone. `INSTALL.md` has the rest: the embedding service, OpenCode, and what to
check when something is slow.

## Configuration

Everything lives in `%APPDATA%\faststrata-gateway\settings.json` (`~/.config/faststrata-gateway/` elsewhere), editable
from the Configuration tab. It is a folder of its own, so a PC with Local-LLM-Gateway keeps that gateway's settings
apart.

| setting | what it decides |
|---|---|
| `models` | which models are served, from which run config, with which window and limits |
| `polystrata_dir` | the FastStrata folder, when it is not the one above the gateway's |
| `lodestar_enabled` | the memory protocol, on or off |
| `port` | where the gateway listens (18000) |

"Export to OpenCode" in the desktop app writes the model list into OpenCode's config under the provider `faststrata`,
touching nothing else in that file and keeping a `.bak` of it.

## Why the CHANGELOG reads the way it does

It is not a list of features. It is a record of what broke, what was believed about it, what was measured, and what
the difference cost — including the entries where the thing that was wrong had been written down as fact. If you are
deciding whether to trust this gateway with your work, that file is the honest answer.

## Where things live

Every module that carries a decision is on this map, and a test fails when one is missing.

```
gateway/
├── app.py                  FastAPI routes, the request pipeline, FSM events, LODESTAR synthesis, the FastStrata supervisor
├── normalizer.py           ChatML rendering, normalization, pruning, both upstream body builders
├── dialects/gemma4.py      Gemma 4 prompt + native tool protocol (one module per non-ChatML family)
├── config.py               GatewayConfig: env, aliases, sampling presets, per-model resolution
├── config_file.py          settings.json load + atomic merge-save, the FastStrata preset
├── client_params.py        every OpenAI parameter: honoured, ignored for a reason, or a 400
├── upstream.py             httpx client, mDNS discovery, retry, token counting
├── model_registry.py       what a preset serves, from which run config, on what terms
├── engines.py              a swap refused before anything stops, and how the refusal reads
├── opencode_export.py      writes the model list into OpenCode's own config
├── parsers/                THE FSM — three implementations, one factory:
│   ├── qwen3_xml.py          the base scan loop
│   ├── qwen25_coder.py       self-healing variant
│   └── gemma.py              delegates to qwen3_xml, adds Gemma's own markers
├── openai_sse.py           the SSE emitter: deferred tool calls, suppression
├── tool_schema.py          coercion, aliases, validation, JSON repair
├── repair.py               truncated JSON/XML repair, backtick state
├── path_repair.py          near-miss file paths → the file the model meant
├── adaptive.py             sampling from retry depth and context pressure
├── runaway.py              the three reasoning guards, asked by every reader
├── loop_breaker.py         same call, same args, third time → stop and say so
├── destructive.py          classifier for calls that can destroy work
├── spill.py                oversized tool results → disk + preview
├── scheduler.py            priority for the single upstream slot
├── audit.py                JSONL audit log with rotation
├── metrics.py              Prometheus counters, fed from the one audit path
├── session_report.py       audit log → sessions, turns, timeline
├── gui.py                  PySide6 desktop app (tray + dashboard)
├── gui_helpers.py          template deploy, autostart shortcut, log classification
├── subprocess_controller.py supervises the gateway child process
├── memory_admin.py         direct-SQLite lesson admin, no model load
├── prompts/                every injected prompt as a versioned .md file
└── lodestar/               the supervisor:
    ├── state.py              the per-request state machine
    ├── todo.py               the checklist, kept current mid-turn
    ├── policy.py             one shared verdict
    ├── context_injector.py   what gets injected, and where in the prompt
    ├── synthesis.py          calls the gateway makes on the model's behalf
    ├── tool_names.py         the ONE vocabulary of tool names and argument keys
    └── client_contract.py    strings the protocol reads from outside itself
```

`reasoning_code.py` and `reasoning_loop.py` sit beside `runaway.py` and hold the two detectors it asks.

## Checking your own rig

`test_scripts/` holds diagnostic runners for what a unit test cannot see — whether the prompt is really append-only,
whether a cancelled turn really releases the slot. Most need a live gateway, and each says so:

```bash
python test_scripts/check_prefix_stability.py    # offline, ~1 s
python test_scripts/check_e2e_gateway.py         # the smoke check, needs a live gateway
```

**The unit suite is not in this repository.** Every release is built with it, scanned, and run through all of it
before a single file is committed — the release refuses itself if anything fails — and it is then removed.

## Licence

MIT, see `LICENSE`.
