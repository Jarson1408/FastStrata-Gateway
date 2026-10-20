# Changelog

Version history collected from the Git tags and the Progress Log in
`.context/memory/active-context.md`, which until now were the only records and
disagreed with each other. Each release is tagged; `git checkout v4.6.0` is the
rollback path.

Requires a gateway restart after every upgrade.

---

## Unreleased

### The FastStrata build: one engine (2026-10-20)

Branch `faststrata`. The gateway that ships with FastStrata, built into a
public tree with no history by `install/release/make_release.py`.

**Removed**, because FastStrata is the only engine it starts: the eGPU
ExLlamaV3 supervisor, the llama-server vision supervisor, `host_memory.py`,
`install/exllamav3/`, the Ollama, llama.cpp, nomic-embed and Mac installers and
launchers, their settings (`exllama_*`, `egpu_restart_cooldown`,
`idle_timeout_seconds`, `llama_server_binary`, `vision_port`, `exllama_url`),
their GUI controls (the host-RAM fields, the eGPU idle timeout, Sleep eGPU),
the llama-server fields of a model spec, and the tests and diagnostics of those
files. A settings.json row naming `exllamav3` or `llama-server` is dropped with
a reason, as any unknown engine is.

**Changed:**

- The FastStrata folder is `polystrata_dir`, else the folder above the
  gateway's own -- FastStrata's setup puts the gateway in
  `<FastStrata>\gateway\`. `installed()`, the Models table and the hand-over
  read it; until now `installed()` named `C:\tools\PolyStrata` whatever the
  setting said.
- A swap to a model whose run config or GGUF is not on disk is refused BEFORE
  the running model is stopped, as content. Before, the running model was
  stopped and the start then failed, which left the card empty.
- The port cleanup on a swap ends only a FastStrata process (python,
  strata-poly) listening on 8080. Before, it ended whatever listened there.
- `_stop_polystrata_server` assigned `_POLYSTRATA_CURRENT_CONFIG = None` to a
  local (no `global`), so the supervisor kept believing the stopped config was
  loaded.
- Settings in `%APPDATA%\faststrata-gateway`, the OpenCode provider
  `faststrata`, the autostart task `faststrata-embeddings`, the package
  `faststrata-gateway`: a PC with Local-LLM-Gateway keeps that one's settings,
  OpenCode models and task. One preset, "FastStrata (this PC)", with the four
  models FastStrata's setup offers and the run config names it writes.
- A vision model on a `remote` server takes the chat route, where llama.cpp
  sees an image (measured 2026-09-13); FastStrata keeps the prompt.
- `/v1/models` judged `installed` against the gateway's own folder and listed
  each run config's file name as a model; it now asks the FastStrata folder
  and lists the id. Found by running the gateway against C:/tools/PolyStrata:
  every model came back `installed: false`. The Ollama-era
  `local-coder:35b-a3b` it always added is gone too.
- The desktop app writes gui.log beside the gateway when it has no console
  (pythonw, the shortcut setup makes): on the first trial install it started
  the model and was gone a minute later with nothing on disk saying why. Its
  single-instance name is its own: with Local-LLM-Gateway's app running, the
  shortcut found that app, asked it to show itself and quit.
- The launchers find their own folder instead of `C:\tools\Local-LLM-Gateway`.

**Kept:** LODESTAR, on by default, with its switch. The eGPU preset survives as
a test fixture (`gateway/tests/_rig_presets.py`, ExLlamaV3 as `remote`, where
discovery wins as it did), so the window, discovery and Gemma tests keep the
data they were measured on.

Verified: pytest 2015 passed, 1 skipped; `evals/runner.py` 25/25; mypy clean on
`gateway/lodestar` and `gateway`; ruff clean.

### The PolyStrata fork: Qwen3.6 on a 6 GB laptop card (2026-10-14 … 10-18)

This line forked from upstream at `3102057` (2026-10-15) and serves
Qwen3.6-35B-A3B, Ornith 1.5 and Gemma 4 through PolyStrata
(`strata-poly-kvq8`, 144k window, KV q8) on an RTX 3060 Laptop 6 GB. **It does
not contain upstream's ten later commits, release v7.1.0 among them**, so
`pyproject.toml` here still says 7.0.0. Two agents worked on it; the commits
are told apart by the `Co-Authored-By: Claude` trailer, which the Gemini
commits do not carry.

**Published 2026-10-18** to
the author's GitHub as Local-LLM-Gateway-PolyStrata
as the branch `fix/qwen36-stability`, eight commits on top of `main`
(`5aad7e2`, the Gemini work), and merged into `main` the same day as a
fast-forward. Verified on the branch: 2254 tests, 25/25 behaviour scenarios,
mypy and ruff clean.

**Gemini — `b169c50`, `e0b90c0`, `5aad7e2` (2026-10-17), earlier `399b4a8`
(2026-10-14):**

- OpenCode 2.0 tools (`399b4a8`): `TOOL_FAMILIES` mapping model calls onto the
  client's names (`shell`, `subagent`, `patch`, native `skill(id)`), the `path`
  parameter, and a set of `test_scripts` probes for OpenCode's logs and schemas.
- The PolyStrata engine (`b169c50`): `ENGINE_POLYSTRATA`, the supervisor in
  `app.py` (`_ensure_polystrata_server`, auto-start, auto-stop, hot swap between
  the three model configs, restart cooldown) and `test_polystrata_supervisor.py`.
- Parser auto-healing (`b169c50`): bare and fenced JSON tool calls, chained
  calls, JSON inside `<function=NAME>`, a speculative buffer of up to 256 KB
  for large writes, `repair_json` on malformed calls.
- Loop guards (`b169c50`): intra-turn duplicate suppression, an intra-turn loop
  stop that ends the stream `tool_calls`, deferral of the search tools so the
  loop breaker can steer in-stream.
- Sampling (`b169c50`): temperature 0.6 for the thinking models, presets
  `qwen-coder-exploratory` (0.6) and `qwen-coder-high` (0.8), the loop
  escalation capped at `max(base, 0.8)`; `context_reserve` 8192 for the then
  64k window; two llama-server launchers for Qwen2.5 7B on 6 GB.
- `e0b90c0`, `5aad7e2`: context and semantic-memory bookkeeping.

**Claude — branch `fix/qwen36-stability`: `aac72fb`, `fc73291` (2026-10-17),
`09fb8c6`, `81c7703` (2026-10-18):**

A read of 208 Qwen3.6 turns against OpenCode's own `opencode.db` found that
most of the reported looping was made by the gateway. Details and measurements
in AGENTS.md, "The Loops the Gateway Made".

- **grep was sent as glob.** A nameless JSON call with `pattern` always
  resolved to `glob`; `choose_search_tool` now weighs the arguments.
- **Windows paths survive repair.** `repair_json` no longer turns the `\t` of
  `lib\tps.py` into a TAB; a JSON body in `<function=NAME>` keeps its arguments.
- **No half-sent calls.** `read` is deferred, and both loop guards ask
  `emitter.has_streamed` before suppressing a call.
- **The token limit, both ways.** Complete calls run when a turn hits
  `max_n_predict`; a call the limit cut in half never does (it used to finish
  `tool_calls`, i.e. a `write` with half a file). One decision for all four
  stream readers.
- **`temperature_floor`** per model: adaptive sampling no longer lowers a
  thinking model or tries greedy decoding; 0.6 for Qwen3.6 and Ornith.
- **A quoted `{...}` no longer stalls the scanner** (`09fb8c6`): a closed but
  invalid object waited for a completion that could never come, and the turn's
  closing `shell` call reached the chat as text.
- **OpenCode 2.0 compactions are recognised** without touching the cached
  prompt prefix: "Create/Update the anchored summary…" in the last user turn
  is `SUMMARY_ANCHORED`; the warning goes to the end of that turn, tools stay
  described, a tool call is dropped at emission. The legacy path is unchanged.
- Qwen3.6 preset: `max_n_predict`, `max_output`, `context_reserve` 16384 at
  the 147456 window. Lint/type findings left by `b169c50` cleared, among them
  an undefined `sys` in `_polystrata_spawn`.

Outside git, same days: `settings.json` (the values above), and OpenCode's
`~/.config/opencode/opencode.json(c)` — default model `qwen3.6-35b-a3b`,
`limit.output` 16384, the stale `coding-3b-q8`, `coding-35b-q4`, `-nvfp4`,
`-mxfp8` and the duplicate `qwen3.6-35b-a3b-ud-q4_k_m` entries removed. Backups
sit beside each file as `*.bak-20261017-qwen36-stability`.

### The client threw the picture away, and nothing anywhere said so

A vision model asked to read oscilloscope captures reported that it cannot
see. It was right: OpenCode removed every image before sending and put
`ERROR: Cannot read "<file>" (this model does not support image input).` in
its place, because `capabilities.input.image` is derived from `modalities`
and `opencode_export.py` wrote only `attachment: true`. **The whole event was
an absence** — no `MULTIMODAL:` line, no image encode upstream, no error from
anyone, because from the gateway's side nothing failed.

- **`models_block` now exports `modalities` for every model.** `??` falls
  through on absence only, so a list omitting `"image"` says false on purpose
  while no list at all leaves the question to a default this gateway does not
  control. A capability the client decides is not one the gateway can assert
  by naming a model `-vision`.
- **`normalizer.detect_stripped_media`** reads the client's substitute note,
  and the request path logs a warning plus the audit status
  `client_stripped_media` (declared in `metrics._ALLOWED`). It cannot recover
  the bytes — they never left the client — but the next occurrence is a log
  line instead of an afternoon. Asked of the RAW messages, user turns only,
  only when no payload arrived, and only for a model that could have served
  one; verified against this repository's own source rather than a fixture,
  because `client_contract.py` really does carry the phrase on one line.
- **The wording lives in `client_contract.py`** as `stripped_media_pattern`,
  with the rest of the strings the gateway reads from outside itself.

### The reasoning budget the gateway actually ran was still 24,000

The raise to 40,000 on 2026-09-12 moved the dataclass field. `from_env` --
the constructor every real start uses -- kept its own fallback of `24000` and
settings.json overrode nothing, so every budget trip in the log since reads
`limit 24000`. Found when it cut a seven-image oscilloscope analysis
mid-sentence on the seventh image.

- The fallback is **derived from `max_n_predict`** rather than respelled, so
  the two cannot drift again and raising the ceiling moves the budget with it.
- `TestTheRunningBudgetIsTheDocumentedOne` compares `from_env` against the
  field default. The existing ordering tests all build `GatewayConfig(...)`
  directly, which is why the running value was free to disagree for two days.
- `qwen3.8-27b-vision` gets `max_n_predict: 24576` (budget 59,999 chars, ~9
  min of reasoning at a measured 35 tok/s) **and `context_reserve: 24576` with
  it** — a ceiling raised alone overflows the window by 8,192 tokens, which an
  upstream answers by discarding the middle of the prompt in silence.
  `TestPromptPlusOutputFitsTheWindow` checks the relation across every preset.

### A turn cut by a guard now counts toward the loop escalation

`count_retry_depth` needs a FAILED tool call, and a cut turn has none, so all
eleven guard cuts in the audit log were invisible to the temperature
escalation -- which has therefore never once reached its depth>=3 branch in
7,273 turns. `runaway.GuardCutStreak` counts consecutive cut turns per session
and feeds `adaptive.effective_loop_depth`.

- Cuts enter at the **exploration** end (2 in a row → temp 0.6, 3 → 0.7, 4+ →
  0.8) and never at the greedy step, because temperature 0.0 is what makes a
  stuck model repeat verbatim -- the condition that just fired.
- One cut changes nothing; `code`-in-reasoning never counts.
- The increment hangs off `RunawayWatch.check()`, the one place every verdict
  is produced, and `TestEveryWatchReportsItsCuts` reads app.py as source.

### Handing the card over now hands the host RAM back too

`/unload` frees VRAM and leaves the ExLlamaV3 process alive holding 18.4 GB of
commit. With the vision server up, system commit sat at 66.5 GB of a 67.8 GB
limit -- and the allocation that failed was the client's: OpenCode died
mid-turn with nothing in its own log, and a pytest run in the same minute
returned MemoryError.

- A hand-over to vision ends the ExLlamaV3 process. The `/unload` ask still
  comes first and `busy` still refuses, so a generation in flight is never
  killed.
- `_stop_egpu_server` clears `_EGPU_LAST_SPAWN`: a stop we ordered is not a
  crash and must not arm the restart cooldown.
- Cost: a cold start on the next text turn, measured at 31 s.

---

## v7.0.0 — 2026-09-13 — one card, two engines

Two bodies of work that never shipped separately, and they belong in one
release because that is what happened: the volatile-frame hardening below
landed first, and the multi-engine rig was built on top of it. `v6.1.0` is
the last tag before both.

### The lock file said "verified against" and named versions nothing was

Cutting this release ran the checklist in `INSTALL.md`, and `pip_audit` found
**twenty advisories in five packages** — every one of them against a version
this project had already moved past. Eleven of fifteen pins in
`requirements.lock` disagreed with what the suite actually runs on: fastapi
0.109.0 pinned against 0.136.1 installed, starlette 0.35.1 against 1.0.0,
pydantic 2.5.3 against 2.13.4.

**A stale lock does not merely fail to protect.** Its header claims the release
was verified against those versions, and it points whoever installs from it at
software nobody tested, carrying the vulnerabilities the pinning was supposed
to make visible. S-2026-03 records that pinning hides advisories because there
is no Dependabot here; it did not anticipate the pins being fiction.

Regenerated from the environment the gates pass in, then `starlette` and
`pytest` raised past their remaining advisories and the whole suite re-run on
them. `pip_audit`: **no known vulnerabilities found.**

    before   20 advisories, 5 packages
    after     0

### The version was in two files and nothing compared them

`gateway/__init__.py` says the version "cannot drift from pyproject.toml"
because it reads installed package metadata. That is true of an INSTALLED
package and false of a source tree: metadata is a snapshot from the last
`pip install`, so bumping to 7.0.0 left `gateway.__version__` saying 6.1.0 —
and nothing noticed, because nothing in the package reads it.

What a reader sees is `GATEWAY_VERSION` in `app.py`, a literal.
`test_version_is_one_number.py` compares the two files a human edits, and
checks the CHANGELOG has a heading for the version being released — this
repository cut v7.0.0 with TWO sections still headed `Unreleased`. **A claim
that two things cannot disagree is worth less than a test that fails when they
do.**

### One card, two engines

The eGPU rig gains a multimodal model on llama-server alongside the text model
on ExLlamaV3. They cannot both hold the card, so the model the client asks for
is now what decides which engine holds it.

- **A preset gains `models`.** `gateway/model_registry.py`: every model, the
  engine that serves it, its address and its own settings. An alias map maps a
  name to another name; what had to be decided was a PROCESS. A preset with no
  `models` list resolves nothing and behaves exactly as before, which is what
  keeps every existing settings.json working.

- **`gateway/engines.py` hands the card over.** To vision: unload ExLlamaV3,
  wait for the VRAM to actually come back, spawn llama-server. To text: kill
  the vision PID (llama-server has no `/unload`; ending it is the only way its
  memory returns), wait, start ExLlamaV3. Both directions under one lock, and
  both after the concurrency slot is held — which is what guarantees nothing is
  generating on the engine about to be unloaded.

- **The kill is by PID and never by image name.** Engram's embedding server on
  `:11435` is also a `llama-server.exe`, and killing it is silent: Engram drops
  to BM25 lexical search rather than reporting an error.
  `TestOnlyTheTrackedPidIsEverKilled` reads the module as source as well as
  exercising the call path.

- **The eGPU supervisor learned to yield.** `_ensure_egpu_server` is called
  from the lifespan and from `/health` every 15 s, neither of which knew a
  vision server could exist. It now declines while llama-server holds the card
  — 23 GB plus 13.6 GB on a 24 GB card is an out-of-memory in whichever process
  asks for the next allocation, quite possibly the one serving the user.

- **A vision model rides a second transport, and the first claim here was
  wrong.** This entry said llama-server takes images on `/completion` through
  `<__media__>` plus an ordered `multimodal_data` array, and that vision
  therefore cost no architectural change. That was read out of
  `llama-server-impl.dll` and never measured. Measured 2026-09-13 against
  builds b10869 AND b10937, with an image of the digits **7431**:

        /completion, multimodal_data beside the prompt ..  37 prompt tok, "1"
        /completion, prompt as the documented object ....  HTTP 400
        /v1/chat/completions with image_url ............. 243 prompt tok, "7431"

  Every earlier check passed because the model guesses confidently — asked the
  colour of a blue square it said red. Only the digits separate reading from
  bluffing.

  So a model with `vision: true` now goes to `/v1/chat/completions`
  (`normalizer.build_chat_body`) and everything else keeps the byte-exact
  ChatML on `/completion`. The transport follows the MODEL, not the turn, so
  the upstream's prompt cache is not invalidated at every image. What the model
  READS is unchanged: the same `render_tool_calls` / `render_tool_results` /
  neutralization, handed over as message content, with tool results as a user
  turn. `--reasoning-format none` keeps `<think>` unparsed, because the FSM is
  what produces `reasoning_content` on every other path.

- **The chunk readers were not touched — and there are SIX, not the five that
  sentence claimed.** `_as_completion_chunk` normalises the chat shape into the
  completion shape and tells the two apart by the payload rather than by a
  flag. It was called from `_upstream_events`, which covers the four streaming
  readers; `_non_stream_completion` parses its own JSON and reads
  `chunk["content"]`, which a chat chunk does not have. The first live vision
  request through the gateway therefore returned an **empty answer with
  `finish_reason: stop`** — nothing raised, nothing logged, and the
  normaliser's own docstring asserting the coverage is what made it hard to
  see. It now calls the normaliser.
- **Vision turns would have reported zero tokens.** The normaliser mapped
  `usage`, which needs `stream_options.include_usage` — a field this gateway
  does not send. Measured against b10869: the final chat chunk carries
  `timings` and no `usage`. `tokens_evaluated` is `cache_n + prompt_n`, not
  `prompt_n`, because `/completion` reports the WHOLE prompt both cold and
  warm.
- **`/v1/models` now says which model can see.** The listing carried
  `context_window` and never `vision`, so a client — including this project's
  own `check_vision_e2e.py` — had to guess which model to send an image to. The five prompt-surgery features (LODESTAR prefill,
  `tool_choice` grammar prefill, silent reasoning retry, content continuation,
  loop recovery) ask `prompt_transport`, derived from the body; on the chat
  transport the guards still fire and still say so, they simply end the turn
  instead of retrying invisibly.

- **`<__media__>` is neutralized like `<|`**, and stays neutralized although
  nothing sends it any more: the marker belongs to a transport this rig may
  return to, and the gateway reads files nobody in the conversation wrote.
  llama-server does NOT check that markers and payloads agree — measured, every
  mismatch returns HTTP 200 with an image silently dropped or shifted, which is
  the forged-observation class rather than the 400 this entry first claimed.

- **`image_tokens_estimate` (1024) charges an image against the prune limit.**
  An image is a few characters of placeholder and costs hundreds to thousands
  of tokens — measured here at ~200 for a 600x300 PNG, over 1 000 with
  `--image-min-tokens 1024`; without the charge the prune limit was wrong in
  the only direction that loses a request.

- **A text model is told, not silently stripped.** An image sent to a model
  without vision becomes a one-line note instead of vanishing.

- **The model registry reached the dashboard and not the gateway.** Found by
  restarting: `from_env` read `models` only from the top level of settings.json,
  and the fallback to the active preset existed only in the GUI — so a freshly
  restarted gateway had an EMPTY registry: no routing, no per-model limits, no
  hot swap, no vision, while the dashboard showed four models. Confirmed
  against the live process: `/v1/models` returned the aliases and neither
  `qwen3.8-27b-vision` nor `gemma-4-26b`. An empty registry is SILENT by
  design — it is the correct state for a single-engine preset — so nothing
  reported it. One resolver, `config_file.active_models()`, for both callers,
  plus a `MODEL REGISTRY` line at startup. The fallback reads the preset the
  user NAMED, never one inferred: guessing gave a localhost gateway the Mac
  Studio's model list.

- **Seven findings from the targeted audit of 2026-09-12, fixed.** The audit
  answered one question — does every function do what it says — by sweeping the
  whole package: 593 functions, 55 log markers, 60 configuration fields, 32
  prompts.

  `O-2026-06`: one of the three stream readers wrote `reasoning_budget_guard`
  into the audit as a literal, so a loop or code trip during a content
  continuation was recorded as a budget trip — in the JSONL and in
  `llmgw_errors_total{type=...}`, which reads that field. `R-2026-05`:
  `/v1/models` knew only `model_aliases`, so the vision model and Gemma were
  invisible to any client that discovers models there. `R-2026-06`: `is_macos`
  removed — never read since the first commit of the skeleton.

  `A-2026-04`, `T-2026-04`, `T-2026-05` and `T-2026-06` are one pattern:
  **the parts were tested and the assembly was not.** `grammar.py` had 100%
  coverage and the `tool_choice` path that uses it had 0%. `destructive.py`
  had 90% and `_guard_destructive_calls`, which calls it, had 0%. Tests now
  drive the real `_stream_chat_v2`, and `_guard_destructive_calls`,
  `sdd_helper` and `UpstreamClient.resolve` directly. Each was verified by
  sabotaging the fixed site and watching the new test go red.

  Measured: functions never executed by the suite, `gui.py` aside, **42 → 33**;
  `upstream.py` 51% → 81%, `sdd_helper.py` 0% → 44%, `app.py` 71% → 76%.

- **The loop brake was keyed on another client's tool names.**
  `count_retry_depth` looked for "the same resource" under Windsurf's keys
  (`TargetFile`, `AbsolutePath`, `SearchPath`) while OpenCode sends `filePath` —
  so the resource was always `None`, the comparison fell back to whole argument
  strings, and an edit retried with a corrected `oldString` looked like a
  different call. Over five weeks: 959 adaptive adjustments, 2 RETRY_LOOP hits
  (both depth 2), **zero** at depth ≥ 3 and **zero** "Loop mitigation
  triggered!". Both halves of the mechanism — raising the temperature AND the
  quiet `loop_warning` prompt — hung on that one list.

- **A second list, twice over in the same file**, named `grep_search`,
  `read_file` and `list_dir` as the tools whose "not found" is an answer rather
  than a failure. This rig's are `grep`, `read` and `glob`, so the exemption
  covered NONE of the tools in use and an empty `grep` pulled the temperature
  from 0.3 down to 0.15. That one was not a dead feature but a live wrong one.

- **The fix is an import, not new logic:** `gateway/lodestar/tool_names.py` had
  the right answer from the start (`PATH_KEYS` contains `filePath`).
  `SEARCH_TOOLS` and `RESOURCE_KEYS` + `get_resource()` were added.
  `RESOURCE_KEYS` is deliberately a SUPERSET of `PATH_KEYS` rather than a
  widening of it: `PATH_KEYS` decides what `path_repair` may REWRITE, and
  widening the set that authorises a rewrite is not the same decision as
  widening the set that recognises a repetition.

- **The most expensive part: `test_adaptive.py` has 27 tests, all of them were
  green, and every fixture uses `TargetFile` and `grep_search`.** The tests
  were written in the same vocabulary as the code, so they confirmed its
  assumption instead of checking it against the client.
  `test_adaptive_vocabulary.py` uses names measured from `gateway_events.jsonl`
  (read 1 945, bash 1 320, glob 877, edit 271, write 160, grep 116); 11 of its
  22 tests go red against the old code, and one reads `adaptive.py` as SOURCE
  so a hand-written list cannot come back.

- **Limits that scale with the window belong to the MODEL.**
  `get_prune_limit()` read one process-wide `n_ctx`, discovered from
  `config.upstream`, so on the eGPU preset EVERY request was pruned to 212 992
  tokens: the vision model (window 196 608) was sent prompts 32 768 tokens too
  long, and the declared Gemma 196 608. Nor was it transient — while
  llama-server holds the card, `/health` asks `:5000`, gets nothing, and skips
  the adopt. `config.for_spec()` now resolves a model's window, reserve, output
  ceiling, spill threshold and sampling into a per-request copy, and a model
  that overrides nothing gets the same object back.

- **Who is authoritative about `n_ctx` depends on the engine.** For
  llama-server the DECLARATION wins — we passed `-c` ourselves. For ExLlamaV3
  and remote upstreams DISCOVERY wins, and only for the address it was
  performed against. Ask the party that knows.

- **`reasoning_budget_chars` is DERIVED from `max_n_predict`** and has no
  per-model knob of its own: the budget must fire before the ceiling, so a
  model that moves one has to move both. What is kept is the measured RATIO
  (78% at 3.13 characters per token), which reproduces 40 000 to within five
  characters.

- **Sampling may belong to the model** (`sampling_preset` + `sampling`), and a
  declared value beats what the CLIENT sends — the one place the gateway puts
  its own configuration ahead of the request, for the same reason the aliases
  exist: OpenCode hardcodes `temperature=0.55` for any id containing "qwen",
  which is a guess keyed on a NAME.

- **The prune limit's floor was on one branch of two**: a reserve larger than
  the discovered window produced a NEGATIVE limit. Unreachable while every
  window was 229k; ordinary from the moment windows stopped being the same.

- **`gateway/parsers/gemma.py`** DELEGATES to `Qwen3XmlFSM` rather than being a
  third copy of the scan loop — the prose-`<` defect of 2026-09-10 was in both
  existing parsers and found in neither. `detect_model_family` takes a family
  DECLARED by the registry, which beats sniffing digits out of a filename.

- **It was written before the weights existed, and every guess in it was
  wrong.** The file said so honestly and named the script that would settle
  it; the script did not exist. **A promise of a measurement reads exactly
  like a measurement.** Downloaded and run on 2026-09-13:

  - The thinking tag was guessed as `<thought>…</thought>`. The model emits
    `<|channel>thought` … `<channel|>` — not mirror images, and the closer is
    GENERIC (the same template frames tool calls), so the rewrite is stateful
    or it fabricates a `</think>` with no opener. While this was wrong nothing
    crashed: the reasoning arrived as `content`, markup and all, and the
    end-to-end vision check PASSED because the digits it greps for were
    somewhere inside the unparsed text. **A check that looks for a substring
    cannot see a turn whose structure is wrong.**
  - The parameter syntax was wrong too. Asked for this gateway's dialect the
    model answers `<function=read>` correctly and then `<filePath>value</filePath>`
    — the argument's NAME as the tag. The base FSM collected nothing and
    emitted `read({})`: valid JSON, and a call that cannot read anything. The
    rewrite is scoped by three conditions at once — inside a `<tool_call>`,
    outside a parameter VALUE, and only for names the declared tools have.
  - `call:name{…}` is now the only guess left, and is labelled as one.

  `test_scripts/check_gemma_dialect.py` exists now. Its own first version
  composed a prompt of its own and measured the answer to THAT, which sent the
  investigation after a call shape the model does not produce.

- **There is no dense `gemma-4-26b`.** The 26B is a MoE with 4B active
  (`26B-A4B`), and the placeholder path named a file in no repository. QAT
  `UD-Q4_K_XL` at 13.3 GB beats an ordinary 4-bit quant at 15.8 GB, because it
  was trained aware of the quantisation.

- **`--cache-ram` is always passed, and its default is why.** llama.cpp
  reserves 8192 MiB of prompt cache, allocated LAZILY — so the engine comes up,
  answers ONE request, and dies on the next with `failed to allocate memory for
  prompt cache state`. The vision check passed twice and the tool check killed
  the server, which reads as a model problem. 0 on this rig; a property of the
  MACHINE, so it is not in `ModelSpec`.

- **A VRAM budget can be right about the easy part and still be useless.**
  Derived from the model's own `config.json`: 20.0 KiB/token, 3 324 MiB free at
  262144. Measured on two windows: 13.6 KiB/token — LESS — against a fixed cost
  of 20 710 MiB where 15 500 was allowed, because llama.cpp's compute buffers
  scale with the window too. The errors did not cancel: 262144 left 384 MiB and
  a single 600x300 PNG costs about 750. Configured at 196608, which is what the
  Qwen vision entry already runs on.

- **A busy engine is not a dead one.** `ensure_vision_server` probed `/health`
  with a 1.5 s timeout and RESTARTED the engine on no reply; llama-server does
  not answer promptly while processing a large prompt on its single slot. A
  tool turn came back with no content and no tool calls — which reads as a
  model refusing to call a tool — because the gateway had killed the engine
  serving it. **We start that process, so `proc.poll()` is the authority, never
  a network timeout.** Neither half was covered: removing the probe broke no
  test.

- **A prompt is counted with the tokenizer of the model it is going to.**
  `count_tokens` asked the CONFIGURED upstream for every request: the wrong
  tokenizer while the text engine was up, and two 5-second timeouts per turn
  while it was down, ending in a character heuristic either way. A-2026-01 in a
  third costume.

- **Gemma 4 E4B was declared, measured and removed.** It runs on a different
  machine; a model has to be in the registry to be testable through the gateway
  at all. It passed — vision, thinking channel, a parsed tool call with its
  argument — and the entry left with it. The weights stay on disk: **a registry
  entry is a promise to SERVE a model; a file on disk is not.**

- **The Gemma parser has twice passed on whole strings and done nothing on a
  stream.** First `_drain_content` held back a partial copy of the word
  `call:`, so `call:read` went out as content one chunk before its `{` arrived.
  Then the parameter rewrite above: `_observe_frame` needs `<tool_call>` whole
  to know where it is, and the hold-back only kept prefixes of the REWRITE
  markers, so the frame arrived two characters at a time. Every parsing test
  runs at four chunk sizes, and that is what caught both.

- **And the fixture was thinner than the code.** `TOOLS` in `test_gemma_fsm.py`
  declared tool NAMES with no `parameters` schemas, so the candidate set the
  parameter rewrite reads was empty and four tests written to exercise it could
  not have touched it.

- **The Configuration tab lists the preset's models** — id, engine, context,
  vision, and whether the files are on disk — with a per-model editor, and a
  button that writes them into OpenCode's own config. That export merges into
  a parsed copy, touches one key, keeps a `.bak`, and asks first: the
  confirmation names which models would disappear from the client and how many
  JSONC comments a round trip would delete.

- **`test_scripts/check_vision_vram_budget.py`** computes the Q8 context that
  fits with Engram's embedding server resident. A script rather than a number,
  because the last hand-made answer in `AGENTS.md` was measured while that
  server was asleep. It says 229 862 tokens fit and recommends 196 608; the
  difference is the gigabyte a large image's vision-tower activations need.

- **Fixed on the way past:** `_non_stream_completion` decided how to name its
  model from `upstream.resolve()` rather than from the address it was sending
  to, so a request routed elsewhere — by escalation then, by the registry now —
  asked the wrong upstream. It uses `is_exllama_upstream` on the resolved
  target now, which is the one spelling of that question.

### The volatile frame

Found while reading `affaan-m/ECC` and other harnesses for ideas worth taking
before a public release. Both are the volatile LODESTAR frame; both were
silent.

- **A quoted marker could switch the supervisor off.** `_has_turn_frame`
  searched every message for `=== LODESTAR TURN ===`, and this repository's
  `AGENTS.md` contains that literal — so a turn in which the model read
  AGENTS.md, `gateway.log` or a captured prompt dump got no frame at all: no
  state directive, no TODO, no active-context handover, no budget notice, and
  no log line saying so. Measured on a two-message conversation: clean gives
  one frame of 1,257 characters, one poisoned tool result gives none.
  `already_injected` had the same shape against `=== LODESTAR PROTOCOL ===`,
  in the system message the CLIENT composes. Both now use a key the gateway
  writes, whose value is a token minted at process start.

- **The frame had no size limit.** The whole of `active-context.md` went into
  the block that is re-prefilled on every turn — 17,782 characters measured
  here, against the 735 `check_prefix_stability.py` calls healthy, about 4,500
  tokens per turn for history the turn does not use.
  `MAX_PROGRESS_ENTRIES` caps entries, not bytes. New setting
  `lodestar_context_max_chars` (4096, `LOCAL_LLM_GATEWAY_LODESTAR_CONTEXT_MAX_CHARS`,
  0 disables) keeps the head verbatim and sheds the oldest Progress Log
  entries, naming how many it left out.

- **The check that guards this could not see it.** Its fixture wrote a
  220-character `active-context.md`, eighty times smaller than the real file.
  `make_workspace` now builds a realistic log and refuses to run if it lands
  under the cap; `WASTE_BUDGET_CHARS` is derived from the cap.

Prompts 1.19.0 → 1.20.0 (`active_context_truncated`, `active_context_clipped`).
New eval expectation `turn_frame` asks whether the frame was actually
delivered, which every other check in `evals/` assumes.

### The budget was cutting work, and one path had no guards at all

- **`reasoning_budget_chars` 24000 -> 40000.** The guard was censoring its own
  evidence: after it landed every stored block above 24,000 sat at 24,00x. The
  uncensored distribution from before shows ten blocks up to 56,285 — and all
  ten come back clean from this project's own loop and code detectors. Measured
  3.13 chars/token with the upstream's tokenizer, so 40,000 is 78% of
  `max_n_predict` and still fires first; the budget retries invisibly,
  n_predict ends the turn dead.
- **`max_n_predict` left at 16,384.** A turn that reaches it takes a measured
  14.1 minutes; 24,576 would make it 23.0. Nine minutes buys four turns in six
  weeks at the price of one turn holding the only slot for a third of an hour.
- **The non-streaming path had none of the four guards.** Seven turns ran
  through it to the ceiling, ~14 minutes of the single slot each. The policy
  moved to `gateway/runaway.py` and all five readers now ask one
  `RunawayWatch`; the three later readers had been asking about the budget
  alone, and `app.py`'s own `_reasoning_budget_exhausted` was a fifth spelling
  of it.
- **A main-session turn that ends with its answer inside `<think>` is now
  continued, not mined.** The conjunction (reasoning >= 2000 AND content < 400)
  isolates 11 of 736 real no-tool turns while leaving the 301 legitimate short
  acknowledgements alone. The continuation closes `</think>` and asks for the
  answer; extraction stays as the last resort.

Prompts 1.20.0 -> 1.21.0 (`answer_rescue`).

### /metrics, and CI

- **`/metrics`** exposes the audit's facts as Prometheus counters, so "how is
  it doing right now" no longer means parsing a 6.4 MB JSONL. Everything hangs
  off `AuditLogger._emit`, the one place every audit event passes through — one
  call site, and a new event type is counted without anybody remembering to.
  No label carries a value the client chooses: `session_id`, model name and
  tool name are each a time series per distinct value, and an unknown value
  folds into `other` rather than being dropped. Requires the bearer token like
  everything else, and joins `/health` in the access-log filter. No new
  dependency.
- **One version string**, from package metadata. It was three: `app.py` twice
  and `gateway/__init__.py`, which said 2.4.0 against pyproject's 6.1.0 because
  nothing read it.
- **CI** runs the same commands as the pre-commit hook, in the same order, on
  windows-latest — a Linux job has never been run here, and a gate pinned to
  `continue-on-error` is not a gate. It ships with the release, so the two
  steps that name `install/release/` are guarded with `hashFiles`.

### Who gets the slot, and what may forge a result

- **The queue is no longer first-come.** `asyncio.Semaphore` is FIFO, so a
  title generator held the one slot while the agentic turn a human was watching
  queued behind it. Measured: 73% of requests carry no tools; queue wait is
  0 ms at p90 and 140 s at p99, worst case 43.5 minutes. `PrioritySemaphore`
  puts a turn with tools — or a compaction, which the client is blocked on —
  ahead of titles, graph builds and `/v1/embeddings`. FIFO within a priority;
  starvation bounded by `semaphore_timeout`, which is enough because agentic
  turns are serial. The subtle part is cancellation: a waiter cancelled in the
  same tick as its grant gives the slot back, or it leaks the only one there is.
- **A tool result could forge another tool result.** Content carrying
  `</tool_response>` left its block early, and a following
  `<tool_response name="write">Saved.</tool_response>` was indistinguishable
  from a tool that ran — a forged observation, from a file nobody in the
  conversation wrote. `_neutralize_wrappers` breaks only the closer of the
  block being rendered (`</tool_response>`, `</think>`, `</parameter>`), so a
  body is mangled exactly where it would escape. The prose markers
  (`=== LODESTAR TURN ===`, `[SYSTEM GUARD]`) are deliberately left alone: they
  are all over AGENTS.md, and the supervisor reads state rather than prompt
  text.

### Silence is not an answer to a parameter

`tool_choice`, `response_format`, `n`, `stop`, `logprobs`, `logit_bias`,
`stream_options` and `parallel_tool_calls` were read by nothing: the upstream
body is composed from scratch, so anything the pipeline does not explicitly
read was dropped without a word. A client sending `tool_choice: "none"` got
tool calls anyway.

`gateway/client_params.py` now gives every parameter a disposition — honoured,
ignored for a stated reason, or a 400 naming it. Newly honoured:
`tool_choice` (`none` suppresses the tool list entirely; `required` gets the
LODESTAR prefill; a named function is still refused, since a prefill cannot
pick one function out of the list), `response_format` (llama.cpp `json_schema`;
refused when the upstream has no grammar field), and
`stream_options.include_usage` (an explicit `false` is honoured; absence keeps
the old always-send behaviour, because OpenCode reads it).

`config.is_exllama_upstream()` replaces four hand-spelled `":5000" in ...`
checks, two of which ask about the resolved url rather than the configured one.

### GBNF, and the much larger thing it must not try

`tool_choice` naming one function is now honoured on llama.cpp: a prefill opens
`<tool_call>` and a one-line GBNF pins which function, leaving everything after
the opener unconstrained.

What is deliberately NOT done is parity with the eGPU's sampler filter. GBNF
has no lazy lexeme, so the nearest expressible rule forbids a parameter value
from containing a line `</parameter>` — and a `write` call's content
legitimately does; this repository's own tool-calling example contains that
line. Such a grammar would stop the model writing this project's files, and
would read as a model problem for a week.

Not verified against a live llama.cpp (none was reachable), which
`gateway/grammar.py` states and `test_scripts/check_forced_function.py`
measures when one is. That script reports SKIP rather than PASS against an
upstream that cannot constrain — its first version guessed the upstream out of
`/health`, which reports a nested status object with no url, and so reported
FAIL for a rig that could never have passed.

### The hook that runs is now the hook you edit

`.git/hooks/pre-commit` is a two-line forwarder to the tracked
`install/hooks/pre-commit`, not a copy. The copy cost a gate the day it
mattered: the secrets check below was added, and the very next commit ran
without it — the installed copy was months old, nothing said so, and the commit
reported four green gates out of five. `--check` distinguishes `STALE COPY`
from `installed`, and an older copy is still recognised as ours so an existing
clone can be re-installed rather than skipped.

### Nothing secret enters git, private repository included

- **`.context/MAC_CONNECTION.md` is untracked and gitignored.** It sat in 274
  commits with the SSH user, the key path, a Tailscale address and an
  employer's VPN network. Nothing in it was a credential by the usual test,
  which is why nobody looked twice. The remote is private, so this is a
  precaution rather than an incident — but the file stays on disk, where its
  value is, and out of a history that a third party holds.
- **New pre-commit gate**: `install/release/check_staged_clean.py` reads the
  STAGED blob against `policy.NEVER_COMMIT` — keys, tokens, VPN profiles,
  Tailscale addresses — plus path shapes (`*.ovpn`, `id_rsa`, `*.pem`, `.env`).
  Deliberately NOT the release's identity patterns: this repository is allowed
  to say where it runs, and redaction handles that on the way out.
- **The semantic memory was audited and redacted.**
  `test_scripts/check_memory_hygiene.py` reports and, with `--redact`, rewrites
  identifiers in `memory.db` using the release's own substitution table.
  Measured: 96 lessons, five naming a client project or a Windows account, none
  carrying a credential, hostname or address. `memory_admin.redact_memories`
  recomputes each changed row's embedding — a vector derived from text that no
  longer exists is the same silent cache inconsistency this project keeps
  finding elsewhere.
- Found on the way: **`lessons.json` was 32 lessons behind `memory.db`**, so an
  audit of the export missed the newest entry — the one with account names in
  it — and the text copy that makes B-2026-01 true had been stale for weeks.
  The redact run refreshes it.

### The release is a separate repository

`install/release/` builds a public tree beside this one, with its own git
history and one commit per release. A release branch could not work: this
repository's private material is in its HISTORY — `MAC_CONNECTION.md` in four
commits, `memory.db` in 359 blob versions — and a clone gets the history.

- `policy.py` holds SHIP, REDACTIONS and FORBIDDEN on one page. The whitelist
  is the safe direction; FORBIDDEN is what makes the policy true, scanned over
  the BUILT tree, so a missed rewrite means no release rather than a leaked
  one. The file quotes every string it forbids, which makes it its own canary.
- `AGENTS.md` is redacted, never trimmed. Identity goes (account names, the Mac
  hostname, the client codename, a real document title); RFC1918 addresses stay,
  because they disclose nothing and the presets are documentation.
- The build runs the scanner **and then the whole suite inside the tree it
  built**, because redaction rewrites strings that tests use as fixtures.
  Nothing pushes.

The cut exposed a pre-existing defect: `embed.py` exists byte-identical in
`LODESTAR_template/` and `.context/`, and both tests imported the `.context/`
copy — so the tested file was the one nobody receives and the shipped one had
no coverage. Both tests now import the shipped copy, and
`TestTheTwoEmbedCopiesAgree` pins the pair. A forwarder would have been the
wrong fix: `embed.py` locates its database from `__file__`, so it would have
written every lesson into the template.

---

## v6.1.0 — 2026-09-01 — what a single turn is allowed to cost

Everything below was driven by incidents on a live rig, and each measurement
is named so a future reader can check it rather than take it on faith.

`v6.0.0` is tagged retroactively at `becca78` in this release: it was written
up here on 2026-08-31 and never tagged, so the rollback path this file
promises did not exist for it.

### Runaway turns

- **Three reasoning guards** (`gateway/reasoning_code.py`,
  `gateway/reasoning_loop.py`, and a budget in `app.py`). They exist because a
  turn cost 32,000 tokens and half an hour of the only upstream slot, and
  nothing noticed. All three are conjunctive on `fsm.in_thinking`: a turn that
  has begun a tool call is never cut, since truncating arguments hands the
  client a JSON parse error.
  - *Code in reasoning* (`reasoning_code_lines`, 25) is the costliest case and
    the only one where the model is not stuck — it is doing the work in the
    place the client discards. Measured: eleven thousand characters of finished
    Python inside `<think>`, and the turn ended with no `write`. Told apart from
    a sketched formula by the RUN of code-like lines: engineering prose never
    produced a run longer than 2, the calculator produced 234. The gateway does
    NOT salvage the code and synthesize the write — it has the text but not the
    path, and fabricating a file write is the one mistake here that destroys
    work rather than wasting time.
  - *Reasoning loop* (`reasoning_loop_max_distinct`, 0.6) counts the share of
    DISTINCT lines in the tail. Counting repeats was the first attempt and was
    wrong: on the real 24,002-character block it fired at 4,500 during genuine
    analysis. Substantive reasoning never fell below 0.90 across nineteen
    thousand characters; the loop reached 0.30.
  - *Reasoning budget* (`reasoning_budget_chars`, 24000) is the backstop for a
    runaway that never repeats itself. Hard, not "wait and see whether it
    stops" — that was true of all 10,742 tokens of the incident.
- **Silent correction** (`reasoning_retry_max`, 1). A tripped guard used to end
  the turn with a `[SYSTEM GUARD]` paragraph, which asks a person to type
  "continue" for something the gateway diagnosed itself. The gateway now
  abandons the generation and starts another inside the SAME client request.
  The retry prompt CLOSES the thinking block before the model resumes — text
  telling a looping model to stop is precisely what it has been ignoring — and
  is an APPEND, so llama-server reuses the cached prefix and prefills only the
  correction.
- **`max_n_predict`** (16384). The client's `max_tokens` is a request, not a
  licence; OpenCode sends 32000.
- **Cancelling now closes the socket.** `worker.cancel()` alone leaves
  llama-server generating into a socket nobody reads: measured, a turn the
  client cancelled at 20:56:02 was still decoding 17,231 tokens later at
  21:09:59, holding the slot the whole time.
- **No anti-repetition sampler.** DRY was added as a third brake and removed
  the next day: it was the CAUSE of the corrupted file paths, not a brake on
  anything. A path is a repeated sequence, which is what DRY suppresses.
  Reproduced against the live model, same seed, DRY the only variable.

### Turn budgets

- **`max_subagent_turns`** (25) and **`subagent_turn_grace`** (10). Nothing
  bounded a subagent — every other limit derives from LODESTAR cycle state, and
  a subagent skips LODESTAR by design. Measured: 51 tool calls over 2 h 52 m on
  a single-slot rig, for a task whose whole output is a paragraph.
- **`max_cycle_turns`** (60) and **`cycle_turn_grace`** (20) now stop something.
  The config carried the number, the request path read it, and the only thing
  that happened was a log line. A configured limit that stops nothing is worse
  than no limit, because it reads as cover.
- **`subagent_min_report_chars`** (400). The client ends a subagent on its first
  turn with no tool call, and that text becomes the `task_result` its parent
  reads — so the announce-then-stop habit cost the parent everything. Measured:
  two subagents ran for minutes and handed back 80 and 150 characters of
  intention, while one of them had the whole answer in its reasoning. The
  threshold is on LENGTH, not phrasing.

### Loops the model cannot see

- **Loop breaker** (`gateway/loop_breaker.py`). The same call, same canonical
  arguments, executed a third time is suppressed. Measured: one identical
  command three times, minutes apart, every one succeeding and returning the
  same 4,773 characters. Only EXECUTED calls count, and any `write`/`edit`
  resets — re-running a suite after a fix is the same command on purpose. Every
  repeat in a turn is suppressed but the model is told ONCE, because a turn
  whose calls are all suppressed contains nothing else, and inside a subagent
  that wall of repetition became the whole answer handed to the parent.

### Paths

- **One path shape everywhere.** `_display_path` now emits forward slashes on
  every platform. One prompt carried the same directory eleven times in two
  separator styles, six of them from the gateway itself, in BOTH styles, from
  one file — half the lesson had been learned and left next to its opposite.
- **`gateway/path_repair.py`** repairs a near-miss path with the thing the model
  lacks: the filesystem. Exactly one candidate within a small edit distance, or
  nothing. NEVER for a write or an edit, where a non-existent path is the normal
  case — that is the only place a bug here could destroy a file.
- **The history is cleaned too.** Every unrepairable call still reaches the
  client, fails, and stays in the conversation as an exemplar. Measured: a
  prompt carrying the model's own `PROJEKTOACME_HW` twice produced
  `PROJKTOACME_HW` next — that string minus one character. When a guard fires
  repeatedly and the failures get WORSE rather than merely repeating, look at
  what they are leaving behind in the context.
- **Invented working directory** is REPAIRED, not refused. The first version
  refused via `error_chunk`, and OpenCode abandoned the whole request 14 ms
  later — strictly worse than the client's own permission prompt.

### Spill

- **The spill file is now the result, byte for byte.** `SpillStore.store` opens
  with an empty `newline`. Without it Windows added a carriage return per line,
  and because the gateway tells the model to read that file, the corruption
  compounded on every read-spill cycle. The consequence was not cosmetic:
  previews decayed to 0 KB for 21–39 KB results, so the model saw nothing,
  re-read, and looped until the client's own detector stopped the session.
  `_preview` now selects head and tail by NON-BLANK lines, and a
  content-addressed cache compares file size against the expected length so it
  can notice the file is not what its name promises.

### Protocol and client contract

- **Summary mode had never once fired.** OpenCode opens a compaction with "You
  are a context summarization **agent**."; the gateway searched for
  "**assistant**". Every compaction ever served ran with the full agentic
  apparatus on. What kept it invisible: `is_summary` was written to the audit
  event eighty lines before it was computed, so the field read False in every
  request event ever written — not silence, a confident wrong answer. The
  wording moved to `client_contract.py` as `summary_markers`, keeping both
  spellings, overridable.
- **Phase A's read is handed over, not asked for.** active-context.md goes into
  the turn frame as `[ACTIVE CONTEXT]` and the read step is marked done, saving
  a full round trip. The memory SEARCH is deliberately not skipped: a new task
  needs its own keywords.

### Operations

- **The `local-llm-gateway` Scheduled Task and the desktop app are
  alternatives, never both.** They race for port 18000 and the task wins, so
  the app's child exits at startup and the tray light goes red for a gateway
  that is running — just not the one that can reach the model.
- **`start_proxy.bat` no longer defaults `LOCAL_LLM_GATEWAY_UPSTREAM`.** The
  environment overrides `settings.json`, so its `http://127.0.0.1:11434`
  fallback silently beat the address configured in the app. Written when the
  model ran on the same machine; it broke the whole autostart path once the
  upstream moved to another host.

### Diagnostics and tests

- `test_scripts/check_prefix_stability.py` measures WASTE per turn — how much
  UNCHANGED conversation is re-prefilled — offline, or over prompts captured
  from a real session with `--dumps`. For months AGENTS.md claimed a
  `TestKvPrefixStability` guarded the KV split and no such test existed.
- `test_scripts/check_cancel_releases_slot.py` watches the upstream `/slots`
  until the slot is actually released. The offline suite mocks the upstream, so
  it can only prove the gateway CALLS the shutdown path.
- Unit tests 893 → **1029**; prompts 1.12.0 → **1.17.0**.

---

## v6.0.0 — 2026-08-31 — context management and a reproducible audit trail

Five stages that move the gateway from supervising the protocol to also
managing the context it fills and recording what the model was actually shown.
The agent loop stays in the client; nothing here changes the
`/v1/chat/completions` contract.

- **Tool-result spill** (`gateway/spill.py`). A result over
  `spill_max_inline_bytes` (16 KB default) is written to a content-addressed
  file under `spill/` and replaced in the prompt by a head/tail preview and its
  path, which the client's own `read`/`grep` can reach. The preview is a pure
  function of the content, so the same result renders to the same bytes on
  every turn — without that the KV prefix churns and costs a full prefill,
  which would be a worse trade than the tokens it saved. Responses the LODESTAR
  state machine reads as evidence (`# Active Context`, `[lessons] sim=`) are
  never spilled: replacing one would mean the model read active-context and the
  gateway did not notice, leaving it in Phase A synthesizing the same read
  forever. An unwritable store leaves the result inline rather than failing.
- **Pinned task.** `prune_messages` no longer evicts the opening request of the
  current LODESTAR cycle. It used to be the oldest block and therefore the
  first to go — the model kept the one-sentence Goal from active-context and
  lost the constraints that came with the request, which is why long sessions
  drifted into work nobody asked for.
- **Compaction note.** Evicted turns leave a deterministic
  `[GATEWAY COMPACTION]` record behind (tools used, files touched, failed calls
  with their target). No model is involved — summarizing with the LLM would
  occupy the single upstream slot in the middle of pruning and can hang.
- **`prompt` audit event** (`gateway/audit.py`). Digest of the exact string
  sent upstream, its size, how many characters of it the gateway injected,
  whether a prefill was applied, and the sampling parameters after adaptive
  adjustment. `log_request` describes what the client sent; only this describes
  what the model saw. Synthesized calls now log their full arguments. Full
  prompt text is written only under `LOCAL_LLM_GATEWAY_AUDIT_FULL_PROMPTS`,
  since it contains user source.
- **Session replay** (`gateway/session_report.py`, `test_scripts/replay_session.py`,
  Sessions tab). Reassembles the JSONL into sessions and turns and prints a
  timeline: state, prompt digest (marked when unchanged from the previous
  turn), injected size, temperature, and every tool call with the gateway's
  verdict. Pure module; the GUI and the CLI both sit on it.
- **Destructive-call guard** (`gateway/destructive.py`). Recognises
  model-issued `rm -rf`, `git reset --hard`, force pushes,
  `Remove-Item -Recurse -Force`, `rmdir /s`, destructive SQL, raw device writes
  and writes outside the workspace. Applies in every state, including repos
  with no `.context/LODESTAR.md` — which are always state 3 and previously
  passed through no verdict at all. `warn` by default (log it, let it through),
  `confirm` blocks and tells the model to ask the user. This is a tripwire, not
  a sandbox: the gateway executes nothing itself.
- **Tiered routing.** After `escalation_retry_depth` (4) repeats of the same
  failing action — past the range where adaptive sampling has already tried
  greedy and exploratory temperatures — the request can be routed to a larger
  model on another backend with a directive to change strategy. Off unless
  `escalation_model` is set, and only meaningful with `escalation_upstream`:
  one `llama-server` holds one model in VRAM, so switching without a second
  backend would reload weights mid-task. Prompts bumped to 1.6.0 for
  `escalation_directive.md`.

722 unit tests, 18 eval scenarios, `mypy` clean on the configured scope.

New settings: `spill_max_inline_bytes`, `compaction_enabled`,
`destructive_action`, `escalation_model`, `escalation_upstream`,
`escalation_retry_depth`, `audit_full_prompts`.

---

## v5.0.0 — 2026-07-30 — technical-debt plan complete

Phases 5 and 6, closing the plan derived from the 19-point audit of v4.3.2.

- Dependencies declared in `pyproject.toml` with `gui` / `memory` / `dev`
  extras and a lockfile. The old `requirements.txt` omitted PySide6 and
  sentence-transformers entirely and got the fastapi bound wrong.
- Versioned schema for `memory.db` with a migration runner and the index the
  queries always needed. `memory_admin` can now tell an unreadable schema from
  an empty database.
- **185 KB of dead code removed**: `_legacy.py`, `local_llm_gateway_v1.py` and
  the hand-rolled `--test` runner, none with a production caller.
- `CHANGELOG.md`, and `mypy` clean on `gateway/lodestar/` — which took fixing
  12 real findings, including one in the FSM where an alias lookup could put
  `None` where a `str` was expected.
- Five diagnostic scripts moved out of the repo root, honouring the rule
  `AGENTS.md` had always stated.

Totals across the plan: 387 → 552 tests, 17 behaviour scenarios where there
were none, `app.py` 2151 → 1714 lines, two security holes closed.

## v4.7.0 — 2026-07-30 — prompts and loop limits

Phase 4 of the technical-debt plan.

- **Prompts live in `gateway/prompts/*.md`, versioned apart from the code.** Nine
  of them were string literals across four modules, so changing a word meant a
  code release and nothing recorded which revision produced a given run. Moved
  snapshot-first: the digest of the rendered tool-calling block is unchanged, so
  the model sees byte-identical text. Every request event now carries
  `prompts_version`.
- **The gateway stops acting for a model that will not converge.** Past
  `max_synthesis_per_cycle` (5) it refuses to synthesize further protocol steps;
  past `max_cycle_turns` (60) it logs an ERROR. Not a silent stop — the prompt
  gains a directive telling the model to act or explain the obstacle, because a
  gateway that quietly stopped answering would leave it retrying forever. Closes
  audit finding C2, open since 2026-06-18.

## v4.6.0 — 2026-07-29 — deduplication, `app.py` split

Phase 3.

- One tool-name vocabulary (`lodestar/tool_names.py`) replacing three
  independent copies that had already drifted: one of them did not lowercase
  commands, so `EMBED.PY SEARCH` advanced two consumers out of three.
- `_handle_fsm_event` (421 lines, 53 branches, five tangled modes) split into a
  dispatcher and named handlers. Synthesis moved to `lodestar/synthesis.py`,
  structural repair to `gateway/repair.py`, skill lookup to `gateway/skills.py`.
- **Both request paths now reach the same verdict.** Streaming performed the
  pending protocol step for the model while non-streaming only dropped the call,
  so the same request behaved differently with `stream: false`.
- No hardcoded `C:/Users/user/...` left in production code.
- Emitter encapsulation, no more mutation of the caller's message history, and
  the module-global token cache removed.

## v4.5.0 — 2026-07-29 — evals and tool-level telemetry

Phase 2.

- **`evals/`** — protocol behaviour checked against recorded conversations from
  real incidents. Offline, under a second, 17 scenarios. Writing them exposed
  `phase_d_pending()` counting failed edits as commitments, and an injected
  regression proved the first six scenarios had a coverage gap.
- `AuditLogger.log_tool_call()` and `log_prune()` finally called — they had
  existed since v2 and were never invoked from anywhere. Events gained
  `session_id` and `synthesized_count`.

## v4.4.0 — 2026-07-29 — security and verifiability

Phase 1.

- **The bearer token now covers every path except `/health`,** and the catch-all
  proxy route runs on an allowlist. `POST /completion` used to reach
  llama-server with no token and skip the whole pipeline; with
  `BIND_ALL=1` that was an open proxy on the LAN.
- **Synthesized `bash` commands are sanitized.** The old rule replaced `"` with
  `'`, but bash still expands `$(...)` inside double quotes, so task text
  containing `$(...)` meant arbitrary command execution.
- The synthetic `active-context.md` write can no longer truncate the document.
- `pytest` exits 0 again, and the edit-test loop went from 71 s to 4 s.
- README documented the wrong env var name for the auth token, so anyone who
  configured one by following it had no authentication at all.

## v4.3.2 — 2026-07-17

Health polling no longer buries errors in the log view. `/health` lines are cut
from uvicorn's access log at source; ERROR/WARNING lines get their own buffer
behind an "⚠ Errors (N)" button in the dashboard.

## v4.3.1 — 2026-07-16

Requests that declare no tools (graph builders, title generators) bypass
LODESTAR entirely instead of computing and logging state they cannot act on.

## v4.3.0 — 2026-07-15

Tray and dashboard hardening, round 2. Template deploy is non-destructive —
before this, re-deploying into a live project wiped `active-context.md` and
`memory.db`. Added the upstream health chip, a live tray tooltip, and a 5000-line
cap on the log buffer.

## v4.2.1 — 2026-07-14

Phase D ran twice: the model re-read the nudge still sitting in history and
repeated the whole finalization, duplicating the lesson and closing the context a
second time. Completed nudges are now defused in the history, and `embed.py save`
refuses exact duplicates.

## v4.2.0 — 2026-07-13

Tray and dashboard hardening, round 1.

## v4.1.0 — 2026-07

Phase D nudge: a task that ends with a plain-text answer while
`active-context.md` still says `W toku` gets one synthetic reminder to save the
lesson and close the context.

## v4.0.0 — 2026-07

LODESTAR v2 module extraction: the protocol moved out of the request pipeline
into the `gateway/lodestar/` package.

## v3.1.0-lodestar-v2 — 2026-06

LODESTAR v2: per-call verdicts, forced-order checklist, deferred tool-call
emission.
