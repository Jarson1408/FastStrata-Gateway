# Installing FastStrata Gateway

FastStrata's setup does all of this: it installs the gateway into `<FastStrata>\gateway\`, writes the models it
downloaded into the gateway's settings and into OpenCode's config, and makes a desktop shortcut for the app. This page
is the same procedure by hand, and what to check when something is slow.

> **The one thing easiest to forget:** besides the gateway, start the **embedding service**. Without it every call to
> the semantic memory takes ~8 s instead of 0.3 s, and LODESTAR makes two per task. Nothing breaks — it is just slower.
> Step 3.

---

## What it is made of

| part | runs where | what for | needed? |
|---|---|---|---|
| **FastStrata** (`serve/server.py`, `strata-poly`) | this PC | the language model; started and stopped by the gateway | yes |
| **gateway** | this PC | OpenAI API ↔ FastStrata, tool calls, guards, LODESTAR | yes |
| **embedding service** | this PC, CPU only | keeps LODESTAR's embedding model warm | no, but strongly recommended |
| `.context/` in a project | every project | LODESTAR's memory and rules | per project |

## Requirements

- Python **3.12+**
- FastStrata installed by its setup, with at least one model
- ~500 MB for the embedding model (`all-MiniLM-L6-v2`, downloaded once) if LODESTAR's memory is used

## Step 1 — the package

```bash
pip install -e ".[gui,memory]"
```

`gui` is the desktop app, `memory` is `embed.py`'s sentence-transformers (a few GB with PyTorch; leave it out if
LODESTAR's memory is not wanted). `pyproject.toml` is the source of truth for dependencies; `requirements.lock` pins
the versions this release was verified on.

## Step 2 — where FastStrata is

When the gateway lives in `<FastStrata>\gateway\` there is nothing to set: it finds FastStrata in the folder above its
own. Anywhere else, set `polystrata_dir` in `%APPDATA%\faststrata-gateway\settings.json` (or
`LOCAL_LLM_GATEWAY_POLYSTRATA_DIR`) to the FastStrata folder.

The Models table in the Configuration tab then says *ready* or *not installed* for each model, from the files on disk:
the run config FastStrata's setup wrote, and the GGUF it names.

## Step 3 — the embedding model, once

The first call downloads `all-MiniLM-L6-v2` from Hugging Face. Do it once, on purpose, before autostart — otherwise
the first LODESTAR task waits for the download:

```bash
python .context/semantic-memory/embed.py search "install"
```

Later loads run offline.

## Step 4 — autostart (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File install\windows\setup_autostart.ps1
```

Registers **one** task in the Task Scheduler, at logon: `faststrata-embeddings`, which runs `start_embed_service.bat`
(log: `embed_service.log`). Check and remove it with `-Action Status` and `-Action Disable`.

> **The gateway is NOT on that list, on purpose.** The desktop app supervises its own `python -m gateway`, and a task
> beside it would race it for port 18000. To run the gateway without the app: `install\windows\start_proxy.bat`.

**Linux** — the embedding service is a plain foreground process; hook it to systemd like anything else:

```bash
python .context/semantic-memory/embed.py serve --port 18001
```

## Step 5 — check

```bash
curl http://127.0.0.1:18000/health          # the gateway
curl http://127.0.0.1:18001/health          # the embedding service
```

The embedding service answers `{"status": "ok", "model": "all-MiniLM-L6-v2", "dim": 384}`.

## Step 6 — OpenCode

"Export to OpenCode" in the desktop app writes the models into OpenCode's config (`~/.config/opencode/opencode.json`)
under the provider `faststrata`, base URL `http://localhost:18000/v1`, each with its context limit. Only that provider's
model list is written; the rest of the file stays as it was, and a `.bak` of it is kept. A `.jsonc` with comments is
reported first, because a round trip through JSON drops them.

Ask for a model by its id (`qwen3.6-35b-a3b`), not by the GGUF's name.

## Step 7 — a new project

Deploy the template into the project's folder — from the app (the *Lodestar Deployer* tab) or by copying
`LODESTAR_template/`. A deploy **never overwrites** existing files, so it is safe on a live project. Fill in
`.context/project_rules.md` with the project's rules; they go into every request, so keep them short.

---

## When it is slow

```bash
python test_scripts/check_prefill_health.py
```

It splits the wait into the queue, the prompt (TTFT) and the generation. The `ms/1k tok` column should stay **flat** as
`tokens_in` grows; if it grows with it, the prefix cache is being lost and the whole prompt is read again.

The most common cause of "everything takes ages": **the embedding service is not running**.

```bash
curl http://127.0.0.1:18001/health || echo "not running - that is ~16 s per task"
```

## Notes

- **The wrappers live in `install\windows\`, and only there.** `start_proxy.bat` and `start_embed_service.bat` in the
  root are one-line forwarders. **Neither sets `LOCAL_LLM_GATEWAY_UPSTREAM`**: an environment variable beats
  `settings.json`, so a default in a wrapper would silently win over the address set in the app.
- **The interpreter is found, not hardcoded.** The wrappers ask Python where it is (`py -3`, then `python` on PATH)
  and use the `pythonw.exe` beside it. To force one: `set LOCAL_LLM_GATEWAY_PYTHONW=C:\path\pythonw.exe`.
- **`.bat` files need CRLF line endings.** `cmd.exe` breaks on LF. `.gitattributes` enforces it on checkout; if you
  edit them with a Unix tool, check the endings.
- **After any change under `gateway/`, restart the gateway** — it does not reload itself.
- The embedding service is **optional by design**: with nothing listening, `embed.py` loads the model in its own
  process. The vectors are bit-identical either way, so `memory.db` stays compatible.
