import os
import json
import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger("gateway.config_file")

#: The folder settings.json lives in, under %APPDATA% or ~/.config. Its own
#: name, not Local-LLM-Gateway's, so a PC that runs both keeps two settings.
SETTINGS_DIR_NAME = "faststrata-gateway"


def get_default_config_path() -> Path:
    """Return the default settings.json path depending on platform."""
    if os.name == "nt":
        base_dir = os.environ.get("APPDATA")
        if base_dir:
            return Path(base_dir) / SETTINGS_DIR_NAME / "settings.json"

    # Fallback/Linux
    return Path(f"~/.config/{SETTINGS_DIR_NAME}/settings.json").expanduser()


def load_settings_from_file(file_path: Path | None = None) -> dict[str, Any]:
    """Load settings dict from the JSON config file. Return empty dict on error."""
    path = file_path or get_default_config_path()
    if not path.is_file():
        return {}
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception as e:
        _log.warning("Failed to load configuration from %s: %s", path, e)
    
    return {}


def save_settings_to_file(config_or_dict: Any, file_path: Path | None = None) -> bool:
    """Merge settings into the JSON config file and write atomically.

    Returns True when the file is on disk, False when it is not.

    Existing keys the caller didn't provide are preserved (so a partial GUI
    save never drops forward-compatible settings), and the write goes through
    a temp file + os.replace so a crash can't corrupt settings.json.

    The return value exists because for months there was none: a failed write
    logged an error, returned None, and the dashboard went on to show
    "Configuration saved and gateway restarted" — after which the gateway
    really did restart, with the OLD upstream. The symptom ("I changed the
    address and it still talks to the old one") is undiagnosable exactly
    because the interface confirmed the save. D-2026-01.
    """
    path = file_path or get_default_config_path()

    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        _log.error("Failed to create settings directory %s: %s", path.parent, e)
        return False

    if isinstance(config_or_dict, dict):
        new_data = config_or_dict
    else:
        new_data = {
            "upstream": getattr(config_or_dict, "upstream", ""),
            "port": getattr(config_or_dict, "port", 18000),
            "context_reserve": getattr(config_or_dict, "context_reserve", 16384),
            "model_aliases": getattr(config_or_dict, "model_aliases", {}),
            "lodestar_enabled": getattr(config_or_dict, "lodestar_enabled", True),
            "spill_max_inline_bytes": getattr(config_or_dict, "spill_max_inline_bytes", 8192),
            "fallback_n_ctx": getattr(config_or_dict, "fallback_n_ctx", 32768),
            "semaphore_timeout": getattr(config_or_dict, "semaphore_timeout", 7200.0),
            "stream_max_duration": getattr(config_or_dict, "stream_max_duration", 7200),
            "loop_recovery_max": getattr(config_or_dict, "loop_recovery_max", 1),
        }

    merged = load_settings_from_file(path)
    merged.update(new_data)

    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        _log.info("Saved settings to %s", path)
        return True
    except Exception as e:
        _log.error("Failed to save settings to %s: %s", path, e)
        return False


# ---------------------------------------------------------------------------
# Configuration presets
# ---------------------------------------------------------------------------
#
# One preset: FastStrata on this PC. Its models are the four FastStrata's setup
# offers, each naming the run config setup writes for it
# (`strata-<family>-<quant>.json` in the FastStrata folder), so a model that
# was not downloaded shows as "not installed" and is never swapped to.
#
# The windows below are what setup gives a 6 GB card with 32 GB of RAM or
# more; setup writes the ones it chose for THIS PC into the top-level `models`
# of settings.json, which wins over the preset (`active_models`). Sampling is
# Qwen's own recommendation for thinking mode, with its temperature as the
# floor: a thinking model's loops get worse, not better, as it cools.

#: The address FastStrata's server is started on (`config.POLYSTRATA_PORT`).
FASTSTRATA_UPSTREAM = "http://127.0.0.1:8080"
FASTSTRATA_PRESET = "FastStrata (this PC)"

_QWEN_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "min_p": 0.05}


def _faststrata_model(model_id: str, label: str, target: str, n_ctx: int,
                      output: int) -> dict[str, Any]:
    return {
        "id": model_id,
        "label": label,
        "engine": "polystrata",
        "upstream": FASTSTRATA_UPSTREAM,
        "target": target,
        "family": "qwen3",
        "n_ctx": n_ctx,
        "context_reserve": output,
        "max_n_predict": output,
        "max_output": output,
        "sampling": dict(_QWEN_SAMPLING),
        "temperature_floor": 0.6,
    }


DEFAULT_PRESETS: dict[str, dict[str, Any]] = {
    FASTSTRATA_PRESET: {
        "upstream": FASTSTRATA_UPSTREAM,
        "port": 18000,
        "context_reserve": 16384,
        "spill_max_inline_bytes": 65536,
        "fallback_n_ctx": 65536,
        "semaphore_timeout": 7200.0,
        "stream_max_duration": 7200,
        "model_aliases": {
            "qwen3.6-35b-a3b-ud-q4_k_m": "qwen3.6-35b-a3b",
            "qwen3.6-35b-a3b-ud-q3_k_m": "qwen3.6-35b-a3b-q3",
            "ornith": "ornith-1.5-35b-a3b",
            "ornith-1.5": "ornith-1.5-35b-a3b",
        },
        "lodestar_enabled": False,   # FastStrata: off at the start, the switch turns it on
        "models": [
            _faststrata_model("qwen3.6-35b-a3b", "Qwen3.6 35B-A3B UD-Q4_K_M",
                              "strata-qwen36-ud-q4_k_m.json", 147456, 16384),
            _faststrata_model("qwen3.6-35b-a3b-q3", "Qwen3.6 35B-A3B UD-Q3_K_M",
                              "strata-qwen36-ud-q3_k_m.json", 147456, 16384),
            _faststrata_model("ornith-1.5-35b-a3b", "Ornith 1.5 35B-A3B APEX-I-QUALITY",
                              "strata-ornith-apex-i-quality.json", 65536, 8192),
            _faststrata_model("ornith-1.5-35b-a3b-compact", "Ornith 1.5 35B-A3B APEX-I-COMPACT",
                              "strata-ornith-apex-i-compact.json", 65536, 8192),
        ],
    },
}


def get_presets(file_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return configured presets from settings.json, seeded with defaults if missing."""
    settings = load_settings_from_file(file_path)
    presets = settings.get("presets")
    if isinstance(presets, dict) and presets:
        # A stored preset gains keys it has never heard of, and loses nothing.
        #
        # Without this, a settings.json written before `models` existed keeps
        # its saved copy of a preset forever, and the hot-swap rig arrives as
        # an empty Models table on the machine that most needs it -- the
        # feature ships and nobody sees it. Measured on the live file: three
        # presets, none of them carrying `models`.
        #
        # The STORED value always wins where there is one, so this can only
        # add. A key the user has edited, emptied or set to something odd is
        # left exactly as it is; only a key that is genuinely absent is filled
        # in from the default of the same name.
        merged: dict[str, dict[str, Any]] = {}
        for name, stored in presets.items():
            base = DEFAULT_PRESETS.get(name)
            if isinstance(base, dict) and isinstance(stored, dict):
                merged[name] = {**base, **stored}
            else:
                merged[name] = dict(stored)
        return merged

    return {k: dict(v) for k, v in DEFAULT_PRESETS.items()}


def active_models(file_path: Path | None = None) -> list[dict[str, Any]]:
    """The models that should be served right now, from settings.json.

    Top-level `models` is the ACTIVE configuration and the only thing
    `GatewayConfig.from_env` reads; the active preset is the fallback.

    **The fallback is the whole point of this function, and it is here rather
    than in either caller because the two callers disagreeing is what went
    wrong.** Every settings.json written before the registry existed has no
    top-level `models`, so on 2026-09-13 a freshly restarted gateway served an
    EMPTY registry: no per-model routing, no per-model limits, no hot swap, no
    vision — while the dashboard, which had its own fallback, showed four
    models. Verified against the live process: `/v1/models` listed the aliases
    and none of the registry's models.

    Nothing said so. An empty registry is silent by design — it is the correct
    state for a single-engine preset — so the feature would have looked
    installed and done nothing until somebody happened to press Save in a tab
    they had no reason to open.

    Only `models` gets this treatment. Every other key has been written to the
    top level by every version of the dashboard, so reading presets for them
    would change behaviour that is currently correct.

    The fallback reads the preset the user NAMED, never one inferred. With no
    `active_preset` in the file, `get_active_preset_name` guesses from the
    upstream and falls back to whichever built-in preset comes first — so a
    gateway configured for one machine could inherit another preset's model
    list, pointed at an address on somebody else's machine. An empty
    registry is the safe answer to "which preset is this"; a guessed one is
    not.
    """
    settings = load_settings_from_file(file_path)
    raw = settings.get("models")
    if not raw:
        named = settings.get("active_preset")
        if named:
            preset = get_presets(file_path).get(str(named)) or {}
            raw = preset.get("models")
    return [dict(m) for m in (raw or []) if isinstance(m, dict)]


def get_active_preset_name(file_path: Path | None = None) -> str:
    """Return the name of the currently active preset from settings.json."""
    settings = load_settings_from_file(file_path)
    active = settings.get("active_preset")
    presets = get_presets(file_path)
    if active and active in presets:
        return active

    if presets:
        return next(iter(presets.keys()))
    return FASTSTRATA_PRESET


def save_preset(
    name: str,
    preset_data: dict[str, Any],
    file_path: Path | None = None,
    set_active: bool = False,
) -> bool:
    """Save or update a named preset in settings.json."""
    if not name or not name.strip():
        return False
    clean_name = name.strip()
    presets = get_presets(file_path)
    presets[clean_name] = dict(preset_data)

    payload: dict[str, Any] = {"presets": presets}
    if set_active:
        payload["active_preset"] = clean_name
    return save_settings_to_file(payload, file_path)


def delete_preset(name: str, file_path: Path | None = None) -> bool:
    """Delete a named preset. Cannot delete if it's the only remaining preset."""
    presets = get_presets(file_path)
    if name not in presets or len(presets) <= 1:
        return False
    del presets[name]

    active = get_active_preset_name(file_path)
    payload: dict[str, Any] = {"presets": presets}
    if active == name:
        payload["active_preset"] = next(iter(presets.keys()))
    return save_settings_to_file(payload, file_path)


def apply_preset(name: str, file_path: Path | None = None) -> dict[str, Any] | None:
    """Apply a preset's settings as active settings in settings.json."""
    presets = get_presets(file_path)
    if name not in presets:
        return None
    preset = presets[name]
    to_apply = dict(preset)
    to_apply["active_preset"] = name
    to_apply["presets"] = presets
    if save_settings_to_file(to_apply, file_path):
        return to_apply
    return None

