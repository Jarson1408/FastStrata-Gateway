"""What it takes to hand the card to another model, and how a refusal reads.

FastStrata serves one model at a time from one card: asking for another model
stops the running engine and starts the other one (`_ensure_polystrata_server`
in app.py). That is minutes of work and it frees every byte of VRAM and RAM the
old model held, so it must not start for a model that cannot come up.

**A swap to a model that is not on disk is refused before anything is
stopped.** The run config FastStrata's setup wrote, and the GGUF it names, are
checked first. Without the check a typo in OpenCode's model list, or a preset
that was declared but never downloaded, would end the working model and leave
the card empty -- and the next turn, for the model that WAS fine, would pay a
cold start.

The refusal is an `EngineError` with a sentence in it. The caller turns that
into CONTENT, never an error object: a failed swap ends the turn, and an error
object ends the session -- OpenCode abandons the agent loop on a response with
no `choices`.
"""
from __future__ import annotations

from gateway.model_registry import ModelSpec


class EngineError(RuntimeError):
    """The card could not be handed over, with a sentence saying why."""


def check_installed(spec: ModelSpec, polystrata_dir: str | None = None) -> None:
    """Raise `EngineError` naming what is missing when `spec` cannot start."""
    missing = spec.missing_files(polystrata_dir)
    if missing:
        raise EngineError(
            f"Model '{spec.display_name}' is not installed on this PC: "
            f"{', '.join(missing)} not found. Run FastStrata's setup to "
            "download it, or pick another model. The model that was running "
            "is still loaded.")
