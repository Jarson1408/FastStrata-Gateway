"""What a preset serves, what runs each one, and on what terms.

A preset names a flat `model_aliases` map and a `models` list: specs, each
naming its engine, its address and its own settings. In this build the engine
is FastStrata (`polystrata`: its `serve/server.py`, started and stopped by the
gateway, one model on the card at a time) or `remote` (an OpenAI-compatible
server somebody else runs). Three properties are worth stating, because each
one is a decision:

- **The id is what the client asks for**, and it is the only name OpenCode ever
  sees. `target` is what the ENGINE calls the same model -- the run config
  FastStrata's setup wrote (`strata-qwen36-ud-q4_k_m.json`) -- and the two are
  separate because the engine's name is an implementation detail that changes
  when the quant does.
- **`model_aliases` still works and still wins nothing.** A registry entry is
  resolved first, an alias second; a name in neither is passed through
  untouched.
- **A spec may describe a model that is not on disk.** Every FastStrata preset
  is declared before anything is downloaded, on purpose: the GUI can then show
  it as a row that says "not installed", and the supervisor refuses to swap the
  card away from a working model for a file that is not there. `installed()` is
  the predicate, and it answers about the FILES, never about the process.

Pure: nothing here starts a process, opens a socket or reads settings.json.
That is what lets the GUI, the gateway and the tests all ask the same object.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

# The engines a spec may name. An unknown engine is a configuration error
# rather than a default, because every default here is a guess about which
# process owns the card.
ENGINE_REMOTE = "remote"
ENGINE_POLYSTRATA = "polystrata"

#: Engines whose process this gateway starts and stops itself. `remote` is
#: somebody else's machine, reachable or not, and nothing here may try to
#: manage it.
LOCAL_ENGINES = frozenset({ENGINE_POLYSTRATA})

KNOWN_ENGINES = frozenset({ENGINE_REMOTE, ENGINE_POLYSTRATA})

#: The run config a `polystrata` spec without a `target` starts.
DEFAULT_POLYSTRATA_CONFIG = "strata-qwen36-ud-q4_k_m.json"


def default_polystrata_dir() -> str:
    """The FastStrata folder when settings.json names none.

    The installer puts this gateway in `<FastStrata>/gateway/`, so the folder
    above the gateway's own is FastStrata's when it holds `serve/server.py`.
    Anywhere else (a checkout of the gateway on its own) there is no folder to
    guess, and the answer is that parent anyway: `installed()` then reports the
    run configs as missing, which is the truth, instead of naming a path that
    belongs to one machine.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ModelSpecError(ValueError):
    """A spec that cannot be served as written."""


@dataclass(frozen=True)
class ModelSpec:
    """One model, the engine that serves it, and the settings that are its own.

    Frozen because a spec is read by the request path, the supervisor and the
    GUI within one turn, and a value that three readers can mutate is the kind
    of shared state this project keeps finding in incident reports.
    """

    id: str
    engine: str
    #: Full base URL of the engine serving THIS model. Two models on two
    #: engines have two addresses, which is the whole reason a preset can no
    #: longer get by with a single `upstream`.
    upstream: str
    #: What the engine itself calls the model: FastStrata's run config (a
    #: name in its folder, or a full path), whatever the remote answers to
    #: otherwise.
    target: str = ""
    #: Human label, shown in the GUI and exported to OpenCode.
    label: str = ""
    #: Parser family. Declared rather than sniffed from the name: the sniffer
    #: reads digits out of a filename and gets `qwen3` for anything with "27b"
    #: in it, which is right today and silently wrong the first time a
    #: differently-named model arrives. Empty means "sniff it".
    family: str = ""
    #: The context window this model is configured for -- the run config's
    #: `--max-context`, which FastStrata's setup chose for this PC. It is also
    #: what gets exported to OpenCode as the client-side limit, so a wrong
    #: value here is a client that sends prompts the engine cannot take.
    n_ctx: int = 0
    #: Advertised output cap for the client's own accounting.
    max_output: int = 32768
    #: Whether this model accepts images. False means image parts are dropped
    #: from the conversation exactly as they always were.
    vision: bool = False
    #: Whether the model is asked to reason before answering, for a family
    #: whose chat format makes that a SWITCH rather than a habit. Gemma 4 is
    #: one: `<|think|>` in the system turn turns reasoning on, and without it
    #: the generation prompt carries an EMPTY thought channel and the model
    #: answers directly. None means "the family's own default" (off for
    #: Gemma 4, which is what its chat template does when nobody asks), and
    #: is what every Qwen entry carries -- Qwen reasons by habit and this
    #: switch does not reach it. Two ids naming the same `target` with
    #: different values here are one set of weights and no hot swap.
    thinking: bool | None = None
    #: How tools are described to the model and how its calls are read back.
    #: Empty means the family's default: Gemma 4 gets its NATIVE protocol
    #: (`<|tool>` declarations, `<|tool_call>call:NAME{...}`), everything else
    #: the Qwen XML dialect this gateway has always injected. "qwen_xml" on a
    #: Gemma entry is the A/B switch back to what the 26B was measured on.
    tool_dialect: str = ""
    #: Reasoning strength, for a family whose prompt format takes one --
    #: Muse Glimmer writes `Reasoning strength: <level>.` into the system turn
    #: (low / medium / high / xhigh; "" = the template's default, high). Like
    #: `thinking`, two ids on one `target` with different values cost no swap.
    reasoning: str = ""

    # -- Limits that scale with the window -----------------------------------
    #
    # 0 means "use the preset's value", which is what keeps a rig whose models
    # all have the same shape down to a single set of numbers. They are here
    # at all because a value that is right for a 229k window is WRONG for a
    # 32k one, and that is a matter of arithmetic rather than of taste:
    #
    #   context_reserve 16384 is 7% of a 229k window and HALF of a 32k one --
    #     half the window held back for output a model of this class will
    #     never produce.
    #   spill_max_inline_bytes 65536 is about 21k tokens: comfortably inline
    #     at 229k, two thirds of the whole context at 32k.
    #
    # The prune limit itself is `n_ctx - context_reserve`, so a model whose
    # window is declared here and whose reserve is not gets the preset's
    # reserve against its own window, which is the sensible default.
    context_reserve: int = 0
    max_n_predict: int = 0
    spill_max_inline_bytes: int = 0

    # -- Sampling ------------------------------------------------------------
    #
    # `sampling_preset` names one of SAMPLING_PRESETS; `sampling` overrides
    # individual parameters on top of it, and is the half that can live in
    # settings.json -- a preset is code, so without this a model needing a
    # different temperature would need a release.
    #
    # An explicit value here beats what the CLIENT sends, which is the one
    # place this gateway inverts that order. The reason is the same one the
    # model aliases exist for: OpenCode hardcodes temperature=0.55 for any
    # model whose id contains "qwen", which is a guess keyed on a NAME, while
    # a value in this field is a decision somebody wrote down for this model.
    # Only the keys actually declared here are taken; everything else follows
    # the old chain.
    sampling_preset: str = ""
    sampling: dict[str, Any] = field(default_factory=dict)
    #: The lowest temperature adaptive sampling may take this model to, 0 for
    #: none. For a thinking model set it to the model's own temperature: its
    #: loops get worse, not better, as the temperature falls.
    temperature_floor: float = 0.0

    # -- Bookkeeping -----------------------------------------------------------
    notes: str = ""

    # -- Derived ---------------------------------------------------------------

    @property
    def is_local(self) -> bool:
        """True when this gateway owns the lifetime of the serving process."""
        return self.engine in LOCAL_ENGINES

    @property
    def display_name(self) -> str:
        return self.label or self.id

    def config_path(self, polystrata_dir: str | None = None) -> str:
        """The FastStrata run config this spec starts, as a full path."""
        target = self.target or DEFAULT_POLYSTRATA_CONFIG
        if os.path.isabs(target):
            return target
        return os.path.join(polystrata_dir or default_polystrata_dir(), target)

    def installed(self, polystrata_dir: str | None = None) -> bool:
        """Whether the files this spec needs are on disk.

        Answers about FILES only. An engine that is not running is not
        "uninstalled" -- it is the normal state of a model that does not
        currently hold the card, and conflating the two would make the GUI
        report a healthy hot-swap rig as half broken.

        A remote spec has no files to check and is installed by definition.
        """
        return not self.missing_files(polystrata_dir)

    def missing_files(self, polystrata_dir: str | None = None) -> list[str]:
        """The paths `installed()` could not find, for a message worth reading.

        For FastStrata: the run config, then the model file its `--model`
        names. A config that cannot be read is reported as missing too, since
        the engine could not start from it either.
        """
        if self.engine != ENGINE_POLYSTRATA:
            return []
        cfg_path = self.config_path(polystrata_dir)
        if not os.path.isfile(cfg_path):
            return [cfg_path]
        try:
            with open(cfg_path, "r", encoding="utf-8-sig") as f:
                args = json.load(f).get("args", [])
        except (OSError, ValueError, AttributeError):
            return [cfg_path]
        if isinstance(args, list) and "--model" in args:
            at = args.index("--model") + 1
            if at < len(args) and not os.path.isfile(str(args[at])):
                return [str(args[at])]
        return []

    # -- Serialisation ---------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelSpec:
        """Build a spec from settings.json, refusing what cannot be served.

        Validation is deliberately narrow: it rejects only what makes the spec
        unusable no matter what happens later (no id, an engine nothing can
        run, no address). It does NOT check
        that files exist -- that is `installed()`, and it must stay a separate
        question so a preset can declare a model before it is downloaded.
        """
        if not isinstance(data, dict):
            raise ModelSpecError(f"model spec must be an object, got {type(data).__name__}")

        model_id = str(data.get("id") or "").strip()
        if not model_id:
            raise ModelSpecError("model spec has no 'id'")

        engine = str(data.get("engine") or "").strip()
        if engine not in KNOWN_ENGINES:
            raise ModelSpecError(
                f"model '{model_id}' names engine '{engine}', which is not one of "
                f"{sorted(KNOWN_ENGINES)}")

        upstream = str(data.get("upstream") or "").strip()
        if not upstream:
            raise ModelSpecError(f"model '{model_id}' has no 'upstream' address")

        sampling = data.get("sampling") or {}
        if not isinstance(sampling, dict):
            raise ModelSpecError(
                f"model '{model_id}' has 'sampling' as {type(sampling).__name__}; "
                "it must be an object of parameter names to values")

        try:
            floor = float(data.get("temperature_floor") or 0.0)
        except (TypeError, ValueError):
            raise ModelSpecError(
                f"model '{model_id}' has 'temperature_floor' "
                f"{data.get('temperature_floor')!r}; it must be a number") from None
        if not 0.0 <= floor <= 2.0:
            raise ModelSpecError(
                f"model '{model_id}' has 'temperature_floor' {floor}; it must be "
                "between 0 (no floor) and 2")

        reserve = int(data.get("context_reserve") or 0)
        window = int(data.get("n_ctx") or 0)
        if window and reserve >= window:
            # Caught here rather than at request time, where it produces a
            # prune limit of zero or less and a turn with no history at all.
            raise ModelSpecError(
                f"model '{model_id}' reserves {reserve} tokens of a {window}-token "
                "window, which leaves nothing for the prompt")

        return cls(
            id=model_id,
            engine=engine,
            upstream=upstream,
            target=str(data.get("target") or "").strip(),
            label=str(data.get("label") or "").strip(),
            family=str(data.get("family") or "").strip(),
            n_ctx=int(data.get("n_ctx") or 0),
            max_output=int(data.get("max_output") or 32768),
            vision=bool(data.get("vision", False)),
            thinking=(None if data.get("thinking") is None
                      else bool(data.get("thinking"))),
            tool_dialect=str(data.get("tool_dialect") or "").strip(),
            reasoning=str(data.get("reasoning") or "").strip().lower(),
            context_reserve=reserve,
            max_n_predict=int(data.get("max_n_predict") or 0),
            spill_max_inline_bytes=int(data.get("spill_max_inline_bytes") or 0),
            sampling_preset=str(data.get("sampling_preset") or "").strip(),
            sampling=dict(sampling),
            temperature_floor=floor,
            notes=str(data.get("notes") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """The form settings.json stores, with defaults left out.

        Round-trips: `from_dict(spec.to_dict()) == spec`. That is what the GUI
        relies on when it edits one row of a preset and writes the rest back
        untouched.
        """
        data: dict[str, Any] = {
            "id": self.id,
            "engine": self.engine,
            "upstream": self.upstream,
        }
        if self.target:
            data["target"] = self.target
        if self.label:
            data["label"] = self.label
        if self.family:
            data["family"] = self.family
        if self.n_ctx:
            data["n_ctx"] = self.n_ctx
        if self.max_output != 32768:
            data["max_output"] = self.max_output
        if self.vision:
            data["vision"] = True
        if self.thinking is not None:
            data["thinking"] = self.thinking
        if self.tool_dialect:
            data["tool_dialect"] = self.tool_dialect
        if self.reasoning:
            data["reasoning"] = self.reasoning
        for key in ("context_reserve", "max_n_predict", "spill_max_inline_bytes"):
            value = getattr(self, key)
            if value:
                data[key] = value
        if self.sampling_preset:
            data["sampling_preset"] = self.sampling_preset
        if self.sampling:
            data["sampling"] = dict(self.sampling)
        if self.temperature_floor:
            data["temperature_floor"] = self.temperature_floor
        if self.notes:
            data["notes"] = self.notes
        return data


class ModelRegistry:
    """The specs of one preset, indexed by every name a client might use.

    Empty is the normal state for a preset that never gained a `models` list --
    any settings.json written before this existed -- and an empty registry
    resolves nothing, which leaves the old alias path exactly as it was.
    """

    __slots__ = ("_specs", "_by_name", "_errors")

    def __init__(self, specs: Iterable[ModelSpec] = (), errors: Iterable[str] = ()):
        self._specs: list[ModelSpec] = list(specs)
        self._errors: list[str] = list(errors)
        self._by_name: dict[str, ModelSpec] = {}
        # Ids first, in order, so a later spec cannot take an earlier one's
        # name; targets afterwards and only where the name is still free. A
        # client that asks by the engine's own name gets served, and a spec
        # whose TARGET collides with another spec's ID never shadows it.
        for spec in self._specs:
            self._by_name.setdefault(spec.id.lower(), spec)
        for spec in self._specs:
            if spec.target:
                self._by_name.setdefault(spec.target.lower(), spec)

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self):
        return iter(self._specs)

    @property
    def specs(self) -> list[ModelSpec]:
        return list(self._specs)

    @property
    def errors(self) -> list[str]:
        """Specs that were dropped, and why.

        Kept rather than raised: one malformed row in settings.json must not
        take the whole registry down with it, or a typo in an unused model
        stops the rig from serving the model that is fine. The GUI shows
        these; the gateway logs them once at startup.
        """
        return list(self._errors)

    def resolve(self, name: str | None) -> ModelSpec | None:
        """The spec a client asked for, by id or by the engine's own name."""
        if not name:
            return None
        return self._by_name.get(str(name).strip().lower())

    def vision_models(self) -> list[ModelSpec]:
        return [s for s in self._specs if s.vision]

    def engines(self) -> list[str]:
        """Distinct engines this preset uses, in declaration order."""
        seen: list[str] = []
        for spec in self._specs:
            if spec.engine not in seen:
                seen.append(spec.engine)
        return seen

    # -- Construction ----------------------------------------------------------

    @classmethod
    def from_preset(cls, preset: dict[str, Any] | None) -> ModelRegistry:
        """Read the `models` list of one preset, keeping what is usable."""
        if not isinstance(preset, dict):
            return cls()
        return cls.from_list(preset.get("models"))

    @classmethod
    def from_list(cls, raw: Any) -> ModelRegistry:
        if not isinstance(raw, list):
            return cls()
        specs: list[ModelSpec] = []
        errors: list[str] = []
        seen: set[str] = set()
        for entry in raw:
            try:
                spec = ModelSpec.from_dict(entry)
            except (ModelSpecError, TypeError, ValueError) as exc:
                errors.append(str(exc))
                continue
            key = spec.id.lower()
            if key in seen:
                errors.append(f"model '{spec.id}' is declared more than once; "
                              "the first declaration is the one that is served")
                continue
            seen.add(key)
            specs.append(spec)
        return cls(specs, errors)

    def to_list(self) -> list[dict[str, Any]]:
        """What settings.json stores. Round-trips through `from_list`."""
        return [s.to_dict() for s in self._specs]
