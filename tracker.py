"""The glue that pairs any frontend with any bar-pointer model (BPM).

Every rung of the ladder is a bar-pointer model -- whatever its machinery (handcrafted HMM,
learned factors, CVAE), it models the latent bar-pointer state (phase, tempo, meter) and reads
beats off its trajectory. That, not the inference engine, is the component-role name; "rung"
stays the ladder-position name. Spelled out as bar_pointer in code because the acronym BPM
collides with beats-per-minute (min_bpm/max_bpm below).

Sits above both packages: frontends/ knows nothing about bar-pointer models, rungs/ knows
nothing about audio. This is the composition surface: frontends resolve by dotted module path
(`frontends.beat_this` -- no registry to edit when adding one), bar-pointer models by the
BAR_POINTERS registry (the ladder is small and central), so a config can say
`frontend: frontends.beat_this, bar_pointer: 2016_dbn` and this file resolves both. The Tracker is the whole plug-and-play story: it wires the frontend's fps into the model's
constructor (a wrong fps mis-scales every tempo grid -- fps is a property OF THE ACTIVATIONS,
never a constant) and converts the activation form once.
"""
import importlib

import numpy as np

from frontends import Frontend
from rungs.r0_madmom_dbn import MadmomDBN
from rungs.r1_2016_dbn import DBN2016

BAR_POINTERS = {
    "madmom_dbn": MadmomDBN,                    # R0: the official madmom processor
    "2016_dbn": DBN2016,                        # R1: same model on our engine (certified == R0)
}


def build_frontend(module_path: str, **kwargs) -> Frontend:
    """Instantiate the Frontend subclass defined in `module_path` (e.g. "frontends.beat_this").

    Dotted-path loading instead of a hand-maintained registry: adding a frontend = adding a module
    that defines exactly one Frontend subclass; nothing here changes. The import happens at call
    time, so heavy frontend deps (torch, checkpoints) load only when that frontend is used.
    """
    module = importlib.import_module(module_path)
    frontend_classes = [obj for obj in vars(module).values()
                        if isinstance(obj, type) and issubclass(obj, Frontend)
                        and obj is not Frontend and obj.__module__ == module.__name__]
    if len(frontend_classes) != 1:
        raise ImportError(f"{module_path!r} must define exactly one Frontend subclass, "
                          f"found {[cls.__name__ for cls in frontend_classes]}")
    return frontend_classes[0](**kwargs)


class Tracker:
    """frontend x bar-pointer model, wired correctly: Tracker(frontend, 'madmom_dbn', threshold=0.05).

    Every model kwarg (min_bpm, max_bpm, beats_per_bar, transition_lambda, observation_lambda,
    num_tempi, threshold, correct, ...) passes straight through to the model's constructor;
    `fps` and the class's declared FRONTEND_KWARGS (e.g. R0's input_form/bounding) come from the
    frontend and cannot be passed (that is the point).
    """

    def __init__(self, frontend: Frontend, bar_pointer: str = "madmom_dbn", **model_kwargs):
        if bar_pointer not in BAR_POINTERS:
            raise KeyError(f"unknown bar-pointer model {bar_pointer!r} (have: {sorted(BAR_POINTERS)})")
        model_class = BAR_POINTERS[bar_pointer]

        for reserved in ("fps", *model_class.FRONTEND_KWARGS):
            if reserved in model_kwargs:
                raise ValueError(f"{reserved!r} comes from the frontend, don't pass it")

        # Channel handshake: the frontend's output mode must emit what the rung consumes
        # (Rung.INPUT_CHANNELS; None = any). Catches a frontend cut at the wrong depth -- e.g.
        # penultimate [T, 512] features into an HMM rung that reads [T, 2] activations.
        required_channels = model_class.INPUT_CHANNELS
        if required_channels is not None and frontend.num_channels != required_channels:
            raise ValueError(
                f"{bar_pointer!r} consumes [T, {required_channels}] input, but frontend "
                f"{frontend.name!r} in output mode {frontend.output!r} emits "
                f"[T, {frontend.num_channels}]")

        self.frontend = frontend
        # Form conversion, driven by the class's declared contract (Rung.FRONTEND_KWARGS): a model
        # that names "input_form" handles the frontend's native form itself (R0, replicating each
        # published recipe), so its frontend kwargs pass through; every other model expects
        # probabilities, so the Tracker sigmoids logit frontends once on the way in. (Measured
        # equivalent: decorr+clip == decorr+squeeze to 4 decimals -- not a modeling choice.)
        frontend_properties = {
            "input_form": frontend.ACTIVATION_FORM,
            "bounding": frontend.BOUNDING
        }

        for name in model_class.FRONTEND_KWARGS:
            model_kwargs[name] = frontend_properties[name]

        # activation_form (and the sigmoid) only applies to the "activations" output mode --
        # rich features are not probabilities and pass through untouched.
        self._should_convert_to_probabilities = (
            frontend.output == "activations"
            and "input_form" not in model_class.FRONTEND_KWARGS
            and frontend.ACTIVATION_FORM == "logit"
        )

        self.bar_pointer = model_class(fps=frontend.fps, **model_kwargs)

    def track(self, signal, sample_rate: int) -> dict:
        """audio -> {'beats': seconds, 'downbeats': seconds}."""
        features = self.frontend.get_features(signal, sample_rate)
        if self._should_convert_to_probabilities:
            features = 1.0 / (1.0 + np.exp(-np.asarray(features, dtype=np.float64)))
        return self.bar_pointer.decode(features)


def build_tracker_from_config(config: dict) -> Tracker:
    """{'frontend': {'name': ..., **kwargs}, 'bar_pointer': {'name': ..., **kwargs}} -> Tracker.

    The shared config->Tracker path for the CLI (track.py), training and evaluation. Takes a plain
    dict -- reading YAML (or wherever the dict comes from) is the caller's business.

    frontend.output and bar_pointer.input must BOTH be declared and must match. The redundancy is
    deliberate: which layer the frontend is cut at (final [T, 2] activations vs penultimate rich
    features) and what the rung consumes are two decisions the config author is forced to make
    together, so swapping one component can never silently feed a rung the wrong depth. (The
    Tracker additionally checks the channel count against the rung class's INPUT_CHANNELS.)
    """
    frontend_config = dict(config["frontend"])
    bar_pointer_config = dict(config["bar_pointer"])

    frontend_output = frontend_config.get("output")
    bar_pointer_input = bar_pointer_config.pop("input", None)   # config-only key, not a model kwarg

    if frontend_output is None or bar_pointer_input is None:
        raise KeyError("declare BOTH frontend.output and bar_pointer.input in the config "
                       "(matching values) -- the pairing must be chosen deliberately")

    if frontend_output != bar_pointer_input:
        raise ValueError(f"frontend.output {frontend_output!r} != bar_pointer.input "
                         f"{bar_pointer_input!r} -- these must be matched deliberately")

    frontend = build_frontend(frontend_config.pop("name"), **frontend_config)
    tracker = Tracker(frontend, bar_pointer_config.pop("name"), **bar_pointer_config)

    return tracker
