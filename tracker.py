"""The glue that pairs any frontend with any bar-pointer model (BPM).

Every rung of the ladder is a bar-pointer model -- whatever its machinery (handcrafted HMM,
learned factors, CVAE), it models the latent bar-pointer state (phase, tempo, meter) and reads
beats off its trajectory. That, not the inference engine, is the component-role name; "rung"
stays the ladder-position name. Spelled out as bar_pointer in code because the acronym BPM
collides with beats-per-minute (min_bpm/max_bpm below).

Sits above both packages: frontends/ knows nothing about bar-pointer models, rungs/ knows
nothing about audio. This is the composition surface -- both by-name registries live here, so a
config or CLI can say `frontend: beat_this, bar_pointer: 2016_dbn` and this file resolves
both. The Tracker is the whole plug-and-play story: it wires the frontend's fps into the model's
constructor (a wrong fps mis-scales every tempo grid -- fps is a property OF THE ACTIVATIONS,
never a constant) and converts the activation form once.
"""
import numpy as np

from frontends import Frontend
from rungs.r0_madmom_dbn import MadmomDBN
from rungs.r1_2016_dbn import DBN2016

BAR_POINTERS = {
    "madmom_dbn": MadmomDBN,                    # R0: the official madmom processor
    "2016_dbn": DBN2016,                        # R1: same model on our engine (certified == R0)
}


def build_frontend(name: str, **kwargs) -> Frontend:
    """Factory by name; imports lazily so heavy frontend deps load only when used."""
    if name == "beat_this":
        from frontends.beat_this import BeatThisFrontend
        return BeatThisFrontend(**kwargs)
    raise KeyError(f"unknown frontend {name!r} (have: ['beat_this'])")


class Tracker:
    """frontend x bar-pointer model, wired correctly: Tracker(frontend, 'madmom_dbn', threshold=0.05).

    Every model kwarg (min_bpm, max_bpm, beats_per_bar, transition_lambda, observation_lambda,
    num_tempi, threshold, correct, ...) passes straight through to the model's constructor;
    `fps` and `input_form` come from the frontend and cannot be passed (that is the point).
    """

    def __init__(self, frontend: Frontend, bar_pointer: str = "madmom_dbn", **model_kwargs):
        if bar_pointer not in BAR_POINTERS:
            raise KeyError(f"unknown bar-pointer model {bar_pointer!r} (have: {sorted(BAR_POINTERS)})")

        for reserved in ("fps", "input_form"):
            if reserved in model_kwargs:
                raise ValueError(f"{reserved!r} comes from the frontend, don't pass it")

        self.frontend = frontend
        # R0 handles logits natively; R1 (and later rungs) take probabilities, so the Tracker
        # sigmoids once on the way in. (Measured equivalent: decorr+clip == decorr+squeeze to 4
        # decimals -- the form conversion is not a modeling choice.)
        if bar_pointer == "madmom_dbn":
            model_kwargs["input_form"] = frontend.activation_form
            model_kwargs.setdefault("bounding", frontend.bounding)
            self._sigmoid = False
        else:
            self._sigmoid = frontend.activation_form == "logit"
        self.bar_pointer = BAR_POINTERS[bar_pointer](fps=frontend.fps, **model_kwargs)

    def track(self, signal, sample_rate: int) -> dict:
        """audio -> {'beats': seconds, 'downbeats': seconds}."""
        activations = self.frontend.activations(signal, sample_rate)
        if self._sigmoid:
            activations = 1.0 / (1.0 + np.exp(-np.asarray(activations, dtype=np.float64)))
        return self.bar_pointer.decode(activations)
