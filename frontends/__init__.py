"""Feature extractors (frontends) + the glue that pairs any frontend with any decoder.

One script per frontend (beat_this.py, later mert.py, ...). A frontend wraps the official upstream
model behind a two-property interface -- WHAT it emits (`fps`, `activation_form`) and HOW to get it
(`activations(signal, sample_rate) -> [num_frames, 2]`). The decoders (the rungs) are already
interchangeable behind `decode(activations) -> {'beats', 'downbeats'}`, so the Tracker below is the
whole plug-and-play story: it wires the frontend's fps into the decoder's constructor (a wrong fps
mis-scales every DBN tempo -- fps is a property OF THE ACTIVATIONS, never a constant) and converts
the activation form once.

Deliberately simple (a resurrected, slimmed version of the archived
data/feature_extractor.py + configs/frontends/*.yaml system): properties live on the wrapper class,
not in YAML, until we have enough frontends to need config files again.
"""
import numpy as np

from rungs.r0_madmom_dbn import MadmomDBN
from rungs.r1_handcrafted_hmm import HandcraftedBarPointerHMM


class Frontend:
    """Interface. A frontend turns audio into [num_frames, 2] (beat, downbeat) activations."""

    name: str = "?"
    fps: float = None               # the activation frame rate -- decoders are built around this
    activation_form: str = "prob"   # "prob" or "logit" -- what activations() returns
    bounding: str = "clip"          # the frontend's PUBLISHED bounding convention (see rungs/r0);
                                    # wired into the DBN so our pipeline == the published one

    def activations(self, signal, sample_rate: int):
        """[num_samples] mono audio -> [num_frames, 2] (beat, downbeat) in `activation_form`."""
        raise NotImplementedError


DECODERS = {
    "madmom_dbn": MadmomDBN,                    # R0: the official madmom processor
    "bar_pointer_hmm": HandcraftedBarPointerHMM,  # R1: same model on our engine (certified == R0)
}


def build_frontend(name: str, **kwargs) -> Frontend:
    """Factory by name; imports lazily so heavy frontend deps load only when used."""
    if name == "beat_this":
        from frontends.beat_this import BeatThisFrontend
        return BeatThisFrontend(**kwargs)
    raise KeyError(f"unknown frontend {name!r} (have: ['beat_this'])")


class Tracker:
    """frontend x decoder, wired correctly: Tracker(frontend, 'madmom_dbn', threshold=0.05).

    Every decoder kwarg (min_bpm, max_bpm, beats_per_bar, transition_lambda, observation_lambda,
    num_tempi, threshold, correct, ...) passes straight through to the decoder's constructor;
    `fps` and `input_form` come from the frontend and cannot be passed (that is the point).
    """

    def __init__(self, frontend: Frontend, decoder: str = "madmom_dbn", **decoder_kwargs):
        if decoder not in DECODERS:
            raise KeyError(f"unknown decoder {decoder!r} (have: {sorted(DECODERS)})")
        for reserved in ("fps", "input_form"):
            if reserved in decoder_kwargs:
                raise ValueError(f"{reserved!r} comes from the frontend, don't pass it")
        self.frontend = frontend
        # R0 handles logits natively; R1 (and later rungs) take probabilities, so the Tracker
        # sigmoids once on the way in. (Measured equivalent: decorr+clip == decorr+squeeze to 4
        # decimals -- the form conversion is not a modeling choice.)
        if decoder == "madmom_dbn":
            decoder_kwargs["input_form"] = frontend.activation_form
            decoder_kwargs.setdefault("bounding", frontend.bounding)
            self._sigmoid = False
        else:
            self._sigmoid = frontend.activation_form == "logit"
        self.decoder = DECODERS[decoder](fps=frontend.fps, **decoder_kwargs)

    def track(self, signal, sample_rate: int) -> dict:
        """audio -> {'beats': seconds, 'downbeats': seconds}."""
        activations = self.frontend.activations(signal, sample_rate)
        if self._sigmoid:
            activations = 1.0 / (1.0 + np.exp(-np.asarray(activations, dtype=np.float64)))
        return self.decoder.decode(activations)
