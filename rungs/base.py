"""The rung contract: every rung on the ladder is a Rung.

A rung is a deployable beat/downbeat tracker with ONE public entry point:

    predict(features, **predict_options) -> {'beats': seconds[N], 'downbeats': seconds[M]}

so evaluate.py can score every rung identically and rung comparisons are always apples-to-apples.
The base class owns what is common to ALL rungs -- the deployment interface, input coercion and
validation, the Böck 2016 decorrelation, and the empty-result shape -- and leaves the model itself
to the subclass:

    coercion      accept numpy or (possibly CUDA) torch [num_frames, 2] (beat, downbeat), hand the
                  subclass float64 numpy. Validated here so no rung ever sees a malformed input.
    decorrelation max(beat - downbeat, floor) -- the standard convention (Böck, Krebs & Widmer,
                  ISMIR 2016) that converts a modern two-sigmoid frontend (beat fires on downbeats
                  too, so beat + downbeat can reach ~2) into the mutually-exclusive
                  [P(beat-not-downbeat), P(downbeat)] pair the bar-pointer observation model
                  assumes (it computes the no-beat mass as 1 - beat - downbeat). Shared here
                  because EVERY rung that emits through that observation model needs the identical
                  transform -- R0 and R1 using different decorrelations would void the
                  certificate.

Rung-specific knobs stay in the subclasses: R0's input_form/bounding exist to replicate each
published system's exact recipe; R1's threshold/correct are deployment options on our own engine.
"""
from abc import ABC, abstractmethod

import numpy as np


class Rung(ABC):
    """Base class for every rung. Subclasses implement _predict_features; predict() is final."""

    # Frontend-owned properties this rung's CONSTRUCTOR consumes (besides fps, which every rung
    # takes). Default: none -- the rung expects PROBABILITIES, and the Tracker sigmoids logit
    # frontends on the way in. A rung that instead does its own form handling (R0 replicates each
    # published recipe bit-exactly, bounding convention included) names the frontend properties it
    # takes, and the Tracker passes them through instead of converting. This is what lets the
    # Tracker stay generic: no per-rung special cases, the class declares its own contract.
    FRONTEND_KWARGS: tuple = ()

    # How many channels this rung's predict() consumes. The HMM family reads exactly [T, 2]
    # (beat, downbeat) activations -- the frontend's FINAL layer. A latent-variable rung that
    # conditions on rich penultimate features instead declares its own count (e.g. 512) or None
    # for "any". The Tracker checks this against the frontend's output mode at construction, so a
    # frontend cut at the wrong depth fails loudly instead of predicting on garbage.
    INPUT_CHANNELS = 2      # int, or None for "any"

    def predict(self, features, **predict_options) -> dict:
        """features: [num_frames, INPUT_CHANNELS]. The common deployment interface.

        For the HMM family (INPUT_CHANNELS=2) that is (beat, downbeat) activations; a
        latent-variable rung consumes whatever width it declared. predict_options are forwarded to
        the subclass (e.g. R1's threshold/correct overrides); rungs without per-call options
        accept none.
        """
        features = self._coerce_features(features)
        return self._predict_features(features, **predict_options)

    @abstractmethod
    def _predict_features(self, features: np.ndarray, **predict_options) -> dict:
        """features is a validated float64 numpy [num_frames, INPUT_CHANNELS]. Return the events
        dict."""

    @classmethod
    def _coerce_features(cls, features) -> np.ndarray:
        """Accept a numpy array or a (possibly CUDA) torch tensor; validate the shape against the
        class's declared INPUT_CHANNELS (None = any channel count)."""
        if hasattr(features, "detach"):
            features = features.detach().cpu().numpy()
        features = np.ascontiguousarray(np.asarray(features, dtype=np.float64))
        if features.ndim != 2 or (cls.INPUT_CHANNELS is not None
                                  and features.shape[1] != cls.INPUT_CHANNELS):
            raise ValueError(f"expected [num_frames, {cls.INPUT_CHANNELS or 'any'}] features, "
                             f"got {features.shape}")
        return features

    def _bound(self, beat_activation, downbeat_activation):
        """Bound activations off exact 0/1 and return the decorrelation floor to pair with them.

        For activation-consuming rungs only (INPUT_CHANNELS=2): reads self.bounding ("clip",
        "squeeze" or "none" -- each published system's convention, see rungs/r0) and self.eps,
        which those rungs set in their constructors.
        """
        if self.INPUT_CHANNELS != 2:
            raise TypeError(f"{type(self).__name__} does not consume [T, 2] activations; "
                            f"_bound does not apply")
        eps = self.eps

        if self.bounding == "clip":
            beat_activation = np.clip(beat_activation, eps, 1 - eps)
            downbeat_activation = np.clip(downbeat_activation, eps, 1 - eps)
            decorrelation_floor = eps
        elif self.bounding == "squeeze":
            beat_activation = beat_activation * (1 - eps) + eps / 2
            downbeat_activation = downbeat_activation * (1 - eps) + eps / 2
            decorrelation_floor = eps / 2
        else:
            # none (Beat Transformer); tiny floor for safety
            decorrelation_floor = 1e-12

        return beat_activation, downbeat_activation, decorrelation_floor

    @staticmethod
    def _decorrelate(beat_activation: np.ndarray, downbeat_activation: np.ndarray,
                     floor: float) -> np.ndarray:
        """Böck 2016 decorrelation -> [num_frames, 2] (beat-not-downbeat, downbeat).

        Assumes the channels are already bounded off exact 0/1 (each rung bounds per its recipe);
        floor keeps the subtraction positive so the observation model's logs stay finite.
        """
        beat_not_downbeat_activation = np.maximum(beat_activation - downbeat_activation, floor)
        return np.stack([beat_not_downbeat_activation, downbeat_activation], axis=-1)

    @staticmethod
    def _empty_events() -> dict:
        return {"beats": np.array([]), "downbeats": np.array([])}
