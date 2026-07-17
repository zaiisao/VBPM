"""The rung contract: every rung on the ladder is a Rung.

A rung is a deployable beat/downbeat tracker with ONE public entry point:

    decode(activations, **decode_options) -> {'beats': seconds[N], 'downbeats': seconds[M]}

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
                  transform -- R0 and R1 decoding different decorrelations would void the
                  certificate.

Rung-specific knobs stay in the subclasses: R0's input_form/bounding exist to replicate each
published system's exact recipe; R1's threshold/correct are deployment options on our own engine.
"""
from abc import ABC, abstractmethod

import numpy as np


class Rung(ABC):
    """Base class for every rung. Subclasses implement _decode_activations; decode() is final."""

    # Frontend-owned properties this rung's CONSTRUCTOR consumes (besides fps, which every rung
    # takes). Default: none -- the rung expects PROBABILITIES, and the Tracker sigmoids logit
    # frontends on the way in. A rung that instead does its own form handling (R0 replicates each
    # published recipe bit-exactly, bounding convention included) names the frontend properties it
    # takes, and the Tracker passes them through instead of converting. This is what lets the
    # Tracker stay generic: no per-rung special cases, the class declares its own contract.
    FRONTEND_KWARGS: tuple = ()

    def decode(self, activations, **decode_options) -> dict:
        """activations: [num_frames, 2] (beat, downbeat). The common deployment interface.

        decode_options are forwarded to the subclass (e.g. R1's threshold/correct overrides);
        rungs without per-call options accept none.
        """
        activations = self._coerce_activations(activations)
        return self._decode_activations(activations, **decode_options)

    @abstractmethod
    def _decode_activations(self, activations: np.ndarray, **decode_options) -> dict:
        """activations is a validated float64 numpy [num_frames, 2]. Return the events dict."""

    @staticmethod
    def _coerce_activations(activations) -> np.ndarray:
        """Accept a numpy array or a (possibly CUDA) torch tensor; validate the [T, 2] shape."""
        if hasattr(activations, "detach"):
            activations = activations.detach().cpu().numpy()
        activations = np.ascontiguousarray(np.asarray(activations, dtype=np.float64))
        if activations.ndim != 2 or activations.shape[1] < 2:
            raise ValueError(f"expected [num_frames, 2] activations (beat, downbeat), "
                             f"got {activations.shape}")
        return activations

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
