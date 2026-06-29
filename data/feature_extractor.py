"""Modular feature-extractor interface.

A feature extractor maps audio -> a per-frame feature sequence [num_frames, feature_dim] that the
bar-pointer DVAE consumes. We deliberately keep this behind a small interface so the model is
agnostic to the frontend: the default path uses features cached on disk (the frozen, fast regime we
train in); an end-to-end path wraps the vendored Beat This frontend so its weights can be fine-tuned.

Swapping frontends = implementing one ``extract`` method; nothing in the model changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent


class FeatureExtractor:
    """Interface: produce [num_frames, feature_dim] features for a batch of audio (or pre-cached input)."""

    @property
    def feature_dim(self) -> int:
        raise NotImplementedError

    def extract(self, audio_or_cached):  # -> torch.Tensor [batch, num_frames, feature_dim]
        raise NotImplementedError


class CachedFeatureExtractor(FeatureExtractor):
    """Pass-through for features already computed and cached on disk.

    In the cached regime the features are loaded by data/dataset.py, so ``extract`` is the identity
    -- it exists only so training/eval code can treat the cached and end-to-end paths uniformly.
    """

    def __init__(self, feature_dim: int):
        self._feature_dim = feature_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract(self, cached_features: torch.Tensor) -> torch.Tensor:
        return cached_features


class BeatThisFeatureExtractor(FeatureExtractor):
    """Wraps the vendored Beat This frontend (external/beat_this) to compute features from audio.

    Used only for the end-to-end divergence (unfreezing the frontend). The 512-dim feature is the
    penultimate representation ``transformer_blocks(frontend(spectrogram))`` -- everything Beat This
    computes BEFORE collapsing to its two beat/downbeat logits. See external/README.md for provenance.
    """

    def __init__(self, checkpoint_name_or_path: str = "final0", trainable: bool = True):
        if str(_REPO_ROOT / "external" / "beat_this") not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT / "external" / "beat_this"))
        from beat_this.model.beat_tracker import BeatThis  # vendored upstream model

        self._model = BeatThis()
        self._trainable = trainable
        for parameter in self._model.parameters():
            parameter.requires_grad_(trainable)
        # The penultimate width is 512 for the default Beat This configuration.
        self._feature_dim = 512

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract(self, spectrogram: torch.Tensor) -> torch.Tensor:
        core = getattr(self._model, "_orig_mod", self._model)  # unwrap torch.compile if present
        return core.transformer_blocks(core.frontend(spectrogram))


def build_feature_extractor(end_to_end: bool, feature_dim: int) -> FeatureExtractor:
    """Pick the extractor implied by the config: cached (frozen) by default, Beat This for end-to-end."""
    if end_to_end:
        return BeatThisFeatureExtractor(trainable=True)
    return CachedFeatureExtractor(feature_dim=feature_dim)
