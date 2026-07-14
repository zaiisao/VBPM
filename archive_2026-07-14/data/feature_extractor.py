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


def build_feature_extractor(name: str, live: bool = False) -> FeatureExtractor:
    """Registry/factory from ``configs/frontends/<name>.yaml``.

    ``live=False`` (the trained regime) returns a CachedFeatureExtractor whose feature_dim comes
    from that frontend's YAML, so cached and live paths always agree; ``live=True`` instantiates
    the actual frontend for extraction / end-to-end work.
    """
    from config import load_frontend_config
    frontend = load_frontend_config(name)
    if not live:
        return CachedFeatureExtractor(feature_dim=frontend.feature_dim)
    registry = {"beat_this": lambda: BeatThisFeatureExtractor(frontend.checkpoint),
                "mert_v1_95m": lambda: MERTFeatureExtractor(target_fps=frontend.cache_fps)}
    if name not in registry:
        raise KeyError(f"no live adapter for frontend '{name}' (have: {sorted(registry)})")
    return registry[name]()


class MERTFeatureExtractor(FeatureExtractor):
    """Wraps MERT (self-supervised music representation; external/MERT-v1-95M submodule).

    Scientifically distinct from the Beat This frontend: MERT saw NO beat supervision, so with this
    frontend every bit of beat knowledge in the system flows through the VBPM objective. 768-dim
    hidden states at ~75 fps, time-interpolated to the target cache fps (default 86.1328, matching
    the bt_*_rich caches so crops/targets code is unchanged).

    NOTE (weight-norm quirk): the released checkpoint stores the positional conv as
    ``weight_g``/``weight_v``; transformers >= 4.31 renames these to
    ``parametrizations.weight.original{0,1}`` and silently re-initializes them on load. We remap
    explicitly -- without this the positional embedding is RANDOM and features are subtly broken.
    """

    NATIVE_SAMPLE_RATE = 24000

    def __init__(self, layer: int = -1, target_fps: float = 22050.0 / 256.0, device: str = "cuda"):
        from transformers import AutoModel
        model_dir = str(_REPO_ROOT / "external" / "MERT-v1-95M")
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        state = torch.load(f"{model_dir}/pytorch_model.bin", map_location="cpu")
        fixes = {}
        for old, new in (("encoder.pos_conv_embed.conv.weight_g",
                          "encoder.pos_conv_embed.conv.parametrizations.weight.original0"),
                         ("encoder.pos_conv_embed.conv.weight_v",
                          "encoder.pos_conv_embed.conv.parametrizations.weight.original1")):
            if old in state:
                fixes[new] = state[old]
        missing, unexpected = self.model.load_state_dict(fixes, strict=False)
        assert not unexpected, f"pos_conv remap failed: {unexpected}"
        self.model = self.model.eval().to(device)
        self.layer = layer
        self.target_fps = target_fps
        self.device = device

    @property
    def feature_dim(self) -> int:
        return self.model.config.hidden_size          # 768 for MERT-v1-95M

    @torch.no_grad()
    def extract(self, waveform_24k: torch.Tensor) -> torch.Tensor:
        """[num_samples] mono @ 24 kHz -> [num_frames_at_target_fps, feature_dim]."""
        wav = waveform_24k.to(self.device).reshape(1, -1)
        hidden = self.model(input_values=wav, output_hidden_states=True).hidden_states[self.layer]
        num_target = int(round(hidden.shape[1] * self.target_fps
                               / (self.NATIVE_SAMPLE_RATE / 320.0)))   # MERT hop = 320 samples
        return torch.nn.functional.interpolate(
            hidden.transpose(1, 2), size=num_target, mode="linear", align_corners=False
        ).transpose(1, 2)[0].cpu()
