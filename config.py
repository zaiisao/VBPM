"""Thin typed configuration: a dataclass schema + ``load_config(path)`` reading YAML.

The YAML files under ``configs/`` are the source of truth (``configs/default.yaml`` is the pinned
2026-07-10 recipe); this module only gives them types, defaults, and dot-access. The frontend block
comes from ``configs/frontends/<name>.yaml`` and carries every frontend-specific property (fps,
feature_dim, sample rate, ...) so that switching frontend = swapping one YAML.
"""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class FrontendConfig:
    name: str = "beat_this"
    feature_dim: int = 512            # penultimate width (after chopping the 2-channel head)
    native_fps: float = 50.0
    cache_fps: float = 22050.0 / 256.0   # the grid every cache/target uses
    sample_rate: int = 22050
    checkpoint: str = "final0"
    submodule: str = "external/beat_this"
    provides_activations: bool = True    # act2 beat/downbeat probs -> filter evidence


@dataclass
class ModelConfig:
    hidden_size: int = 64
    num_meters: int = 4               # meter latent classes (k -> k+1 beats/bar)
    emission: str = "parametric"      # parametric | full | no_tempo | phase_only (diagnosis ladder)
    transition_correction_scale: float = 0.5
    fixed_prior_scales: list | None = None   # or [sigma, concentration]


@dataclass
class ObjectiveConfig:
    free_bits_nats_per_frame: float = 0.3
    prior_preserving_free_bits: bool = True
    sawtooth_weight: float = 0.0      # NOSAW verdict 2026-07-10: optional given the slope term
    sawtooth_family: str = "von_mises"
    sawtooth_wc_rho: float = 0.7
    tempo_slope_weight: float = 0.5
    meter_ce_weight: float = 0.1
    # Sohn et al. 2015 eq. (9): L = alpha * L_CVAE + (1 - alpha) * L_GSNN. 1.0 = pure CVAE.
    hybrid_alpha: float = 1.0
    # Ramp-target beat spacing when annotations cannot decide it; an explicit labelled fallback,
    # never a modeling assumption (the model's beats-per-bar always comes from the meter latent).
    target_beats_per_bar: int = 4


@dataclass
class TrainingConfig:
    steps: int = 2000
    batch_size: int = 32
    crop_frames: int = 1024
    learning_rate: float = 1.0e-3
    grad_clip_norm: float = 5.0
    train_songs: int = 9999
    val_songs: int = 16
    train_feature_dir: str = "cache/acts/bt_train_rich"
    val_feature_dir: str = "cache/acts/bt_val_rich"
    extra_train_dirs: list = field(default_factory=list)   # tempo-aug pool directories
    aug_songs_per_dir: int = 200
    log_every_steps: int = 100
    eval_max_frames: int = 1600
    save_path: str = ""


@dataclass
class FilterConfig:
    num_particles: int = 800
    observation_temperature: float = 3.0
    proposal_tempo_sigma_scale: float = 0.01
    proposal_phase_concentration_scale: float = 50.0
    downbeat_evidence_weight: float = 3.0
    stratified_gauge_init: bool = True


@dataclass
class Config:
    seed: int = 0
    device: str = "cuda"
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)


def _build(dataclass_type, mapping):
    if not mapping:
        return dataclass_type()
    known = {f for f in dataclass_type.__dataclass_fields__}
    unknown = set(mapping) - known
    if unknown:
        raise KeyError(f"unknown {dataclass_type.__name__} keys: {sorted(unknown)}")
    return dataclass_type(**mapping)


def load_frontend_config(name):
    with open(REPO_ROOT / "configs" / "frontends" / f"{name}.yaml") as fh:
        return _build(FrontendConfig, yaml.safe_load(fh))


def load_config(path="configs/default.yaml"):
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    frontend = load_frontend_config(raw.pop("frontend", "beat_this"))
    return Config(
        seed=raw.pop("seed", 0),
        device=raw.pop("device", "cuda"),
        frontend=frontend,
        model=_build(ModelConfig, raw.pop("model", {})),
        objective=_build(ObjectiveConfig, raw.pop("objective", {})),
        training=_build(TrainingConfig, raw.pop("training", {})),
        filter=_build(FilterConfig, raw.pop("filter", {})),
    )
