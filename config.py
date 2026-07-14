"""Configuration for the bar-pointer Neural HMM (Baseline B of the A -> B -> C progression).

    A: handcrafted HMM (madmom DBN)  ->  B: neural HMM (this repo)  ->  C: Transformer VAE (later)

The generative model is an INPUT-DRIVEN HMM: a Markov transition p(z_t | z_{t-1}, h_t) with the
audio context h_t = f(x), and a factorized emission p(obs_t | z_t). That structure (finite state +
Markov + factorized) is exactly what makes the forward algorithm and Viterbi EXACT (Mechanism 3).
"""
from dataclasses import dataclass, field


@dataclass
class StateSpaceConfig:
    """Discretization of the continuous bar-pointer state (phase x tempo x meter)."""
    n_phase: int = 180          # phase bins over [0, 2pi) == one bar
    n_tempo: int = 32           # tempo bins
    min_bpm: float = 50.0
    max_bpm: float = 215.0
    meters: tuple = (3, 4)      # candidate beats-per-bar values


@dataclass
class ModelConfig:
    feature_dim: int = 512      # frozen frontend feature dim (e.g. Beat This penultimate)
    hidden_size: int = 64       # context / factor-MLP width
    obs_dim: int = 2            # observation channels (beat, downbeat)


@dataclass
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 8
    max_steps: int = 20_000
    grad_clip: float = 1.0
    device: str = "cuda"
    save_path: str = "checkpoints/neural_hmm.pt"


@dataclass
class DataConfig:
    cache_dir: str = "cache/acts/foldhonest_train_rich"
    val_dir: str = "cache/acts/foldhonest_val_rich"
    fps: float = 22050.0 / 256.0


@dataclass
class Config:
    state: StateSpaceConfig = field(default_factory=StateSpaceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_config(path: str | None = None) -> Config:
    """Return a Config. With no path, return defaults; with a path, overlay values from that YAML."""
    if path is None:
        return Config()
    raise NotImplementedError("YAML overlay not implemented yet")
