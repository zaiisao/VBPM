"""Cached-activation dataset: frozen frontend features + observations + targets, per song."""
from dataclasses import dataclass

import torch


@dataclass
class Song:
    features: torch.Tensor          # [T, feature_dim]  frozen frontend features (x)
    observations: torch.Tensor      # [T, obs_dim]      beat/downbeat evidence channels
    beat_targets: torch.Tensor      # [T]               for supervision / scoring
    downbeat_targets: torch.Tensor  # [T]
    name: str = ""


def load_cached_songs(cache_dir: str, n: int | None = None, selection_seed: int = 0):
    """Yield Song objects from a cache directory (respecting the fold split)."""
    raise NotImplementedError


class BeatDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split: str = "train"):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError


def build_dataloader(cfg, split: str = "train"):
    """Return a torch DataLoader (handles cropping / padding songs to equal length)."""
    raise NotImplementedError
