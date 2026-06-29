"""Loading cached frontend features and sampling training batches.

Each cached file is a dict with at least:
    "activations":      FloatTensor [num_frames, feature_dim]   -- frozen frontend features for one song
    "beat_targets":     FloatTensor [num_frames]                -- 1.0 at frames containing a beat, else 0.0
    "downbeat_targets": FloatTensor [num_frames]                -- 1.0 at frames containing a downbeat
These are produced once by the feature extractor (see feature_extractor.py) and reused, so training
never re-runs the (frozen) frontend. One loaded song is a ``Song`` triple of CPU tensors.
"""
from __future__ import annotations

import glob
import random
from dataclasses import dataclass

import torch


@dataclass
class Song:
    features: torch.Tensor          # [num_frames, feature_dim]
    beat_targets: torch.Tensor      # [num_frames]
    downbeat_targets: torch.Tensor  # [num_frames]


def load_songs(feature_dir: str, num_songs: int, seed: int,
               min_frames: int = 400, min_beats: int = 8) -> list[Song]:
    """Load up to ``num_songs`` usable songs from a directory of cached feature files.

    Songs that are too short or nearly beatless are skipped (they make the metrics meaningless and
    destabilize training). Selection is deterministic given ``seed``.
    """
    file_paths = sorted(glob.glob(f"{feature_dir}/*.pt"))
    random.Random(seed).shuffle(file_paths)
    songs: list[Song] = []
    for file_path in file_paths:
        if len(songs) >= num_songs:
            break
        record = torch.load(file_path, map_location="cpu")
        features = record["activations"].float()
        beat_targets = record["beat_targets"].float()
        downbeat_targets = record["downbeat_targets"].float()
        if features.shape[0] < min_frames or beat_targets.sum() < min_beats:
            continue
        songs.append(Song(features, beat_targets, downbeat_targets))
    return songs


def sample_training_batch(songs: list[Song], crop_length_frames: int, batch_size: int,
                          device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a batch of random fixed-length crops, returned as [batch, crop_length, ...] tensors.

    Returns (features, beat_targets, downbeat_targets) already moved to ``device``.
    """
    chosen_features, chosen_beats, chosen_downbeats = [], [], []
    for _ in range(batch_size):
        song = songs[random.randrange(len(songs))]
        num_frames = song.features.shape[0]
        start = random.randrange(0, max(1, num_frames - crop_length_frames))
        end = start + crop_length_frames
        chosen_features.append(song.features[start:end])
        chosen_beats.append(song.beat_targets[start:end])
        chosen_downbeats.append(song.downbeat_targets[start:end])
    # Crops near the end of a short song may be shorter than crop_length_frames; trim all to the min.
    usable_length = min(crop.shape[0] for crop in chosen_features)
    features = torch.stack([c[:usable_length] for c in chosen_features]).to(device)
    beat_targets = torch.stack([c[:usable_length] for c in chosen_beats]).to(device)
    downbeat_targets = torch.stack([c[:usable_length] for c in chosen_downbeats]).to(device)
    return features, beat_targets, downbeat_targets
