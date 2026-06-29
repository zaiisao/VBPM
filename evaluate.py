"""Evaluation: deploy the way VBPM does -- discard the decoder, read beats/downbeats from the phase.

Reports beat/downbeat F-measure under three audio conditions to expose any leakage:
  * "real"    -> the song's own features (the real score).
  * "shuffle" -> another song's features (must collapse; otherwise the model isn't using THIS audio).
  * "zero"    -> zeroed features (must collapse; otherwise the read-out is input-independent).
A model that genuinely tracks the audio scores high on "real" and near-chance on "shuffle"/"zero".
"""
from __future__ import annotations

import numpy as np
import torch

from config import Config, FRAMES_PER_SECOND
from data.dataset import Song
from data.targets import ground_truth_beat_times
from model.bar_pointer_vae import BarPointerVAE
from model import readout


@torch.no_grad()
def evaluate_geometric(model: BarPointerVAE, songs: list[Song], config: Config,
                       audio_condition: str = "real") -> dict:
    """Mean geometric beat/downbeat F-measure over songs for one audio condition."""
    model.eval()
    beat_scores, downbeat_scores = [], []
    num_songs = len(songs)
    silent_channel_cache = {}

    for song_index, song in enumerate(songs):
        if audio_condition == "shuffle":
            source_features = songs[(song_index + 1) % num_songs].features
        else:
            source_features = song.features
        num_frames = min(source_features.shape[0], song.beat_targets.shape[0], config.eval_max_frames)

        if audio_condition == "zero":
            features = torch.zeros(1, num_frames, config.feature_dim, device=config.device)
        else:
            features = source_features[:num_frames].unsqueeze(0).to(config.device)
        silent = torch.zeros(1, num_frames, device=config.device)  # no teacher-forced beats at eval

        result = model.rollout(features, silent, silent, sample=False, compute_kl=False)
        phase = result.phase[0].cpu().numpy()

        reference_beats = ground_truth_beat_times(song.beat_targets.numpy()[:num_frames], FRAMES_PER_SECOND)
        reference_downbeats = ground_truth_beat_times(song.downbeat_targets.numpy()[:num_frames], FRAMES_PER_SECOND)
        if len(reference_beats) >= 2:
            estimated_beats = readout.phase_to_beat_times(phase, config.beats_per_bar, FRAMES_PER_SECOND)
            beat_scores.append(readout.f_measure(reference_beats, estimated_beats, config.eval_beat_tolerance_seconds))
        if len(reference_downbeats) >= 2:
            estimated_downbeats = readout.phase_to_downbeat_times(phase, FRAMES_PER_SECOND)
            downbeat_scores.append(readout.f_measure(reference_downbeats, estimated_downbeats, config.eval_beat_tolerance_seconds))

    model.train()
    mean = lambda values: float(np.nanmean(values)) if values else float("nan")
    return {"beat_f": mean(beat_scores), "downbeat_f": mean(downbeat_scores)}


def evaluate_with_leak_test(model: BarPointerVAE, songs: list[Song], config: Config) -> dict:
    """Run all three audio conditions; the gap real-vs-(shuffle,zero) is the audio-locking evidence."""
    return {
        "real": evaluate_geometric(model, songs, config, "real"),
        "shuffle": evaluate_geometric(model, songs, config, "shuffle"),
        "zero": evaluate_geometric(model, songs, config, "zero"),
    }
