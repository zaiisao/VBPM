"""Data layer: cached-feature loading, ground-truth/sawtooth targets, and feature extractors."""
from .dataset import Song, load_songs, sample_training_batch
from .targets import (
    build_sawtooth_phase_target,
    build_sawtooth_phase_target_batch,
    ground_truth_beat_times,
    ground_truth_tempo_bpm,
)
from .feature_extractor import (
    FeatureExtractor,
    CachedFeatureExtractor,
    BeatThisFeatureExtractor,
    build_feature_extractor,
)

__all__ = [
    "Song", "load_songs", "sample_training_batch",
    "build_sawtooth_phase_target", "build_sawtooth_phase_target_batch",
    "ground_truth_beat_times", "ground_truth_tempo_bpm",
    "FeatureExtractor", "CachedFeatureExtractor", "BeatThisFeatureExtractor", "build_feature_extractor",
]
