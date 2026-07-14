"""Cached frozen-frontend features + beat annotations: the Song container and loaders.

Single source for both the package and the notebook (nblib.data folds into this). Records are the
``.pt`` activation caches written by the extraction scripts: ``activations`` [T, feature_dim],
``beat_targets``/``downbeat_targets`` [T], optional ``act2`` [T, 2] frontend beat/downbeat
probabilities (the deploy-time observable evidence for the particle filter).
"""
import glob
import random
from dataclasses import dataclass

import torch


@dataclass
class Song:
    features: torch.Tensor            # [num_frames, feature_dim] frozen frontend activations
    beat_targets: torch.Tensor        # [num_frames] 1.0 on beat frames
    downbeat_targets: torch.Tensor    # [num_frames] 1.0 on downbeat frames
    frontend_activations: torch.Tensor = None   # [num_frames, 2] frontend beat/downbeat probabilities
                                      # (cached "act2") -- the deploy-time OBSERVABLE filter evidence


class FoldContaminationError(RuntimeError):
    """Raised when TRAINING data violates the Beat This 8-fold protocol (user directive
    2026-07-11: fold violation must not even be an option). A training record must carry fold
    provenance ("fold" + "extractor" == "beat_this-fold{fold}", i.e. the checkpoint that held the
    song out) or be marked "clean_frontend" (song never in Beat This training, e.g. GTZAN).
    Legacy final0-extracted caches (bt_*_rich, ...) are memorized evidence: the model would never
    see a real frontend error, and every cross-system claim off them was retracted on 2026-07-10.
    There is deliberately NO bypass flag -- re-extract with data/extract_fold_honest.py."""


def _assert_fold_honest(record, file_path):
    if record.get("clean_frontend", False):
        return
    fold = record.get("fold")
    extractor = record.get("extractor", "")
    if fold is None or extractor != f"beat_this-fold{fold}":
        raise FoldContaminationError(
            f"{file_path}: no fold-honest provenance (fold={fold!r}, extractor={extractor!r}). "
            "Training on frontend-memorized evidence is disabled; see FoldContaminationError.")


def load_cached_songs(feature_dir, num_songs, selection_seed, min_frames=400, min_beats=8,
                      for_training=False):
    # Deterministic selection; skips songs too short/beatless to score meaningfully.
    # ``for_training=True`` enforces the Beat This fold protocol on every record (no bypass).
    file_paths = sorted(glob.glob(f"{feature_dir}/*.pt"))
    random.Random(selection_seed).shuffle(file_paths)
    songs = []
    for file_path in file_paths:
        if len(songs) >= num_songs:
            break
        record = torch.load(file_path, map_location="cpu")
        if for_training:
            _assert_fold_honest(record, file_path)
        if record["activations"].shape[0] < min_frames or record["beat_targets"].sum() < min_beats:
            continue
        songs.append(Song(record["activations"].float(), record["beat_targets"].float(),
                          record["downbeat_targets"].float(),
                          record["act2"].float() if "act2" in record else None))
    return songs


def load_meter_only_clips(feature_dir, min_frames=400):
    """Clips carrying a meter label but NO beat annotations (e.g. Meter2800). Enforces the fold
    protocol: records must be clean_frontend (sources outside Beat This's beat-training data).
    Returns a list of (features [T, D], meter_class_index) with class k = k+1 beats/bar."""
    clips = []
    for file_path in sorted(glob.glob(f"{feature_dir}/*.pt")):
        record = torch.load(file_path, map_location="cpu")
        _assert_fold_honest(record, file_path)
        if record["activations"].shape[0] < min_frames:
            continue
        clips.append((record["activations"].float(), int(record["meter"]) - 1))
    return clips


def sample_meter_only_crops(clips, crop_length_frames, batch_size, num_meters, device="cuda"):
    """Random crops from meter-only clips -> (features, meter_class_targets); clips whose meter
    exceeds the model's class range are excluded by the caller's filtering at load time."""
    feature_crops, classes = [], []
    while len(feature_crops) < batch_size:
        features, meter_class = clips[random.randrange(len(clips))]
        if meter_class >= num_meters:
            continue
        start = random.randrange(0, max(1, features.shape[0] - crop_length_frames))
        feature_crops.append(features[start:start + crop_length_frames])
        classes.append(meter_class)
    shortest = min(crop.shape[0] for crop in feature_crops)
    return (torch.stack([c[:shortest] for c in feature_crops]).to(device),
            torch.tensor(classes, device=device))


def sample_training_crops(songs, crop_length_frames, batch_size, device="cuda", return_obs=False):
    # Random fixed-length crops -> (features, beat_targets, downbeat_targets[, obs]) on device.
    # return_obs also returns the frontend_activations crop (aligned to the same window) -- the
    # test-time-available observations the FIVO filter scores against.
    feature_crops, beat_crops, downbeat_crops, obs_crops = [], [], [], []
    for _ in range(batch_size):
        song = songs[random.randrange(len(songs))]
        start_frame = random.randrange(0, max(1, song.features.shape[0] - crop_length_frames))
        end_frame = start_frame + crop_length_frames
        feature_crops.append(song.features[start_frame:end_frame])
        beat_crops.append(song.beat_targets[start_frame:end_frame])
        downbeat_crops.append(song.downbeat_targets[start_frame:end_frame])
        if return_obs:
            if song.frontend_activations is None:
                raise ValueError("return_obs=True but a sampled song has no frontend_activations")
            obs_crops.append(song.frontend_activations[start_frame:end_frame])
    shortest = min(crop.shape[0] for crop in feature_crops)
    stack = lambda crops: torch.stack([crop[:shortest] for crop in crops]).to(device)
    if return_obs:
        return stack(feature_crops), stack(beat_crops), stack(downbeat_crops), stack(obs_crops)
    return stack(feature_crops), stack(beat_crops), stack(downbeat_crops)
