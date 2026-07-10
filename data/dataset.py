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


def sample_training_crops(songs, crop_length_frames, batch_size, device="cuda"):
    # Random fixed-length crops -> (features, beat_targets, downbeat_targets) on device.
    feature_crops, beat_crops, downbeat_crops = [], [], []
    for _ in range(batch_size):
        song = songs[random.randrange(len(songs))]
        start_frame = random.randrange(0, max(1, song.features.shape[0] - crop_length_frames))
        end_frame = start_frame + crop_length_frames
        feature_crops.append(song.features[start_frame:end_frame])
        beat_crops.append(song.beat_targets[start_frame:end_frame])
        downbeat_crops.append(song.downbeat_targets[start_frame:end_frame])
    shortest = min(crop.shape[0] for crop in feature_crops)
    stack = lambda crops: torch.stack([crop[:shortest] for crop in crops]).to(device)
    return stack(feature_crops), stack(beat_crops), stack(downbeat_crops)
