"""Dataset definitions for CHART training.

Phase .npy files contain columns [tempo, beat_phase, bar_phase] (3-col legacy)
or [tempo, beat_phase, bar_phase, meter_class] (4-col).  This module converts
them to the paper's representation:

- phase in radians [0, 2*pi)
- log-tempo in log(radians/frame)
- meter as one-hot vector

and derives binary beat targets from phase wrap-arounds.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
import random
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

TWO_PI = 2.0 * math.pi
_DEFAULT_NUM_METER_CLASSES = 8


def pad_collate(batch: list[dict]) -> dict:
    """Collate variable-length samples by padding to the max length in the batch."""
    keys = batch[0].keys()
    result: dict = {}

    for key in keys:
        vals = [sample[key] for sample in batch]
        if isinstance(vals[0], torch.Tensor):
            # Pad along the last dim to the max length
            max_len = max(v.shape[-1] for v in vals)
            padded = []
            for v in vals:
                pad_size = max_len - v.shape[-1]
                if pad_size > 0:
                    padded.append(F.pad(v, (0, pad_size)))
                else:
                    padded.append(v)
            result[key] = torch.stack(padded, dim=0)
        elif isinstance(vals[0], str):
            result[key] = vals
        else:
            result[key] = vals

    return result


def _meter_from_beat_downbeat(
    beats: np.ndarray,
    downbeats: np.ndarray,
    num_meter_classes: int = _DEFAULT_NUM_METER_CLASSES,
) -> np.ndarray:
    """Derive per-frame meter class by counting beats between downbeats.

    Meter class = (number of beats per bar) - 1, clamped to [0, K-1].
    E.g., 3/4 → class 2, 4/4 → class 3, 6/8 → class 5.

    Args:
        beats: [T] binary beat indicators.
        downbeats: [T] binary downbeat indicators.
        num_meter_classes: K.

    Returns:
        meter_class: [T] int array with meter class per frame.
    """
    T = len(beats)
    meter_class = np.full(T, 3, dtype=np.int64)  # default 4/4 = class 3

    beat_pos = np.where(beats > 0.5)[0]
    db_pos = np.where(downbeats > 0.5)[0]

    if len(db_pos) < 2 or len(beat_pos) < 2:
        return meter_class

    # For each pair of consecutive downbeats, count beats in between
    for i in range(len(db_pos) - 1):
        bar_start = db_pos[i]
        bar_end = db_pos[i + 1]
        beats_in_bar = int(np.sum((beat_pos >= bar_start) & (beat_pos < bar_end)))
        if beats_in_bar < 1:
            beats_in_bar = 4  # fallback
        cls = min(beats_in_bar - 1, num_meter_classes - 1)
        meter_class[bar_start:bar_end] = cls

    # Extend last bar to end of sequence
    if len(db_pos) >= 2:
        last_bar_start = db_pos[-1]
        prev_bar_start = db_pos[-2]
        beats_in_last = int(np.sum((beat_pos >= prev_bar_start) & (beat_pos < last_bar_start)))
        if beats_in_last < 1:
            beats_in_last = 4
        cls = min(beats_in_last - 1, num_meter_classes - 1)
        meter_class[last_bar_start:] = cls

    return meter_class


def _phase_npy_to_structured(
    phase_np: np.ndarray,
    num_meter_classes: int = _DEFAULT_NUM_METER_CLASSES,
) -> dict[str, torch.Tensor]:
    """Convert a raw phase .npy array to the paper's structured representation.

    Args:
        phase_np: ``[T, 3]`` or ``[T, 4]`` array with columns
            ``[tempo, beat_phase, bar_phase (, meter_class)]``.
        num_meter_classes: Number of supported meter categories K.

    Returns:
        Dict with keys: ``phase`` [T,1], ``log_tempo`` [T,1],
        ``meter_index`` [T], ``meter_onehot`` [T,K], ``beat_targets`` [T].
    """
    n_cols = phase_np.shape[1]

    tempo_bpf = phase_np[:, 0]       # beats/frame
    beat_phase_01 = phase_np[:, 1]   # [0, 1)
    # bar_phase_01 = phase_np[:, 2]  # unused directly; meter is explicit

    if n_cols >= 4:
        meter_class = phase_np[:, 3].astype(np.int64)
    else:
        # Legacy 3-col: assume 4/4 (class index 2)
        meter_class = np.full(phase_np.shape[0], 2, dtype=np.int64)

    meter_class = np.clip(meter_class, 0, num_meter_classes - 1)

    # Convert to paper's units
    # tempo: beats/frame -> radians/frame (one beat = 2*pi radians)
    tempo_rad = tempo_bpf * TWO_PI
    log_tempo = np.log(np.maximum(tempo_rad, 1e-8))

    # beat phase: [0, 1) -> [0, 2*pi)
    phase_rad = beat_phase_01 * TWO_PI

    # Binary beat targets: beat onset at phase wrap-around
    beat_targets = np.zeros(phase_np.shape[0], dtype=np.float32)
    beat_targets[1:] = (beat_phase_01[1:] < beat_phase_01[:-1]).astype(np.float32)

    phase_t = torch.as_tensor(phase_rad, dtype=torch.float32).unsqueeze(-1)      # [T, 1]
    log_tempo_t = torch.as_tensor(log_tempo, dtype=torch.float32).unsqueeze(-1)   # [T, 1]
    meter_idx_t = torch.as_tensor(meter_class, dtype=torch.long)                  # [T]
    meter_oh_t = F.one_hot(meter_idx_t, num_meter_classes).float()                # [T, K]
    beat_t = torch.as_tensor(beat_targets, dtype=torch.float32)                   # [T]

    return {
        "phase": phase_t,
        "log_tempo": log_tempo_t,
        "meter_index": meter_idx_t,
        "meter_onehot": meter_oh_t,
        "beat_targets": beat_t,
    }


def _build_prev_shifted(structured: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Shift structured latent variables right by 1 for autoregressive input.

    The first frame is zero-initialized.
    """
    T = structured["phase"].shape[0]

    phase_prev = torch.zeros_like(structured["phase"])
    phase_prev[1:] = structured["phase"][:-1]

    log_tempo_prev = torch.zeros_like(structured["log_tempo"])
    log_tempo_prev[1:] = structured["log_tempo"][:-1]

    meter_oh_prev = torch.zeros_like(structured["meter_onehot"])
    meter_oh_prev[1:] = structured["meter_onehot"][:-1]
    # Default first frame to uniform meter
    meter_oh_prev[0] = 1.0 / structured["meter_onehot"].shape[-1]

    return {
        "phase_prev": phase_prev,
        "log_tempo_prev": log_tempo_prev,
        "meter_onehot_prev": meter_oh_prev,
    }


class ActivationDataset(Dataset):
    """Dataset for paired activation/phase ``.npy`` files used by CHART.

    The dataset scans two directories, matches files by basename, and yields
    fixed-length training windows with structured latent variable targets.
    """

    def __init__(
        self,
        activations_dir: str | Path,
        phases_dir: str | Path,
        seq_len: int = 1024,
        num_meter_classes: int = _DEFAULT_NUM_METER_CLASSES,
    ) -> None:
        self.activations_dir = Path(activations_dir)
        self.phases_dir = Path(phases_dir)
        self.seq_len = seq_len
        self.num_meter_classes = num_meter_classes

        if seq_len <= 0:
            raise ValueError("seq_len must be a positive integer.")
        if not self.activations_dir.is_dir():
            raise FileNotFoundError(f"Activations directory not found: {self.activations_dir}")
        if not self.phases_dir.is_dir():
            raise FileNotFoundError(f"Phases directory not found: {self.phases_dir}")

        activation_map = {path.stem: path for path in self.activations_dir.rglob("*.npy")}
        phase_map = {path.stem: path for path in self.phases_dir.rglob("*.npy")}

        common_keys = sorted(set(activation_map).intersection(phase_map))
        self.matched_files: list[tuple[Path, Path]] = [
            (activation_map[key], phase_map[key]) for key in common_keys
        ]

        if len(self.matched_files) == 0:
            raise RuntimeError(
                "No matched .npy files found between activations_dir and phases_dir. "
                "Expected shared basenames (e.g., song_01.npy)."
            )

    def __len__(self) -> int:
        return len(self.matched_files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load, window, convert units, and build autoregressive history.

        Returns:
            Dictionary with keys: ``activations`` [T,2], ``beat_targets`` [T],
            ``phase`` [T,1], ``log_tempo`` [T,1], ``meter_index`` [T],
            ``meter_onehot`` [T,K], ``phase_prev`` [T,1],
            ``log_tempo_prev`` [T,1], ``meter_onehot_prev`` [T,K].
        """
        activation_path, phase_path = self.matched_files[index]

        activation_np = np.load(activation_path)
        phase_np = np.load(phase_path)

        if activation_np.ndim != 2 or activation_np.shape[1] != 2:
            raise ValueError(
                f"Expected activations shape [T, 2], got {activation_np.shape} for {activation_path}"
            )
        if phase_np.ndim != 2 or phase_np.shape[1] not in (3, 4):
            raise ValueError(
                f"Expected phases shape [T, 3] or [T, 4], got {phase_np.shape} for {phase_path}"
            )

        min_len = min(activation_np.shape[0], phase_np.shape[0])
        activation_np = activation_np[:min_len]
        phase_np = phase_np[:min_len]
        total_len = min_len

        # Window to fixed seq_len
        if total_len < self.seq_len:
            pad_len = self.seq_len - total_len
            activation_np = np.pad(activation_np, ((0, pad_len), (0, 0)))
            pad_cols = phase_np.shape[1]
            phase_np = np.pad(phase_np, ((0, pad_len), (0, 0)))
        elif total_len > self.seq_len:
            start = random.randint(0, total_len - self.seq_len)
            end = start + self.seq_len
            activation_np = activation_np[start:end]
            phase_np = phase_np[start:end]

        activations = torch.as_tensor(activation_np, dtype=torch.float32)
        structured = _phase_npy_to_structured(phase_np, self.num_meter_classes)
        prev = _build_prev_shifted(structured)

        return {
            "activations": activations,
            **structured,
            **prev,
        }


class AudioPhaseBridgeDataset(Dataset):
    """Bridge dataset for end-to-end extractor -> CHART training.

    Wraps an audio-based extractor dataset and pairs each audio file with a
    phase ``.npy`` file, converting to the paper's structured representation.
    """

    def __init__(
        self,
        source_dataset: Dataset,
        phases_dir: str | Path,
        num_meter_classes: int = _DEFAULT_NUM_METER_CLASSES,
    ) -> None:
        self.source_dataset = source_dataset
        self.phases_dir = Path(phases_dir)
        self.num_meter_classes = num_meter_classes

        if not self.phases_dir.is_dir():
            raise FileNotFoundError(f"Phases directory not found: {self.phases_dir}")

        phase_files = list(self.phases_dir.rglob("*.npy"))
        if len(phase_files) == 0:
            raise RuntimeError(f"No .npy phase files found in: {self.phases_dir}")

        self.phase_by_stem: dict[str, Path] = {phase_path.stem: phase_path for phase_path in phase_files}

        audio_files = getattr(self.source_dataset, "audio_files", None)
        if audio_files is None:
            raise AttributeError("source_dataset must expose 'audio_files' for filename matching.")

        self.phase_path_by_audio_stem: dict[str, Path] = {}
        for audio_path in audio_files:
            phase_path = self._resolve_phase_path_from_audio(audio_path)
            audio_stem = Path(audio_path).stem.replace("_L+R", "")
            self.phase_path_by_audio_stem[audio_stem] = phase_path

    def __len__(self) -> int:
        if isinstance(self.source_dataset, Sized):
            return len(cast(Sized, self.source_dataset))
        raise TypeError("source_dataset must implement __len__().")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.source_dataset[index]
        if not isinstance(sample, tuple) or len(sample) < 2:
            raise TypeError("Expected tuple sample from source dataset: (audio, target, ...)")

        audio = sample[0].float()
        extractor_target = sample[1].float()

        audio_path = self._get_audio_path_for_index(index)

        audio_stem = Path(audio_path).stem.replace("_L+R", "")
        phase_path = self.phase_path_by_audio_stem.get(audio_stem)
        if phase_path is None:
            phase_path = self._resolve_phase_path_from_audio(audio_path)
            self.phase_path_by_audio_stem[audio_stem] = phase_path

        phase_np = np.load(phase_path).astype(np.float32)
        if phase_np.ndim != 2 or phase_np.shape[1] not in (3, 4):
            raise ValueError(
                f"Expected phase shape [T, 3] or [T, 4], got {phase_np.shape} for {phase_path}"
            )

        # Align phase file with the audio crop.
        #
        # The phase file is at phase_fps (default 100). The extractor target
        # is at target_fps (sample_rate/256). We resample the FULL-SONG phase
        # to target_fps, then find the crop offset by matching beat positions.
        target_len = extractor_target.shape[-1]  # cropped length
        beat_target = extractor_target[0].numpy()  # [target_len] binary beats

        # Compute full-song length at target fps from the phase file's length.
        # Phase file covers the full song at phase_fps; convert to target_fps.
        phase_fps = 100.0  # default from phase_generation/common.py
        target_factor = getattr(self.source_dataset, 'target_factor', 256)
        sample_rate = getattr(self.source_dataset, 'audio_sample_rate', 22050)
        target_fps = sample_rate / target_factor  # 22050/256 = 86.13
        full_target_len = int(phase_np.shape[0] * (target_fps / phase_fps))

        if full_target_len > target_len:
            # Resample phase to full-song length at target fps
            full_phase = self._fit_phase_length_np(
                phase_np, full_target_len,
                phase_fps=phase_fps, target_fps=target_fps,
            )

            # Find crop offset by matching beat positions
            crop_offset = self._find_phase_crop_offset(full_phase, beat_target, target_len)
            phase_np = full_phase[crop_offset : crop_offset + target_len]

        if phase_np.shape[0] != target_len:
            # Fallback: resample directly
            phase_np = self._fit_phase_length_np(
                np.load(phase_path).astype(np.float32), target_len,
                phase_fps=phase_fps, target_fps=target_fps,
            )

        structured = _phase_npy_to_structured(phase_np, self.num_meter_classes)

        # Derive meter from beat/downbeat annotations instead of hardcoding 4/4.
        # extractor_target: [2, T_ext] — channel 0 = beats, channel 1 = downbeats
        if extractor_target.shape[0] >= 2:
            beat_np = extractor_target[0].numpy()
            db_np = extractor_target[1].numpy()
            meter_derived = _meter_from_beat_downbeat(
                beat_np, db_np, self.num_meter_classes
            )
            meter_idx = torch.as_tensor(meter_derived, dtype=torch.long)
            structured["meter_index"] = meter_idx
            structured["meter_onehot"] = F.one_hot(
                meter_idx, self.num_meter_classes
            ).float()

        # Re-derive beat_phase from WaveBeat beat positions to eliminate
        # the resampling offset. The sawtooth ramps 0→2π between consecutive
        # beats, guaranteeing exact alignment with beat_targets.
        wavebeat_beat_targets = extractor_target[0].float()  # [T_ext] binary, aligned
        beat_pos = torch.where(wavebeat_beat_targets > 0.5)[0].tolist()
        if len(beat_pos) >= 2:
            T_phase = structured["phase"].shape[0]
            realigned_phase = torch.zeros(T_phase, 1)
            # Before first beat
            if beat_pos[0] > 0:
                realigned_phase[:beat_pos[0], 0] = torch.linspace(
                    0, TWO_PI * (1 - 1 / max(beat_pos[0], 1)), beat_pos[0]
                )
            # Between consecutive beats
            for i in range(len(beat_pos) - 1):
                s, e = beat_pos[i], beat_pos[i + 1]
                if s < T_phase and e <= T_phase:
                    span = e - s
                    realigned_phase[s:e, 0] = torch.linspace(0, TWO_PI * (1 - 1 / span), span)
            # After last beat
            last = beat_pos[-1]
            if last < T_phase:
                remaining = T_phase - last
                realigned_phase[last:, 0] = torch.linspace(
                    0, TWO_PI * (1 - 1 / max(remaining, 1)), remaining
                )
            structured["phase"] = realigned_phase

        prev = _build_prev_shifted(structured)

        return {
            "audio": audio,
            "extractor_target": extractor_target,
            **structured,
            **prev,
            "beat_targets": wavebeat_beat_targets,  # override phase-derived targets
            "audio_path": str(audio_path),
            "phase_path": str(phase_path),
        }

    def _get_audio_path_for_index(self, index: int) -> str:
        get_audio_path_for_index = getattr(self.source_dataset, "get_audio_path_for_index", None)
        if callable(get_audio_path_for_index):
            return str(get_audio_path_for_index(index))

        audio_files = getattr(self.source_dataset, "audio_files", None)
        if audio_files is None or len(audio_files) == 0:
            raise AttributeError(
                "source_dataset must expose 'audio_files' or implement get_audio_path_for_index(index)."
            )

        audio_idx = index % len(audio_files)
        return str(audio_files[audio_idx])

    def _resolve_phase_path_from_audio(self, audio_path: str | Path) -> Path:
        basename = Path(audio_path).name
        candidates = [
            Path(basename).stem,
            Path(basename).stem.replace("_L+R", ""),
            basename.replace(".wav", ""),
            basename.replace("_L+R.wav", ""),
        ]

        for key in candidates:
            if key in self.phase_by_stem:
                return self.phase_by_stem[key]

        raise FileNotFoundError(
            f"No phase file found for audio '{audio_path}'. Tried keys: {candidates} in {self.phases_dir}"
        )

    @staticmethod
    def _find_phase_crop_offset(
        full_phase: np.ndarray, beat_target: np.ndarray, crop_len: int,
    ) -> int:
        """Find the frame offset where the phase file aligns with the beat target.

        Uses full cross-correlation (every offset checked) between beat
        positions derived from the phase file and the binary beat_target
        from WaveBeat.

        Args:
            full_phase: [T_full, C] resampled phase at target fps.
            beat_target: [crop_len] binary beat indicators from extractor.
            crop_len: Length of the crop window.

        Returns:
            Best offset (0-indexed) into full_phase.
        """
        T_full = full_phase.shape[0]
        if T_full <= crop_len:
            return 0

        if beat_target.sum() < 1:
            return 0

        # Create beat indicator from phase file:
        # beats occur where beat_phase (column 1) wraps from ~1 back to ~0
        beat_phase = full_phase[:, 1]  # [T_full]
        phase_diff = np.diff(beat_phase, prepend=beat_phase[0])
        phase_beats = (phase_diff < -0.5).astype(np.float32)

        # Full cross-correlation: check EVERY offset via np.correlate
        # mode='valid' gives scores for each valid offset where beat_target
        # fits entirely within phase_beats — exactly (T_full - crop_len + 1) values
        scores = np.correlate(phase_beats, beat_target, mode='valid')

        if len(scores) == 0:
            return 0

        return int(np.argmax(scores))

    @staticmethod
    @staticmethod
    def _fit_phase_length_np(phase_np: np.ndarray, target_len: int,
                             phase_fps: float = 100.0,
                             target_fps: float | None = None) -> np.ndarray:
        """Temporally resample a phase array to ``target_len`` frames.

        Maps each target frame to its exact time in seconds, then
        interpolates.  Circular columns (beat_phase, bar_phase in [0,1))
        are unwrapped before linear interpolation and re-wrapped after,
        so that the wrap point is placed at the correct sub-frame time
        with no systematic offset.
        """
        current_len = phase_np.shape[0]
        if current_len == target_len:
            return phase_np

        if target_fps is None:
            target_fps = phase_fps * target_len / current_len

        fps_ratio = phase_fps / target_fps

        # Continuous source positions for each target frame
        x_tgt = np.arange(target_len, dtype=np.float64) * fps_ratio
        x_tgt_clipped = np.clip(x_tgt, 0.0, current_len - 1.0)
        x_src = np.arange(current_len, dtype=np.float64)

        n_cols = phase_np.shape[1]
        result = np.empty((target_len, n_cols), dtype=np.float32)

        # Column 0 (tempo): linear interpolation + fps rescaling
        result[:, 0] = np.interp(x_tgt_clipped, x_src, phase_np[:, 0].astype(np.float64))
        result[:, 0] *= fps_ratio  # beats/src_frame → beats/tgt_frame

        # Columns 1-2 (beat_phase, bar_phase): unwrap → linear interp → re-wrap
        # Circular values in [0, 1) have discontinuous wraps (0.99 → 0.01).
        # Unwrapping converts to a monotonically increasing signal, enabling
        # correct linear interpolation across wrap boundaries.
        for c in range(1, min(n_cols, 3)):
            src = phase_np[:, c].astype(np.float64)
            # Unwrap: accumulate forward differences, treating jumps < -0.5 as wraps
            unwrapped = np.empty_like(src)
            unwrapped[0] = src[0]
            for i in range(1, len(src)):
                diff = src[i] - src[i - 1]
                if diff < -0.5:
                    diff += 1.0  # wrap: 0.99 → 0.01 becomes +0.02
                unwrapped[i] = unwrapped[i - 1] + diff
            # Linear interpolation on the unwrapped signal
            interp = np.interp(x_tgt_clipped, x_src, unwrapped)
            # Re-wrap to [0, 1)
            result[:, c] = (interp % 1.0).astype(np.float32)

        if n_cols >= 4:
            # Column 3 (meter_class) is discrete: nearest-neighbor
            idx = np.clip(np.floor(x_tgt).astype(np.int64), 0, current_len - 1)
            result[:, 3] = phase_np[idx, 3].astype(np.float32)

        return result


class WaveBeatPhaseDataset(AudioPhaseBridgeDataset):
    """Backward-compatible alias for previous class name."""


@dataclass(frozen=True)
class WaveBeatDatasetSpec:
    """Resolved dataset specification for a WaveBeat-compatible source."""

    key: str
    wavebeat_dataset: str
    audio_dir: Path
    annot_dir: Path


class BaseWaveBeatSpecResolver:
    """Base class for root-folder dataset resolvers."""

    key: str

    def resolve(self, root_dir: Path) -> WaveBeatDatasetSpec | None:
        raise NotImplementedError


def _resolve_data_label_dirs(root_dir: Path, dataset_dir_name: str) -> tuple[Path, Path] | None:
    layout_candidates = [
        root_dir / "labeled_data" / dataset_dir_name,
        root_dir / dataset_dir_name,
    ]

    for dataset_root in layout_candidates:
        audio_dir = dataset_root / "data"
        annot_dir = dataset_root / "label"
        if audio_dir.is_dir() and annot_dir.is_dir():
            return audio_dir, annot_dir

    return None


class BallroomSpecResolver(BaseWaveBeatSpecResolver):
    key = "ballroom"

    def resolve(self, root_dir: Path) -> WaveBeatDatasetSpec | None:
        resolved_dirs = _resolve_data_label_dirs(root_dir, "ballroom")
        if resolved_dirs is not None:
            audio_dir, annot_dir = resolved_dirs
            return WaveBeatDatasetSpec(
                key=self.key,
                wavebeat_dataset="ballroom",
                audio_dir=audio_dir,
                annot_dir=annot_dir,
            )
        return None



class BeatlesSpecResolver(BaseWaveBeatSpecResolver):
    key = "beatles"

    def resolve(self, root_dir: Path) -> WaveBeatDatasetSpec | None:
        resolved_dirs = _resolve_data_label_dirs(root_dir, "beatles")
        if resolved_dirs is not None:
            audio_dir, annot_dir = resolved_dirs
            return WaveBeatDatasetSpec(
                key=self.key,
                wavebeat_dataset="beatles",
                audio_dir=audio_dir,
                annot_dir=annot_dir,
            )
        return None


class BeatlesOldSpecResolver(BaseWaveBeatSpecResolver):
    key = "beatles_old"

    def resolve(self, root_dir: Path) -> WaveBeatDatasetSpec | None:
        resolved_dirs = _resolve_data_label_dirs(root_dir, "beatles_old")
        if resolved_dirs is not None:
            audio_dir, annot_dir = resolved_dirs
            return WaveBeatDatasetSpec(
                key=self.key,
                wavebeat_dataset="beatles",
                audio_dir=audio_dir,
                annot_dir=annot_dir,
            )
        return None


class GTZANSpecResolver(BaseWaveBeatSpecResolver):
    key = "gtzan"

    def resolve(self, root_dir: Path) -> WaveBeatDatasetSpec | None:
        resolved_dirs = _resolve_data_label_dirs(root_dir, "gtzan")
        if resolved_dirs is not None:
            audio_dir, annot_dir = resolved_dirs
            return WaveBeatDatasetSpec(
                key=self.key,
                wavebeat_dataset="gtzan",
                audio_dir=audio_dir,
                annot_dir=annot_dir,
            )
        return None


class HainsSpecResolver(BaseWaveBeatSpecResolver):
    key = "hains"

    def resolve(self, root_dir: Path) -> WaveBeatDatasetSpec | None:
        resolved_dirs = _resolve_data_label_dirs(root_dir, "hains")
        if resolved_dirs is not None:
            audio_dir, annot_dir = resolved_dirs
            return WaveBeatDatasetSpec(
                key=self.key,
                wavebeat_dataset="hainsworth",
                audio_dir=audio_dir,
                annot_dir=annot_dir,
            )
        return None


class RWCPopularSpecResolver(BaseWaveBeatSpecResolver):
    key = "rwc_popular"

    def resolve(self, root_dir: Path) -> WaveBeatDatasetSpec | None:
        resolved_dirs = _resolve_data_label_dirs(root_dir, "rwc_popular")
        if resolved_dirs is not None:
            audio_dir, annot_dir = resolved_dirs
            return WaveBeatDatasetSpec(
                key=self.key,
                wavebeat_dataset="rwc_popular",
                audio_dir=audio_dir,
                annot_dir=annot_dir,
            )
        return None


def discover_wavebeat_dataset_specs(
    root_dir: str | Path,
    include_keys: set[str] | None = None,
) -> list[WaveBeatDatasetSpec]:
    """Discover known dataset folders from a single root directory."""
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset_root not found: {root}")

    if include_keys is not None and len(include_keys) == 0:
        raise ValueError("include_keys must be non-empty when provided.")

    resolvers: list[BaseWaveBeatSpecResolver] = [
        BallroomSpecResolver(),
        BeatlesSpecResolver(),
        BeatlesOldSpecResolver(),
        GTZANSpecResolver(),
        HainsSpecResolver(),
        RWCPopularSpecResolver(),
    ]

    specs: list[WaveBeatDatasetSpec] = []
    for resolver in resolvers:
        if include_keys is not None and resolver.key not in include_keys:
            continue
        if spec := resolver.resolve(root):
            specs.append(spec)

    return specs


class MultiSourceAudioDataset(Dataset):
    """Dataset adapter that combines multiple audio extractor datasets."""

    def __init__(self, source_datasets: list[Dataset], source_keys: list[str]) -> None:
        if len(source_datasets) == 0:
            raise ValueError("source_datasets must contain at least one dataset.")
        if len(source_datasets) != len(source_keys):
            raise ValueError("source_datasets and source_keys must have the same length.")

        self.source_datasets = source_datasets
        self.source_keys = source_keys

        self._lengths: list[int] = []
        self._offsets: list[int] = [0]
        self.audio_files: list[str] = []

        for dataset in self.source_datasets:
            if not isinstance(dataset, Sized):
                raise TypeError("Each source dataset must implement __len__().")

            dataset_len = len(cast(Sized, dataset))
            self._lengths.append(dataset_len)
            self._offsets.append(self._offsets[-1] + dataset_len)

            dataset_audio_files = getattr(dataset, "audio_files", None)
            if dataset_audio_files is None:
                raise AttributeError("Each source dataset must expose 'audio_files'.")
            self.audio_files.extend([str(path) for path in dataset_audio_files])

    def __len__(self) -> int:
        return self._offsets[-1]

    def _find_source_index(self, index: int) -> int:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index out of range: {index}")
        return bisect.bisect_right(self._offsets, index) - 1

    def get_audio_path_for_index(self, index: int) -> str:
        source_idx = self._find_source_index(index)
        local_index = index - self._offsets[source_idx]
        audio_files = getattr(self.source_datasets[source_idx], "audio_files", None)
        if audio_files is None or len(audio_files) == 0:
            raise RuntimeError("Source dataset has no audio_files for index mapping.")
        return str(audio_files[local_index % len(audio_files)])

    def __getitem__(self, index: int) -> object:
        source_idx = self._find_source_index(index)
        local_index = index - self._offsets[source_idx]
        return self.source_datasets[source_idx][local_index]
