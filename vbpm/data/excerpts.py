"""Frontend-agnostic training excerpts: the shared shape of both official pipelines."""
from __future__ import annotations

import pathlib

import numpy as np
import torch

from .dataset import MIN_DOWNBEATS
from .features import atomic_save_npy

INPUT_CACHE_DIR = "/disk4/jaehoon/vbpm_input_cache"


def input_cache_path(frontend_name: str, song, cache_root: str = INPUT_CACHE_DIR):
    """<cache>/<frontend>/<dataset>/<song_id>.npy — one flat file per song."""
    return pathlib.Path(cache_root) / frontend_name / song.dataset / f"{song.song_id}.npy"


def _compute_input(frontend, song):
    """Audio file -> the frontend's model input (soundfile, mono mix, their recipe)."""
    import soundfile

    signal, sample_rate = soundfile.read(str(song.audio_path), dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    return frontend.prepare_input(signal, sample_rate)


class ExcerptDataset(torch.utils.data.Dataset):
    """Per-epoch random windows of cached frontend input + framewise VAE targets."""

    def __init__(self, songs, frontend, excerpt_seconds: float = 45.0,
                 deterministic: bool = False, cache_root: str = INPUT_CACHE_DIR,
                 target_tol_frames: int = 0, beat_only: bool = False):
        self.frontend_name = frontend.name
        self.fps = float(frontend.FPS)
        self.excerpt_frames = int(round(excerpt_seconds * self.fps))
        self.deterministic = deterministic
        self.target_tol_frames = int(target_tol_frames)
        self.cache_root = cache_root

        self.items, self.rejects = [], []
        computed = 0
        for song in songs:
            beat_times, downbeat_times = song.beats()
            downbeat_times = np.asarray(downbeat_times, dtype=np.float64)
            enough = (len(np.asarray(beat_times)) >= MIN_DOWNBEATS if beat_only
                      else len(downbeat_times) >= MIN_DOWNBEATS)
            if not enough:
                self.rejects.append(song.song_id)
                continue
            path = input_cache_path(self.frontend_name, song, cache_root)
            if not path.exists():
                try:
                    array = _compute_input(frontend, song)
                except Exception as error:  # noqa: BLE001 — one bad file costs one song
                    print(f"  input FAILED for {song.song_id}: {error!r}", flush=True)
                    self.rejects.append(song.song_id)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_save_npy(path, array)
                computed += 1
                if computed % 200 == 0:
                    print(f"  input cache: {computed} computed...", flush=True)
            self.items.append((song, downbeat_times,
                               np.asarray(beat_times, dtype=np.float64), path))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        song, downbeat_times, beat_times, path = self.items[index]
        array = np.load(path, mmap_mode="r")
        total = len(array)
        frames = min(self.excerpt_frames, total)
        longer = total - frames

        # Fresh random window per call (Beat This's policy); val/test take the middle
        # so every scored window is identical across runs.
        start = longer // 2 if self.deterministic else int(np.random.randint(0, longer + 1))
        targets = self._targets(downbeat_times, beat_times, start, frames,
                                target_tol_frames=self.target_tol_frames)

        window = np.array(array[start:start + frames], dtype=np.float32)
        labeled = (len(targets["downbeat_times"]) > 0
                   or len(targets["beat_times"]) > 0)
        mask = np.full(frames, float(labeled), dtype=np.float32)

        pad = self.excerpt_frames - frames
        if pad > 0:                                            # song shorter than the window
            window = np.pad(window, [(0, pad)] + [(0, 0)] * (window.ndim - 1))
            targets["y"] = np.pad(targets["y"], (0, pad))
            targets["cls"] = np.pad(targets["cls"], (0, pad))
            mask = np.pad(mask, (0, pad))

        return {"input": window, "y": targets["y"], "cls": targets["cls"],
                "mask": mask, "has_downbeats": np.float32(len(downbeat_times) > 0),
                "beat_times": targets["beat_times"],
                "t0": np.float32(start / self.fps), "fps": np.float32(self.fps),
                "downbeat_times": targets["downbeat_times"],
                "anchors": targets["anchors"],
                "dataset": song.dataset, "song_id": song.song_id}

    def _targets(self, downbeat_times, beat_times, start: int, frames: int,
                 target_tol_frames: int = 0):
        """build_crop's target math on a [start, start+frames) window, or None.

        ``cls`` is the three-way label the tutorial's emission reads, 0 = non-beat,
        1 = beat, 2 = downbeat, written downbeat-last so a downbeat overwrites the beat
        that shares its time.
        """
        lo_t, hi_t = start / self.fps, (start + frames) / self.fps

        inside = downbeat_times[(downbeat_times >= lo_t) & (downbeat_times <= hi_t)]

        first = np.searchsorted(downbeat_times, lo_t, side="left")
        last = np.searchsorted(downbeat_times, hi_t, side="right")
        anchors = downbeat_times[max(0, first - 1):min(len(downbeat_times), last + 1)]

        y = np.zeros(frames, dtype=np.float32)
        for t in inside:
            centre = int(round(t * self.fps)) - start
            y[max(0, centre - target_tol_frames):centre + target_tol_frames + 1] = 1.0
        cls = np.zeros(frames, dtype=np.int64)
        for times, label in ((beat_times, 1), (inside, 2)):
            for t in times[(times >= lo_t) & (times <= hi_t)]:
                centre = int(round(t * self.fps)) - start
                lo = max(0, centre - target_tol_frames)
                cls[lo:centre + target_tol_frames + 1] = label
        return {"y": y, "cls": cls,
                "downbeat_times": np.asarray(inside, dtype=np.float64),
                "beat_times": np.asarray(
                    beat_times[(beat_times >= lo_t) & (beat_times <= hi_t)],
                    dtype=np.float64),
                "anchors": np.asarray(anchors, dtype=np.float64)}


def collate_excerpts(batch: list) -> dict:
    """Fixed-size fields stack to tensors; variable/scoring fields stay python lists.

    Items with no live frames or no annotations are DROPPED. They carry no information,
    and they are not merely useless: an all-masked item still runs the anchor's circular
    mean and the interval terms, whose 0/0 backward writes nan into the SHARED encoder
    parameters while the forward loss stays finite (frames.clamp(min=1) guards the
    division only). Measured: one such window at epoch 9 of a 1661-song run killed the
    whole model -- rare because the crop is redrawn per epoch, so it never fired in
    700-song smokes.
    """
    batch = [item for item in batch
             if float(item["mask"].sum()) > 0
             and (len(item["downbeat_times"]) > 0 or len(item["beat_times"]) > 0)]
    if not batch:
        raise ValueError("collate_excerpts: every item in the batch was empty")
    out = {}
    for key in ("input", "y", "cls", "mask"):
        out[key] = torch.from_numpy(np.stack([item[key] for item in batch]))
    for key in ("t0", "fps", "has_downbeats"):
        out[key] = torch.tensor([item[key] for item in batch])
    for key in ("downbeat_times", "beat_times", "anchors", "dataset", "song_id"):
        out[key] = [item[key] for item in batch]
    return out
