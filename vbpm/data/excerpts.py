"""Frontend-agnostic training excerpts: the shared shape of both official pipelines."""

from __future__ import annotations

import numpy as np
import torch

MIN_DOWNBEATS = 4


class ExcerptDataset(torch.utils.data.Dataset):
    """Per-epoch random windows of cached frontend input + framewise VAE targets."""

    def __init__(self, source, frontend, excerpt_seconds: float = 45.0, centered: bool = False):
        self.output_fps = frontend.output_fps
        self.spect_fps = source.fps
        self.excerpt_frames = int(round(excerpt_seconds * self.output_fps))
        self.spect_excerpt_frames = int(round(excerpt_seconds * self.spect_fps))
        self.centered = centered
        self.source = source
        self.items, self.rejects = self._annotated_songs(source)

    @staticmethod
    def _annotated_songs(source):
        items, rejects = [], []

        for song in source.items:
            beat_times = song["beat_time"]
            downbeat_times = beat_times[song["beat_value"] == 1]
            song_id = song["spect_path"].parts[1]

            beat_times = np.asarray(beat_times, dtype=np.float64)
            downbeat_times = np.asarray(downbeat_times, dtype=np.float64)

            if len(downbeat_times) < MIN_DOWNBEATS:
                rejects.append(song_id)
                continue

            items.append((song, downbeat_times, beat_times))

        return items, rejects

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        song, downbeat_times, beat_times = self.items[index]
        spect = self.source._get_spect(song)

        song_frames = int(len(spect) * self.output_fps / self.spect_fps)
        window_frames = min(self.excerpt_frames, song_frames)
        spare_frames = song_frames - window_frames

        # Fresh random window per call (Beat This's policy); val/test take the middle
        # so every scored window is identical across runs.
        start = spare_frames // 2 if self.centered else int(np.random.randint(0, spare_frames + 1))
        spect_window = self._spect_window(spect, start, window_frames)
        t0 = np.float32(start / self.output_fps)

        targets = self._targets(downbeat_times, beat_times, start, window_frames)

        annotated = len(targets["downbeat_times"]) > 0 or len(targets["beat_times"]) > 0
        frame_mask = np.full(window_frames, float(annotated), dtype=np.float32)

        pad = self.excerpt_frames - window_frames
        if pad > 0:  # song shorter than the window
            targets["cls"] = np.pad(targets["cls"], (0, pad))
            frame_mask = np.pad(frame_mask, (0, pad))

        return {
            "input": spect_window,
            "cls": targets["cls"],
            "mask": frame_mask,
            "t0": t0,
            "beat_times": targets["beat_times"],
            "downbeat_times": targets["downbeat_times"],
            "dataset": song["dataset"],
        }

    def _spect_window(self, spect, start: int, window_frames: int):
        spect_start = int(round(start * self.spect_fps / self.output_fps))
        spect_frames = int(round(window_frames * self.spect_fps / self.output_fps))
        spect_window = np.array(spect[spect_start : spect_start + spect_frames], dtype=np.float32)

        spect_pad = self.spect_excerpt_frames - len(spect_window)
        if spect_pad > 0:
            spect_window = np.pad(
                spect_window, [(0, spect_pad)] + [(0, 0)] * (spect_window.ndim - 1)
            )

        return spect_window

    def _targets(self, downbeat_times, beat_times, start: int, frames: int):
        """build_crop's target math on a [start, start+frames) window, or None.

        ``cls`` is the three-way label the tutorial's emission reads, 0 = non-beat,
        1 = beat, 2 = downbeat, written downbeat-last so a downbeat overwrites the beat
        that shares its time.
        """

        def in_window(times):
            centers = np.round(times * self.output_fps) - start
            return times[(centers >= 0) & (centers < frames)]

        window_beats = in_window(beat_times)
        window_downbeats = in_window(downbeat_times)

        cls = np.zeros(frames, dtype=np.int64)
        for times, label in ((window_beats, 1), (window_downbeats, 2)):
            for t in times:
                center = int(round(t * self.output_fps)) - start
                if center < frames:
                    cls[center] = label

        return {"cls": cls, "beat_times": window_beats, "downbeat_times": window_downbeats}


def collate_excerpts(batch: list) -> dict:
    """Drop windows with no annotations; stack fixed-size fields, list the rest."""
    batch = [item for item in batch if float(item["mask"].sum()) > 0]
    if not batch:
        raise ValueError("collate_excerpts: every item in the batch was empty")

    out = {}
    for key in ("input", "cls", "mask"):
        out[key] = torch.from_numpy(np.stack([item[key] for item in batch]))

    out["t0"] = torch.tensor([item["t0"] for item in batch])
    for key in ("downbeat_times", "beat_times", "dataset"):
        out[key] = [item[key] for item in batch]

    return out
