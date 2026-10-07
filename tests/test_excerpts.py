"""The restored spectrogram-source API: windows, labels, padding and collation."""

from pathlib import Path
import numpy as np
import pytest

from vbpm.data.excerpts import ExcerptDataset, collate_excerpts


class Frontend:
    output_fps = 50


class Source:
    fps = 100

    def __init__(self, seconds=70.0, songs=1, sparse=False):
        downbeats = np.array([1.0, 3.0]) if sparse else np.arange(0.0, seconds, 2.0)
        self.items = [
            {
                "spect_path": Path(f"toy/song{i}"),
                "dataset": "toy",
                "beat_time": downbeats,
                "beat_value": np.ones(len(downbeats)),
            }
            for i in range(songs)
        ]
        self.spect = np.arange(int(seconds * self.fps), dtype=np.float32)[:, None].repeat(4, 1)

    def _get_spect(self, _song):
        return self.spect


def test_window_is_contiguous_and_uses_spectrogram_frame_rate():
    source = Source()
    ds = ExcerptDataset(source, Frontend(), excerpt_seconds=45.0)
    item = ds[0]
    start = int(round(float(item["t0"]) * source.fps))
    np.testing.assert_array_equal(item["input"], source.spect[start : start + 4500])
    assert item["mask"].all() and item["cls"].shape == (2250,)


def test_centered_windows_are_stable():
    ds = ExcerptDataset(Source(), Frontend(), excerpt_seconds=45.0, centered=True)
    assert float(ds[0]["t0"]) == 12.5
    assert float(ds[0]["t0"]) == 12.5


def test_random_windows_are_fresh():
    ds = ExcerptDataset(Source(), Frontend())
    assert len({float(ds[0]["t0"]) for _ in range(20)}) > 1


def test_short_song_is_padded_at_both_frame_rates():
    ds = ExcerptDataset(Source(seconds=30.0), Frontend())
    item = ds[0]
    assert item["input"].shape == (4500, 4)
    assert item["mask"][:1500].all() and not item["mask"][1500:].any()
    assert not item["cls"][1500:].any() and np.all(item["input"][3000:] == 0)


def test_downbeats_overwrite_beats_at_the_annotation_frame():
    ds = ExcerptDataset(Source(), Frontend(), centered=True)
    item = ds[0]
    for t in item["downbeat_times"]:
        centre = int(round(t * Frontend.output_fps)) - int(
            round(float(item["t0"]) * Frontend.output_fps)
        )
        assert item["cls"][centre] == 2


def test_sparse_annotations_are_rejected():
    ds = ExcerptDataset(Source(sparse=True), Frontend())
    assert len(ds) == 0 and ds.rejects == ["song0"]


def test_collate_stacks_and_preserves_annotation_lists():
    ds = ExcerptDataset(Source(songs=3), Frontend())
    batch = collate_excerpts([ds[i] for i in range(3)])
    assert batch["input"].shape == (3, 4500, 4)
    assert batch["cls"].shape == batch["mask"].shape == (3, 2250)
    assert len(batch["downbeat_times"]) == len(batch["dataset"]) == 3


def test_collate_removes_empty_windows_and_rejects_an_empty_batch():
    ds = ExcerptDataset(Source(), Frontend())
    item, empty = ds[0], ds[0]
    empty["mask"][:] = 0
    assert collate_excerpts([item, empty])["cls"].shape[0] == 1
    with pytest.raises(ValueError, match="every item"):
        collate_excerpts([empty])
