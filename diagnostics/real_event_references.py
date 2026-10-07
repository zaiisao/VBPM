"""Exact annotation timestamps for scoring cached real windows, not model input."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def event_references(batch):
    """Read beat and downbeat references for cached excerpts."""
    destination = ROOT / "runs/generator_isolation/event_references.json"
    if destination.exists():
        result = json.loads(destination.read_text())
        assert result["songs"] == batch["songs"]
        assert result["start_frames"] == [r["start_frame"] for r in batch["oracle_checks"]]
        return result
    sys.path.insert(0, str(ROOT / "external/beat_this"))
    from vbpm.data.dataset import load_beat_this

    source = load_beat_this(7).train_dataset
    by_id = {str(s["spect_path"]): s for s in source.items}
    beats, downbeats = [], []
    frames = batch["labels"].shape[1]
    for i, sid in enumerate(batch["songs"]):
        song = by_id[sid]
        start = batch["oracle_checks"][i]["start_frame"]
        all_beats = np.asarray(song["beat_time"], dtype=np.float64)
        all_down = all_beats[np.asarray(song["beat_value"]) == 1]
        expected = np.zeros(frames, dtype=np.int64)
        window = []
        for times, label in ((all_beats, 1), (all_down, 2)):
            indices = np.rint(times * 50).astype(int) - start
            valid = (indices >= 0) & (indices < frames)
            expected[indices[valid]] = label
            window.append((times[valid] - start / 50).tolist())
        assert np.array_equal(expected, batch["labels"][i].numpy()), sid
        beats.append(window[0])
        downbeats.append(window[1])
    result = dict(
        songs=batch["songs"],
        start_frames=[r["start_frame"] for r in batch["oracle_checks"]],
        fps=50,
        beat_times=beats,
        downbeat_times=downbeats,
        use=(
            "Exact original annotations for scoring only; labels independently"
            " matched to cached batch"
        ),
    )
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def attach_references(data, batch):
    """Attach annotation events for scoring only."""
    references = event_references(batch)
    data["beat_times"] = references["beat_times"]
    data["downbeat_times"] = references["downbeat_times"]
    return data


def interbeat_tempo_errors(phase, beat_times, fps=50, meter=4):
    """Compare mean latent motion over observed beat intervals with their BPM.

    Annotations identify these averages, but not instantaneous within-beat
    motion. The caller must keep instantaneous interpolation scores separate.
    """
    estimates, references = [], []
    timeline = np.arange(phase.shape[1], dtype=np.float64) / fps
    for i, times in enumerate(beat_times):
        times = np.asarray(times, dtype=np.float64)
        times = times[(times >= timeline[0]) & (times <= timeline[-1])]
        if len(times) < 2:
            continue
        positions = np.interp(times, timeline, phase[i].detach().cpu().numpy())
        duration = np.diff(times)
        estimates.extend((np.diff(positions) / duration * meter * 60 / (2 * np.pi)).tolist())
        references.extend((60 / duration).tolist())
    errors = np.asarray(estimates) - np.asarray(references)
    return dict(
        interbeat_tempo_RMSE_BPM=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        interbeat_tempo_max_error_BPM=float(np.max(np.abs(errors))) if len(errors) else None,
        interbeat_interval_count=len(errors),
        predicted_interbeat_BPM=estimates,
        reference_interbeat_BPM=references,
    )
