"""Small, deterministic native-audio split with annotated beat/downbeat labels.

Latent references are annotation-derived diagnostics, never training targets.
Beat This's official fold assignment is used at the song level.
"""
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

STORE = Path('/disk1/jaehoon/dataset_store')
FPS, SAMPLE_RATE = 25, 24000
FRAMES, CONTEXT_FRAMES = 750, 750
CONTEXT_SECONDS = CONTEXT_FRAMES / FPS
OFFSET_FRAMES = (CONTEXT_FRAMES - FRAMES) // 2
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs/30s'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rasterize(beat_times, downbeat_times, start_frame, frames=FRAMES):
    labels = np.zeros(frames, dtype=np.int64)
    selected = {}
    for name, times, value in (('beats', beat_times, 1), ('downbeats', downbeat_times, 2)):
        indices = np.rint(times * FPS).astype(np.int64) - start_frame
        keep = (indices >= 0) & (indices < frames)
        chosen = indices[keep]
        if len(np.unique(chosen)) != len(chosen):
            raise ValueError(f'{name}: multiple annotations quantize to the same frame')
        labels[chosen] = value  # downbeat overwrites the coincident beat
        selected[name] = dict(indices=chosen, times=times[keep] - start_frame / FPS)
    return labels, selected


def annotation_reference(beat_annotations, frame_times):
    """Linear bar phase and average bar velocity; not observed continuous truth."""
    downbeats = beat_annotations[beat_annotations[:, 1] == 1, 0]
    bar = np.searchsorted(downbeats, frame_times, side='right') - 1
    valid = (bar >= 0) & (bar + 1 < len(downbeats))
    # A recording can begin/end inside a bar. Do not invent downbeats or
    # extrapolate a physical reference there. Zero placeholders are masked
    # in diagnostics; the full label sequence remains a training target.
    phase = np.zeros(len(frame_times), dtype=np.float32)
    velocity = np.zeros(len(frame_times), dtype=np.float32)
    meters = np.zeros(len(frame_times), dtype=np.int64)
    left, right = downbeats[bar[valid]], downbeats[bar[valid] + 1]
    phase[valid] = 2 * np.pi * (frame_times[valid] - left) / (right - left)
    velocity[valid] = 2 * np.pi / (right - left) / FPS
    for index in np.unique(bar[valid]):
        count = int(((beat_annotations[:, 0] >= downbeats[index]) &
                     (beat_annotations[:, 0] < downbeats[index + 1])).sum())
        if count not in (2, 3, 4):
            raise ValueError(f'Unsupported or inconsistent annotated bar length: {count}')
        meters[(bar == index) & valid] = count
    return phase, velocity, meters, valid


def select_clips(per_meter=2, fold=7, store=STORE):
    annotation_dir = store / 'beat_this_annotations/ballroom'
    audio_dir = store / 'audio_by_stem/ballroom'
    counts = {(split, meter): 0 for split in ('training', 'heldout') for meter in (3, 4)}
    clips = []
    candidates = sorted(line.split('\t') for line in (annotation_dir / '8-folds.split').read_text().splitlines())
    for stem, part in candidates:
        split = 'heldout' if int(part) == fold else 'training'
        path = audio_dir / (stem + '.wav')
        annotation_path = annotation_dir / 'annotations/beats' / (stem + '.beats')
        if not path.exists():
            continue
        annotations = np.loadtxt(annotation_path)
        if annotations.ndim != 2 or len(annotations) < 20:
            continue
        if not (np.diff(annotations[:, 0]) > 0).all():
            continue
        meter = int(annotations[:, 1].max())
        if meter not in (3, 4) or counts[(split, meter)] >= per_meter:
            continue
        downbeats = annotations[annotations[:, 1] == 1, 0]
        if len(downbeats) < 4:
            continue
        info = sf.info(path)
        # Full native-audio context, chosen deterministically near the middle
        # of the annotated region. Crop times are integer MusicFM frames.
        midpoint = (downbeats[0] + downbeats[-1]) / 2
        if info.duration < CONTEXT_SECONDS:
            continue
        start_seconds = np.clip(midpoint - CONTEXT_SECONDS / 2,
                                0, info.duration - CONTEXT_SECONDS)
        start_frame = int(np.floor(start_seconds * FPS))
        target_start = start_frame + OFFSET_FRAMES
        times = (target_start + np.arange(FRAMES)) / FPS
        try:
            phase, velocity, meters, valid = annotation_reference(annotations, times)
            labels, events = rasterize(annotations[:, 0], downbeats, target_start)
        except ValueError:
            continue
        if valid.mean() < .8 or not (meters[valid] == meter).all() or len(events['downbeats']['indices']) < 2:
            continue
        clips.append(dict(split=split, song_id='ballroom/' + stem, meter=meter,
            fold=int(part), audio_path=str(path), audio_resolved=str(path.resolve()),
            audio_sha256=sha256(path), annotation_path=str(annotation_path),
            annotation_sha256=sha256(annotation_path), native_sample_rate=info.samplerate,
            native_channels=info.channels, duration_seconds=info.duration,
            context_start_frame=start_frame, target_start_frame=target_start,
            context_start_seconds=start_frame / FPS, target_start_seconds=target_start / FPS,
            context_seconds=CONTEXT_SECONDS, target_seconds=FRAMES / FPS,
            reference_valid_frames=int(valid.sum()),
            b=torch.from_numpy(labels), phase_reference=torch.from_numpy(phase),
            velocity_reference=torch.from_numpy(velocity), meter_reference=torch.from_numpy(meters),
            reference_valid=torch.from_numpy(valid),
            beat_times=events['beats']['times'].tolist(),
            downbeat_times=events['downbeats']['times'].tolist(),
            beat_indices=events['beats']['indices'].tolist(),
            downbeat_indices=events['downbeats']['indices'].tolist()))
        counts[(split, meter)] += 1
        if all(value == per_meter for value in counts.values()):
            break
    if not all(value == per_meter for value in counts.values()):
        raise RuntimeError(f'Insufficient native-audio clips with stable meter: {counts}')
    return clips


def load_waveform(clip):
    sr = clip['native_sample_rate']
    start = int(round(clip['context_start_seconds'] * sr))
    count = int(round(CONTEXT_SECONDS * sr))
    wav, actual_sr = sf.read(clip['audio_path'], start=start, frames=count,
                             dtype='float32', always_2d=True)
    if actual_sr != sr or len(wav) != count:
        raise RuntimeError('Audio crop length/sample rate differs from the manifest')
    mono = torch.from_numpy(wav.mean(axis=1))
    mono = AF.resample(mono, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else mono
    if len(mono) != SAMPLE_RATE * CONTEXT_SECONDS or not torch.isfinite(mono).all():
        raise RuntimeError('Invalid resampled MusicFM waveform')
    return mono.unsqueeze(0)


def stack_split(clips, features, split):
    selected = [i for i, clip in enumerate(clips) if clip['split'] == split]
    return dict(x=torch.stack([features[i] for i in selected]),
        b=torch.stack([clips[i]['b'] for i in selected]),
        phase_reference=torch.stack([clips[i]['phase_reference'] for i in selected]),
        velocity_reference=torch.stack([clips[i]['velocity_reference'] for i in selected]),
        meter_reference=torch.stack([clips[i]['meter_reference'] for i in selected]),
        reference_valid=torch.stack([clips[i]['reference_valid'] for i in selected]),
        clips=[{key: value for key, value in clips[i].items() if not torch.is_tensor(value)} for i in selected])
