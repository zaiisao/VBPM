"""Frozen native-waveform MusicFM, concatenating all 12 Conformer layers."""
import importlib
import json
import sys
from pathlib import Path

import torch

from data import (
    FPS, FRAMES, CONTEXT_FRAMES, OFFSET_FRAMES, SAMPLE_RATE, OUTPUT_DIR, STORE,
    load_waveform, select_clips, sha256, stack_split)

HERE = Path(__file__).resolve().parent
VENDOR = HERE / 'vendor'


def prepare(device='cuda:0', *, weights=None, statistics=None, store=STORE, output=OUTPUT_DIR):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(VENDOR))
    module = importlib.import_module('musicfm.model.musicfm_25hz')
    data_dir = HERE / 'assets/musicfm'
    weights = Path(weights) if weights is not None else data_dir / 'pretrained_msd.pt'
    statistics = Path(statistics) if statistics is not None else data_dir / 'msd_stats.json'
    if not weights.is_file() or not statistics.is_file():
        raise FileNotFoundError('Supply MusicFM MSD weights/statistics using the CLI or assets/musicfm/')
    model = module.MusicFM25Hz(is_flash=False, stat_path=str(statistics),
                               model_path=str(weights)).to(device).eval().requires_grad_(False)
    clips = select_clips(store=Path(store))
    features, audits = [], []
    with torch.no_grad():
        for index, clip in enumerate(clips):
            wav = load_waveform(clip).to(device)
            _, states = model.get_predictions(wav)
            if len(states) != 13 or not all(tuple(s.shape) == (1, CONTEXT_FRAMES, 1024) for s in states):
                raise RuntimeError(f'Unexpected MusicFM hidden-state layout: {[s.shape for s in states]}')
            joined = torch.cat(states[1:], dim=-1)
            exact = all(torch.equal(joined[..., layer * 1024:(layer + 1) * 1024], states[layer + 1])
                        for layer in range(12))
            if not exact or not torch.isfinite(joined).all() or joined.requires_grad:
                raise RuntimeError('Invalid layer concatenation or accidentally trainable MusicFM')
            cropped = joined[0, OFFSET_FRAMES:OFFSET_FRAMES + FRAMES].float().cpu().contiguous()
            features.append(cropped)
            audits.append(dict(song_id=clip['song_id'], split=clip['split'],
                meter=clip['meter'], full_hidden_shape=list(states[-1].shape),
                feature_shape=list(cropped.shape), exact_layer_concatenation=exact,
                waveform_shape=list(wav.shape), finite=True,
                layer_norms=[float(s.float().square().mean().sqrt()) for s in states[1:]]))
            print('EXTRACTED', index + 1, len(clips), clip['split'], clip['song_id'],
                  'meter', clip['meter'], tuple(cropped.shape), flush=True)
    cached = {split: stack_split(clips, features, split) for split in ('training', 'heldout')}
    torch.save(cached, output / 'features.pt')
    metadata = dict(model='MusicFM MSD', weights=str(weights.resolve()), weights_sha256=sha256(weights),
        statistics=str(statistics.resolve()), statistics_sha256=sha256(statistics),
        musicfm_source=str(Path(module.__file__).resolve()), source_sha256=sha256(module.__file__),
        native_waveform=True, sample_rate=SAMPLE_RATE, feature_fps=FPS, layers=list(range(1, 13)),
        excluded_state=0, channels_per_layer=1024, concatenated_channels=12288,
        frames=FRAMES, context_frames=CONTEXT_FRAMES, seconds_per_clip=FRAMES / FPS,
        context_seconds=CONTEXT_FRAMES / FPS, feature_dtype='float32',
        frozen=True, eval_mode=True, cache_sha256=sha256(output / 'features.pt'),
        annotation_split='Official Beat This Ballroom 8-fold split, validation fold 7',
        annotation_reference='Linear phase between downbeats; bar-average radians/frame; evaluation only; unbracketed edge frames masked',
        audits=audits, clips=[{key: value for key, value in c.items() if not torch.is_tensor(value)} for c in clips])
    (output / 'feature_manifest.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    del model
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return cached, metadata
