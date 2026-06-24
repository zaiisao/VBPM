"""Real-data adapter: raw audio -> fixed log-mel spectrogram + binary beat targets.

This is what makes the run END-TO-END FROM RANDOM WEIGHTS: the observation ``h`` fed to
the model is a *fixed, non-learned* log-mel spectrogram (exactly the kind of input Beat
This / WaveBeat themselves ingest), NOT a pretrained frontend's activations. The only
learned parameters in the whole pipeline are the VAE's own, initialised randomly.

Audio + annotation loading is delegated to WaveBeat's validated ``DownbeatDataset``
(it correctly parses ballroom/.beats, beatles/.txt, hainsworth/.txt, rwc/.BEAT.TXT and
builds single-frame binary beat/downbeat targets). We only add the log-mel transform.
"""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# torchaudio >= 2.2 removed set_audio_backend, which wavebeat/data.py calls at import
# time. Install a no-op shim BEFORE importing it.
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda *a, **k: None  # type: ignore[attr-defined]

_REPO = Path(__file__).resolve().parent.parent
_WB_DATA = _REPO / "extractors" / "wavebeat" / "wavebeat" / "data.py"
_spec = importlib.util.spec_from_file_location("_wb_data", _WB_DATA)
_wb = importlib.util.module_from_spec(_spec)            # type: ignore[arg-type]
_spec.loader.exec_module(_wb)                           # type: ignore[union-attr]
DownbeatDataset = _wb.DownbeatDataset

# Project-standard audio recipe: 22.05 kHz, hop 256 -> 86.13 fps (matches the rest of CHART).
SR = 22050
HOP = 256
N_FFT = 1024
N_MELS = 128
FPS = SR / HOP

# data dir name (on disk) -> the dataset name WaveBeat's loader expects
_DATASETS = {
    "ballroom": "ballroom",
    "beatles": "beatles",
    "hains": "hainsworth",
    "rwc_popular": "rwc_popular",
}


class LogMel(nn.Module):
    """Fixed log-mel front-end. NOT learned -- just the input representation."""

    def __init__(self, sr=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=1.0)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """[B, N] waveform -> [B, T, n_mels] log-mel."""
        m = self.mel(audio)                 # [B, n_mels, T]
        m = torch.log1p(m)                  # log compression
        return m.transpose(1, 2)            # [B, T, n_mels]


def _build_base(root: Path, key: str, subset: str, frames: int,
                examples_per_epoch: int, seed: int = 42) -> "DownbeatDataset":
    random.seed(seed)  # reproducible 80/10/10 split (DownbeatDataset shuffles via global RNG)
    return DownbeatDataset(
        audio_dir=str(root / key / "data"),
        annot_dir=str(root / key / "label"),
        audio_sample_rate=SR, target_factor=HOP,
        dataset=_DATASETS[key], subset=subset,
        length=frames * HOP, augment=False, half=False, preload=False,
        examples_per_epoch=examples_per_epoch,
    )


class _TrainItems(Dataset):
    """Wrap a train-subset DownbeatDataset -> (audio[N], beats[T], downbeats[T])."""

    def __init__(self, base: "DownbeatDataset"):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        audio, target = self.base[i]                 # [1, N], [2, T]
        return audio.squeeze(0), target[0], target[1]


def _collate(batch):
    audios, beats, dbs = zip(*batch)
    return torch.stack(audios), torch.stack(beats), torch.stack(dbs)


def build_train_loader(root: str | Path, keys: list[str], frames: int, batch_size: int,
                       examples_per_epoch: int = 1000, num_workers: int = 4,
                       seed: int = 42) -> DataLoader:
    root = Path(root)
    parts = [_TrainItems(_build_base(root, k, "train", frames, examples_per_epoch, seed))
             for k in keys]
    return DataLoader(ConcatDataset(parts), batch_size=batch_size, shuffle=True,
                      collate_fn=_collate, num_workers=num_workers, drop_last=True)


def iter_val_songs(root: str | Path, keys: list[str], max_per_dataset: int | None = None,
                   seed: int = 42):
    """Yield full val songs one at a time: (key, audio[N], beats[T], downbeats[T], meta)."""
    root = Path(root)
    for k in keys:
        base = _build_base(root, k, "val", frames=1, examples_per_epoch=1, seed=seed)
        n = len(base) if max_per_dataset is None else min(len(base), max_per_dataset)
        for i in range(n):
            audio, target, meta = base[i]            # full song; [1, N], [2, T]
            yield k, audio.squeeze(0), target[0], target[1], meta
