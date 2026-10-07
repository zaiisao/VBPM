"""Frozen or fine-tunable MusicFM encoder frontend."""

from __future__ import annotations

from pathlib import Path

import torch

from ..util.spect_convert import (
    bt_to_mag,
    build_freq_map,
    level_offset,
    mel_bt_to_fm,
    power_to_musicfm,
    upsample_time,
)
from . import FrontendBase


class MusicFMFrontend(FrontendBase):
    """MusicFM MSD embeddings from MusicFM-native mel inputs.

    Input is the 50 fps Beat This spectrogram, converted to the 100 fps MusicFM mel.
    The final encoder representation is 1024 channels at 25 fps.
    """

    output_fps = 25

    def __init__(self, checkpoint: str | None = None, device: str = "cuda"):
        from musicfm.model import musicfm_25hz

        default = Path(musicfm_25hz.__file__).parents[1] / "data" / "pretrained_msd.pt"
        checkpoint_path = Path(checkpoint).expanduser() if checkpoint else default
        stats_path = checkpoint_path.with_name("msd_stats.json")

        self.device = torch.device(device)
        self.model = (
            musicfm_25hz.MusicFM25Hz(
                is_flash=False, stat_path=stats_path, model_path=checkpoint_path
            )
            .to(self.device)
            .eval()
        )
        self.num_channels = self.model.conv.linear.out_features

        width_bt, freq_map = build_freq_map()
        self.db_offset = level_offset(width_bt, freq_map)
        self.width_bt, self.freq_map = width_bt.to(self.device), freq_map.to(self.device)

    def forward_features(self, batch: torch.Tensor) -> torch.Tensor:
        """Beat This spectrogram [B, T@50fps, 128] -> final hidden state [B, T@25fps, 1024]."""
        bt_mel = batch.to(device=self.device, dtype=torch.float32)

        bt_mag = bt_to_mag(bt_mel.transpose(1, 2))
        fm_power = mel_bt_to_fm(bt_mag, self.width_bt, self.freq_map)
        fm_mel = upsample_time(power_to_musicfm(fm_power, self.model.stat, self.db_offset))

        _, hidden_states = self.model.encoder(fm_mel)
        return hidden_states[-1]
