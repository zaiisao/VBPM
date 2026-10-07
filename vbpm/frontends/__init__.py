"""Frontends: Beat This spectrogram windows -> per-frame features for VBPM."""


class FrontendBase:
    """Interface: [B, T, 128] spectrogram windows -> [B, T', num_channels] at output_fps."""

    num_channels: int
    output_fps: int

    def forward_features(self, batch):
        """Spectrogram windows -> features."""
        raise NotImplementedError

    def prediction_head_parameters(self):
        """Return beat/downbeat linear-head weights and bias for audio proposals."""
        raise NotImplementedError("This frontend does not provide a beat/downbeat proposal head")


Frontend = FrontendBase


from .beat_this import BeatThisFrontend
from .musicfm import MusicFMFrontend

_FRONTENDS = {
    "beat_this": BeatThisFrontend,
    "musicfm": MusicFMFrontend,
}


def build_frontend(name: str, **kwargs):
    """Instantiate a configured frontend by its short name."""
    try:
        frontend_type = _FRONTENDS[name]
    except KeyError as error:
        choices = ", ".join(sorted(_FRONTENDS))
        raise ValueError(f"unknown frontend {name!r}; choose one of: {choices}") from error
    return frontend_type(**kwargs)
