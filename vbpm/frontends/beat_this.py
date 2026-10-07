"""Beat This frontend (Foscarin, Schlueter & Widmer, ISMIR 2024; external/beat_this submodule)."""

import torch

from . import FrontendBase


class BeatThisFrontend(FrontendBase):
    """Pretrained Beat This with its heads removed: 50 fps spectrograms -> transformer features."""

    output_fps = 50

    def __init__(self, checkpoint: str = "final0", device: str = "cuda"):
        from beat_this.inference import load_model

        self.device = torch.device(device)
        self.model = load_model(checkpoint, self.device)
        head = self.model.task_heads.beat_downbeat_lin
        self.prediction_weight = head.weight.detach().clone()
        self.prediction_bias = head.bias.detach().clone()
        self.model.task_heads = torch.nn.Identity()
        self.num_channels = self.model.frontend.linear.out_features

    def forward_features(self, batch) -> torch.Tensor:
        """[B, T, 128] mel windows -> [B, T, num_channels] features."""
        return self.model(batch.to(self.device))

    def prediction_head_parameters(self):
        """Return the frozen output head paired with these features."""
        return self.prediction_weight, self.prediction_bias
