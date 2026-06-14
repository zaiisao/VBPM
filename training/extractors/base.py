"""Extractor backend interfaces for modular end-to-end CHART training."""

from __future__ import annotations

import argparse
from typing import Protocol

import torch
from torch.utils.data import DataLoader


class ExtractorBackend(Protocol):
    """Protocol implemented by extractor-specific training backends."""

    name: str

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        """Register extractor-specific CLI options."""

    def build_dataloader(self, args: argparse.Namespace) -> DataLoader:
        """Build training dataloader that yields audio/target + CHART phase tensors."""

    def build_val_dataloader(self, args: argparse.Namespace) -> DataLoader | None:
        """Build validation dataloader. Returns None if no validation split is available."""

    def build_model(self, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
        """Build extractor model used for end-to-end training."""

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        args: argparse.Namespace,
        device: torch.device,
    ) -> None:
        """Load optional extractor checkpoint into the model."""

    def compute_loss_and_activations(
        self,
        model: torch.nn.Module,
        audio: torch.Tensor,
        target: torch.Tensor,
        frozen: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(extractor_loss, activations)` for CHART training.

        When ``frozen`` is True, the extractor forward runs under no_grad and
        the loss is returned as a detached zero (purely a logging placeholder).
        """
