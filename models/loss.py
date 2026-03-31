"""ELBO loss for the variational bar pointer model.

Implements the complete loss function:

    L = Reconstruction (Gaussian log-likelihood / MSE on smoothed targets)
      + KL_meter(Categorical)
      + KL_phase(von Mises)
      + KL_tempo(Log-Normal)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from models.distributions import categorical_kl, lognormal_kl, von_mises_kl


def gaussian_smooth_targets(
    targets: Tensor,
    sigma: float = 3.0,
) -> Tensor:
    """Smooth binary spike targets with a Gaussian kernel.

    Converts binary beat/downbeat indicators [B, T] into soft activation
    curves, spreading each spike into a Gaussian bump. This serves as the
    observation for the Gaussian reconstruction term in the ELBO.

    Args:
        targets: [B, T] binary indicators.
        sigma: Standard deviation of Gaussian kernel in frames.

    Returns:
        Smoothed targets [B, T] in [0, 1].
    """
    if sigma <= 0:
        return targets

    # Build 1D Gaussian kernel
    radius = int(math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=targets.dtype, device=targets.device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.max()  # normalize peak to 1 (not area)
    kernel = kernel.view(1, 1, -1)  # [1, 1, kernel_size]

    # Convolve each sample: [B, T] → [B, 1, T] → conv1d → [B, T]
    t = targets.unsqueeze(1)  # [B, 1, T]
    smoothed = F.conv1d(t, kernel, padding=radius)
    smoothed = smoothed.squeeze(1).clamp(0, 1)  # [B, T]

    return smoothed


def compute_elbo_loss(
    beat_logits: Tensor,
    beat_targets: Tensor,
    posterior: dict[str, Tensor],
    prior: dict[str, Tensor],
    beta: float = 1.0,
    pos_weight: float = 20.0,
    free_bits: float = 0.0,
    free_bits_meter: float | None = None,
    free_bits_phase: float | None = None,
    free_bits_tempo: float | None = None,
    downbeat_targets: Tensor | None = None,
    smooth_sigma: float = 3.0,
    smooth_sigma_db: float = 5.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the negative ELBO loss.

    Reconstruction uses MSE against Gaussian-smoothed targets (Gaussian
    observation model), maintaining a valid ELBO while naturally handling
    class imbalance by spreading positive mass over multiple frames.

    Args:
        beat_logits: ``[B, T, 2]`` decoder output (raw, pre-activation).
            Channel 0: beat, Channel 1: downbeat.
        beat_targets: ``[B, T]`` binary beat indicators.
        posterior, prior: Distribution parameter dicts.
        beta: KL weight.
        pos_weight: Unused (kept for API compat). Imbalance handled by smoothing.
        free_bits*: Per-latent free bits overrides.
        downbeat_targets: ``[B, T]`` binary downbeat indicators.
        smooth_sigma: Gaussian kernel sigma for target smoothing (frames).

    Returns:
        Tuple of (total_loss, component_dict).
    """
    fb_meter = free_bits_meter if free_bits_meter is not None else free_bits
    fb_phase = free_bits_phase if free_bits_phase is not None else free_bits
    fb_tempo = free_bits_tempo if free_bits_tempo is not None else free_bits

    # --- Reconstruction: BCE on Gaussian-smoothed targets ---
    # Smoothed targets b̃_t ∈ [0,1] spread positive mass over multiple
    # frames, preventing the all-zeros degenerate solution while keeping
    # the loss in nats (naturally balanced with KL).
    beat_smooth = gaussian_smooth_targets(beat_targets, sigma=smooth_sigma)
    recon_beat = F.binary_cross_entropy_with_logits(
        beat_logits[:, :, 0], beat_smooth, reduction="mean",
    )

    if downbeat_targets is not None:
        db_smooth = gaussian_smooth_targets(downbeat_targets, sigma=smooth_sigma_db)
        recon_db = F.binary_cross_entropy_with_logits(
            beat_logits[:, :, 1], db_smooth, reduction="mean",
        )
    else:
        recon_db = torch.tensor(0.0, device=beat_logits.device)

    bce = recon_beat + recon_db

    def _kl_with_free_bits(kl: Tensor, fb: float) -> Tensor:
        # kl: [B, T] — mean over T per sample, clamp, then mean over B
        if fb > 0.0:
            return kl.mean(dim=-1).clamp(min=fb).mean()
        return kl.mean()

    # --- KL: Meter (Categorical) ---
    kl_m = _kl_with_free_bits(categorical_kl(
        posterior["meter_logits"],
        prior["meter_logits"],
    ), fb_meter)

    # --- KL: Phase (von Mises) ---
    kappa_q = posterior["phase_log_kappa"].exp()
    kl_phi = _kl_with_free_bits(von_mises_kl(
        posterior["phase_mu"],
        kappa_q,
        prior["phase_mu"],
        prior["phase_kappa"],
    ), fb_phase)

    # --- KL: Tempo (Log-Normal / Gaussian in log-space) ---
    sigma_q = posterior["tempo_log_sigma"].exp()
    kl_tempo = _kl_with_free_bits(lognormal_kl(
        posterior["tempo_mu"],
        sigma_q,
        prior["tempo_mu"],
        prior["tempo_sigma"],
    ), fb_tempo)

    # --- Total ---
    total = bce + beta * (kl_m + kl_phi + kl_tempo)

    components = {
        "bce": bce.detach(),
        "kl_meter": kl_m.detach(),
        "kl_phase": kl_phi.detach(),
        "kl_tempo": kl_tempo.detach(),
    }

    return total, components
