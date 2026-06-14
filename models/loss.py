"""ELBO loss for the variational bar-pointer model.

Faithful to ELBO_for_DBN.pdf §5:

    L = sum_t -[b_t log b̂_t + (1-b_t) log(1-b̂_t)]   (Bernoulli reconstruction)
      + sum_t KL(q_phi(m_t) || p_psi(m_t))            (Categorical)
      + sum_t KL(q_phi(φ_t) || p_psi(φ_t))            (von Mises)
      + sum_t KL(q_phi(φ̇_t) || p_psi(φ̇_t))           (Log-Normal)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from models.distributions import categorical_kl, lognormal_kl, von_mises_kl


def compute_elbo_loss(
    beat_logits: Tensor,
    beat_targets: Tensor,
    posterior: dict[str, Tensor],
    prior: dict[str, Tensor],
    beta: float = 1.0,
    pos_weight: float = 1.0,
    pos_weight_db: float | None = None,
    free_bits: float = 0.0,
    free_bits_meter: float | None = None,
    free_bits_phase: float | None = None,
    free_bits_tempo: float | None = None,
    downbeat_targets: Tensor | None = None,
    # legacy kwargs (ignored, kept for API compatibility during transition)
    smooth_sigma: float | None = None,
    smooth_sigma_db: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Negative ELBO with strict Bernoulli reconstruction (PDF §5.4).

    Args:
        beat_logits: ``[B, T, 2]`` decoder logits (channel 0 beat, 1 downbeat).
        beat_targets: ``[B, T]`` binary {0,1}.
        posterior, prior: per-step distribution-parameter dicts.
        beta: KL weight (1.0 == ELBO).
        pos_weight: BCE positive-class weight; default 1.0 (strict Bernoulli).
        free_bits*: per-latent free-bits floor in nats (set to 0 to disable).
        downbeat_targets: ``[B, T]`` binary {0,1} (or None).
    """
    del smooth_sigma, smooth_sigma_db  # ignored; smoothing removed

    fb_meter = free_bits_meter if free_bits_meter is not None else free_bits
    fb_phase = free_bits_phase if free_bits_phase is not None else free_bits
    fb_tempo = free_bits_tempo if free_bits_tempo is not None else free_bits

    pw = torch.tensor(pos_weight, device=beat_logits.device) if pos_weight != 1.0 else None
    pw_db_val = pos_weight_db if pos_weight_db is not None else pos_weight
    pw_db = torch.tensor(pw_db_val, device=beat_logits.device) if pw_db_val != 1.0 else None

    # ---- Reconstruction: Bernoulli BCE on raw {0, 1} targets ----
    recon_beat = F.binary_cross_entropy_with_logits(
        beat_logits[:, :, 0], beat_targets, pos_weight=pw, reduction="mean",
    )
    if downbeat_targets is not None:
        recon_db = F.binary_cross_entropy_with_logits(
            beat_logits[:, :, 1], downbeat_targets, pos_weight=pw_db, reduction="mean",
        )
    else:
        recon_db = torch.tensor(0.0, device=beat_logits.device)
    bce = recon_beat + recon_db

    def _kl_with_free_bits(kl: Tensor, fb: float) -> Tensor:
        if fb > 0.0:
            return kl.mean(dim=-1).clamp(min=fb).mean()
        return kl.mean()

    # ---- KL: meter (Categorical) ----
    kl_m = _kl_with_free_bits(
        categorical_kl(posterior["meter_logits"], prior["meter_logits"]),
        fb_meter,
    )

    # ---- KL: phase (von Mises) ----
    kappa_q = posterior["phase_log_kappa"].exp()
    kl_phi = _kl_with_free_bits(
        von_mises_kl(
            posterior["phase_mu"], kappa_q,
            prior["phase_mu"], prior["phase_kappa"],
        ),
        fb_phase,
    )

    # ---- KL: tempo (Log-Normal in log-space) ----
    sigma_q = posterior["tempo_log_sigma"].exp()
    kl_tempo = _kl_with_free_bits(
        lognormal_kl(
            posterior["tempo_mu"], sigma_q,
            prior["tempo_mu"], prior["tempo_sigma"],
        ),
        fb_tempo,
    )

    total = bce + beta * (kl_m + kl_phi + kl_tempo)

    components = {
        "bce": bce.detach(),
        "kl_meter": kl_m.detach(),
        "kl_phase": kl_phi.detach(),
        "kl_tempo": kl_tempo.detach(),
    }
    return total, components
