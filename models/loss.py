"""ELBO loss for the variational bar-pointer model.

Faithful to ELBO_for_DBN.pdf §5:

    L = sum_t -[b_t log b̂_t + (1-b_t) log(1-b̂_t)]   (Bernoulli reconstruction)
      + sum_t KL(q_phi(m_t) || p_psi(m_t))            (Categorical)
      + sum_t KL(q_phi(φ_t) || p_psi(φ_t))            (von Mises)
      + sum_t KL(q_phi(φ̇_t) || p_psi(φ̇_t))           (Log-Normal)
"""

from __future__ import annotations

import math

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
    tempo_density_weight: float = 0.0,
    tempo_bar: dict[str, Tensor] | None = None,
    taubar_sup_weight: float = 0.0,
    meter_targets: Tensor | None = None,
    meter_sup_weight: float = 0.0,
    phase_targets: Tensor | None = None,
    phase_sup_weight: float = 0.0,
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

    # ---- KL: hierarchical global-tempo latent τ_bar (Log-Normal; exact ELBO term) ----
    # A per-sequence tempo latent whose KL is a proper 4th ELBO term. NO free-bits: we
    # WANT it informative (it must carry the clip's metrical level to break double-time).
    if tempo_bar is not None:
        kl_taubar = lognormal_kl(
            tempo_bar["mu_q"], tempo_bar["sigma_q"],
            tempo_bar["mu_p"], tempo_bar["sigma_p"],
        ).mean()
    else:
        kl_taubar = torch.zeros((), device=beat_logits.device)

    # ---- τ_bar supervision (opt-in auxiliary; weight 0 ⇒ pure ELBO) ----
    # With a weak OU anchor the unsupervised τ_bar latent collapses (inert, kl≈0). Pin
    # the POSTERIOR τ_bar mean to the GT clip log-tempo log(2π·N_beats/T) so τ_bar
    # carries the CORRECT metrical level; the OU reversion (α) then holds the free-running
    # tempo there, breaking double-time. One per-clip scalar — far milder than the failed
    # per-frame tempo-density loss, and it targets the per-clip LEVEL (what CMLt needs).
    if taubar_sup_weight > 0.0 and tempo_bar is not None:
        two_pi = 2.0 * math.pi
        n_beats = beat_targets.sum(dim=1).clamp(min=1.0)           # [B]
        T_frames = beat_targets.shape[1]
        target_log_tempo = torch.log(two_pi * n_beats / T_frames)  # [B]
        taubar_sup = ((tempo_bar["mu_q"] - target_log_tempo) ** 2).mean()
    else:
        taubar_sup = torch.zeros((), device=beat_logits.device)

    # ---- Meter supervision (Dir 3; opt-in auxiliary; weight 0 ⇒ pure ELBO) ----
    # The meter latent (beats-per-bar, which DEFINES the metrical level) collapses to
    # kl≈0 under the ELBO, leaving the model with no representation of the level → it
    # double-times. Supervise the POSTERIOR meter logits to the GT meter_class so the
    # latent carries the correct subdivision; pairs with --free_bits_meter to keep it
    # alive. Cross-entropy over per-frame logits.
    if meter_sup_weight > 0.0 and meter_targets is not None:
        K = posterior["meter_logits"].shape[-1]
        tgt_idx = meter_targets.argmax(dim=-1) if meter_targets.dim() == 3 else meter_targets.long()
        meter_sup = F.cross_entropy(
            posterior["meter_logits"].reshape(-1, K), tgt_idx.reshape(-1),
        )
    else:
        meter_sup = torch.zeros((), device=beat_logits.device)

    # ---- Phase supervision (Dir 2; opt-in auxiliary; weight 0 ⇒ pure ELBO) ----
    # The diagnostic showed the free-running failure is phase MISALIGNMENT (AMLt≈CMLt),
    # not a clean octave error. Directly align the PRIOR phase mean to the GT beat-phase
    # sawtooth so wraps land on the beats. Circular loss 1−cos(Δ) (phase is an angle).
    # Targets the prior (the free-running read-out source); pairs with scheduled sampling
    # so the alignment transfers to the free-running rollout.
    if phase_sup_weight > 0.0 and phase_targets is not None:
        pt = phase_targets.squeeze(-1) if phase_targets.dim() == 3 else phase_targets
        phase_sup = (1.0 - torch.cos(prior["phase_mu"] - pt)).mean()
    else:
        phase_sup = torch.zeros((), device=beat_logits.device)

    # ---- Tempo-density regularizer (opt-in; weight 0 ⇒ pure ELBO) ----
    # The free-running PRIOR tempo can lock to a wrong metrical level (double-time):
    # the latent-only decoder is indifferent to wrap RATE, so nothing penalises 2×.
    # A sequence with N beats over T frames should advance ~N phase cycles, i.e. the
    # mean tempo ≈ 2π·N/T rad/frame. Pin the per-sequence MEAN prior log-tempo to
    # that GT-derived target. Touches only the PRIOR tempo (not the already-correct
    # posterior), so unlike a tighter KL it cannot drag the posterior down.
    if tempo_density_weight > 0.0:
        two_pi = 2.0 * math.pi
        n_beats = beat_targets.sum(dim=1).clamp(min=1.0)            # [B]
        T = beat_targets.shape[1]
        target_log_tempo = torch.log(two_pi * n_beats / T)         # [B]
        pred_log_tempo = prior["tempo_mu"].mean(dim=1)             # [B]
        tempo_density = ((pred_log_tempo - target_log_tempo) ** 2).mean()
    else:
        tempo_density = torch.zeros((), device=beat_logits.device)

    total = (
        bce
        + beta * (kl_m + kl_phi + kl_tempo + kl_taubar)
        + tempo_density_weight * tempo_density
        + taubar_sup_weight * taubar_sup
        + meter_sup_weight * meter_sup
        + phase_sup_weight * phase_sup
    )

    components = {
        "bce": bce.detach(),
        "kl_meter": kl_m.detach(),
        "kl_phase": kl_phi.detach(),
        "kl_tempo": kl_tempo.detach(),
        "kl_taubar": kl_taubar.detach(),
        "taubar_sup": taubar_sup.detach(),
        "meter_sup": meter_sup.detach(),
        "phase_sup": phase_sup.detach(),
        "tempo_density": tempo_density.detach(),
    }
    return total, components
