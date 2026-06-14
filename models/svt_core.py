"""Sequential Variational Transformer for the bar-pointer VAE.

Faithful to ELBO_for_DBN.pdf, Algorithm 1:

- Posterior at t>=2 conditions on the *sampled* latent ẑ_{t-1}: q_phi(z_t | b, ẑ_{t-1}, h).
- Prior means at t>=2 use the sampled ẑ_{t-1} (NOT teacher-forced GT).
- Phase φ̂_t is sampled before the meter prior π^p_t = f^m_psi(m̂_{t-1}, φ̂_t, φ̂_{t-1}, h).
- Initial-state prior and posterior are separate functions of h (and b for posterior).
- Sequential rollout over t = 0..T-1; small per-step heads, large encoders run once.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.distributions import (
    gumbel_softmax_sample,
    lognormal_sample_logspace,
    von_mises_sample,
)

TWO_PI = 2.0 * math.pi


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (batch-first)."""

    def __init__(self, d_model: int, max_len: int = 20000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        pe = cast(Tensor, self.pe)
        return x + pe[:, :seq_len, :]


# ---------------------------------------------------------------------------
# Positivity utilities (PDF-faithful: Softplus, no hard upper/lower bounds)
# ---------------------------------------------------------------------------

def _softplus_pos(raw: Tensor) -> Tensor:
    """PDF spec for κ and σ: Softplus(NN) > 0, unbounded above."""
    return F.softplus(raw)


# ---------------------------------------------------------------------------
# Per-step posterior head (transition: t >= 1)
# ---------------------------------------------------------------------------

def _split_head(out: Tensor, K: int) -> dict[str, Tensor]:
    """Slice fused [..., K+4] head output into the five distribution params.

    Layout: [meter_logits (K) | phase_mu_raw | phase_kappa_raw | tempo_mu | tempo_sigma_raw]
    """
    return {
        "meter_logits": out[..., :K],
        "phase_mu": math.pi * torch.tanh(out[..., K]),
        "phase_kappa": _softplus_pos(out[..., K + 1]),
        "tempo_mu": out[..., K + 2],
        "tempo_sigma": _softplus_pos(out[..., K + 3]),
    }


class _TransPosteriorHead(nn.Module):
    """Posterior FFN at t>=1: q_phi(z_t | h_post_t, ẑ_{t-1})."""

    def __init__(self, hidden_dim: int, K: int) -> None:
        super().__init__()
        # z_prev_feat = [cos φ̂, sin φ̂, log τ̂, m̂_soft (K)]  -> 3 + K
        in_dim = hidden_dim + 3 + K
        self.K = K
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        # Fused head: meter_logits (K) + 4 scalar params concatenated.
        self.fused_head = nn.Linear(hidden_dim, K + 4)

    def forward(self, h_post_t: Tensor, z_prev_feat: Tensor) -> dict[str, Tensor]:
        h = self.shared(torch.cat([h_post_t, z_prev_feat], dim=-1))
        return _split_head(self.fused_head(h), self.K)


# ---------------------------------------------------------------------------
# Initial-state posterior head (t=0)
# ---------------------------------------------------------------------------

class _InitPosteriorHead(nn.Module):
    """Posterior FFN at t=0: q_phi(z_1 | h_post_0)."""

    def __init__(self, hidden_dim: int, K: int) -> None:
        super().__init__()
        self.K = K
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fused_head = nn.Linear(hidden_dim, K + 4)

    def forward(self, h_post_0: Tensor) -> dict[str, Tensor]:
        h = self.shared(h_post_0)
        return _split_head(self.fused_head(h), self.K)


# ---------------------------------------------------------------------------
# Initial-state prior head (t=0)
# ---------------------------------------------------------------------------

class _InitPriorHead(nn.Module):
    """Prior at t=0: f^init_psi(h_{1:T}). Reads a global summary of h_prior."""

    def __init__(self, hidden_dim: int, K: int) -> None:
        super().__init__()
        self.K = K
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fused_head = nn.Linear(hidden_dim, K + 4)

    def forward(self, h_global: Tensor) -> dict[str, Tensor]:
        h = self.shared(h_global)
        return _split_head(self.fused_head(h), self.K)


# ---------------------------------------------------------------------------
# SVT model
# ---------------------------------------------------------------------------

class SVTModel(nn.Module):
    """Sequential Variational Transformer faithful to Algorithm 1."""

    def __init__(
        self,
        hidden_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_meter_classes: int = 8,
        h_prior_bottleneck: int = 0,
        input_dim: int = 2,
        **kwargs,  # absorb extras for backward compat
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_meter_classes = num_meter_classes
        K = num_meter_classes

        # ---- Prior encoder (parallel; once per forward) ----
        self.prior_input_proj = nn.Linear(input_dim, hidden_dim)
        self.prior_pos_enc = PositionalEncoding(d_model=hidden_dim)
        self.prior_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True),
            num_layers=num_layers,
        )

        # ---- Posterior encoder (parallel; once per forward) ----
        # Input: [activations, beat_distance, downbeat_distance]
        self.post_proj = nn.Linear(input_dim + 2, hidden_dim)
        self.post_pos_enc = PositionalEncoding(d_model=hidden_dim)
        self.post_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True),
            num_layers=num_layers,
        )

        # ---- Per-frame prior uncertainty heads (parallel from h_prior) ----
        # phase kappa, tempo sigma (the means are bar-pointer dynamics from samples)
        self.prior_phase_kappa_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )
        self.prior_tempo_sigma_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )

        # ---- Meter transition prior (per-step) ----
        # PDF §3 (Our model): bar boundary detection is based on the predicted
        # phase mean (φ̂_{t-1} + φ̇̂_{t-1}) ≥ 2π, NOT the noisy sample. We feed
        # this indicator explicitly so the NN doesn't have to rediscover it.
        # Input: [m̂_{t-1} (K), cos/sin φ̂_t (2), cos/sin φ̂_{t-1} (2),
        #         predicted-mean boundary indicator (1), h_prior_t (D)]
        meter_trans_in = K + 4 + 1 + hidden_dim
        self.prior_meter_trans_ffn = nn.Sequential(
            nn.Linear(meter_trans_in, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, K),
        )

        # ---- Initial-state heads ----
        self.init_prior_head = _InitPriorHead(hidden_dim, K)
        self.init_post_head = _InitPosteriorHead(hidden_dim, K)

        # ---- Per-step posterior head (transition) ----
        self.trans_post_head = _TransPosteriorHead(hidden_dim, K)

        # ---- Decoder p_theta(b_t | z_t, h) ----
        self.h_prior_bottleneck_dim = h_prior_bottleneck
        if h_prior_bottleneck > 0:
            self.h_prior_bottleneck_proj = nn.Linear(hidden_dim, h_prior_bottleneck)
            h_dim_for_decoder = h_prior_bottleneck
        else:
            h_dim_for_decoder = hidden_dim
        decoder_input_dim = 3 + K + h_dim_for_decoder  # [cos φ, sin φ, log τ, m, h]
        self.emission_decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # beat, downbeat
        )

    # ---------------------------------------------------------------------
    # Encoders
    # ---------------------------------------------------------------------

    def encode_prior(self, activations: Tensor) -> Tensor:
        """Bidirectional prior Transformer over activations -> h_prior_{1:T}."""
        x = self.prior_input_proj(activations)
        x = self.prior_pos_enc(x)
        return self.prior_encoder(x)  # [B, T, D]

    def encode_posterior(
        self,
        activations: Tensor,
        beat_distance: Tensor,
        downbeat_distance: Tensor,
    ) -> Tensor:
        """Posterior Transformer over [activations, b_dist, db_dist]."""
        x = torch.cat([
            activations,
            beat_distance.unsqueeze(-1),
            downbeat_distance.unsqueeze(-1),
        ], dim=-1)
        x = self.post_proj(x)
        x = self.post_pos_enc(x)
        return self.post_encoder(x)  # [B, T, D]

    # ---------------------------------------------------------------------
    # Beat-distance feature (continuous interpolation between beats)
    # ---------------------------------------------------------------------

    @staticmethod
    def _beat_targets_to_distance(beat_targets: Tensor) -> Tensor:
        """Fraction of time elapsed since the last beat, per frame, in [0, 1).

        Fully vectorized: at each frame t, distance = (t - last_beat_t) /
        (next_beat_t - last_beat_t). At a beat frame, distance = 0; at the
        frame just before the next beat, distance = (span-1)/span.

        Math-equivalent to the original Python loop but no GPU↔CPU syncs.
        """
        B, T = beat_targets.shape
        device = beat_targets.device
        mask = beat_targets > 0.5  # [B, T]
        arange = torch.arange(T, device=device, dtype=torch.float32).expand(B, T)

        # Last beat index ≤ current frame (or 0 for "no prior beat" sentinel).
        last_beat = torch.where(mask, arange, torch.full_like(arange, -1.0))
        last_beat = torch.cummax(last_beat, dim=1)[0].clamp(min=0.0)

        # Next beat index > current frame (or T for "no next beat" sentinel).
        # T (not T-1) so distance after the last beat ramps to (T-1-prev)/(T-prev)
        # = 1 - 1/remaining, matching the original linspace(0, 1-1/span, span).
        next_beat_rev = torch.where(
            mask.flip(1), arange.flip(1), torch.full_like(arange, float(T)),
        )
        next_beat = torch.cummin(next_beat_rev, dim=1)[0].flip(1)

        span = (next_beat - last_beat).clamp(min=1.0)
        distance = ((arange - last_beat) / span).clamp(0.0, 1.0)
        return distance

    # ---------------------------------------------------------------------
    # Forward — Algorithm 1 sequential rollout (training)
    # ---------------------------------------------------------------------

    def forward(
        self,
        activations: Tensor,
        temperature: float = 1.0,
        beat_targets: Tensor | None = None,
        downbeat_targets: Tensor | None = None,
        # legacy kwarg accepted but unused; kept for callers in transition
        z_prev_init: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """Algorithm 1 sequential rollout.

        Returns dict with:
            - beat_logits: [B, T, 2]
            - posterior:   per-step params, [B, T, ...]
            - prior:       per-step params, [B, T, ...]
            - samples:     per-step samples, [B, T, ...]
        """
        del z_prev_init  # ignored — kept for API compat
        B, T, _ = activations.shape
        device = activations.device

        if beat_targets is None:
            beat_targets = torch.zeros(B, T, device=device)
        if downbeat_targets is None:
            downbeat_targets = torch.zeros(B, T, device=device)

        # Distance features for posterior input
        beat_dist = self._beat_targets_to_distance(beat_targets)
        db_dist = self._beat_targets_to_distance(downbeat_targets)

        # Pre-compute encoders (parallel, once)
        h_prior = self.encode_prior(activations)                     # [B, T, D]
        h_post = self.encode_posterior(activations, beat_dist, db_dist)  # [B, T, D]

        # Pre-compute per-frame prior uncertainties (parallel; PDF: Softplus)
        prior_phase_kappa_all = _softplus_pos(
            self.prior_phase_kappa_ffn(h_prior).squeeze(-1)
        )  # [B, T]
        prior_tempo_sigma_all = _softplus_pos(
            self.prior_tempo_sigma_ffn(h_prior).squeeze(-1)
        )  # [B, T]

        # ---- Initial state (t = 0) ----
        h_global = h_prior.mean(dim=1)  # [B, D]
        init_prior = self.init_prior_head(h_global)
        init_post = self.init_post_head(h_post[:, 0])

        phase_t = von_mises_sample(init_post["phase_mu"], init_post["phase_kappa"])
        phase_t = torch.remainder(phase_t, TWO_PI)
        log_tempo_t = lognormal_sample_logspace(
            init_post["tempo_mu"], init_post["tempo_sigma"],
        )
        meter_soft_t = gumbel_softmax_sample(
            init_post["meter_logits"], temperature=temperature, hard=False,
        )

        post_meter_logits = [init_post["meter_logits"]]
        post_phase_mu = [init_post["phase_mu"]]
        post_phase_kappa = [init_post["phase_kappa"]]
        post_tempo_mu = [init_post["tempo_mu"]]
        post_tempo_sigma = [init_post["tempo_sigma"]]

        prior_meter_logits = [init_prior["meter_logits"]]
        prior_phase_mu = [init_prior["phase_mu"]]
        prior_phase_kappa = [init_prior["phase_kappa"]]
        prior_tempo_mu = [init_prior["tempo_mu"]]
        prior_tempo_sigma = [init_prior["tempo_sigma"]]

        sample_phase = [phase_t]
        sample_log_tempo = [log_tempo_t]
        sample_meter_soft = [meter_soft_t]

        # Cache trig of the most recent phase to avoid recomputing each iter.
        cos_t = torch.cos(phase_t)
        sin_t = torch.sin(phase_t)

        # ---- Transition (t = 1..T-1) ----
        for t in range(1, T):
            # Roll cached values: previous = current.
            phase_prev = phase_t
            log_tempo_prev = log_tempo_t
            meter_soft_prev = meter_soft_t
            cos_pp, sin_pp = cos_t, sin_t

            # z_prev feature for the transition posterior
            z_prev_feat = torch.cat([
                cos_pp.unsqueeze(-1),
                sin_pp.unsqueeze(-1),
                log_tempo_prev.unsqueeze(-1),
                meter_soft_prev,
            ], dim=-1)  # [B, K + 3]

            post_t = self.trans_post_head(h_post[:, t], z_prev_feat)

            # Sample phase first (needed by meter prior)
            phase_t = von_mises_sample(post_t["phase_mu"], post_t["phase_kappa"])
            phase_t = torch.remainder(phase_t, TWO_PI)
            cos_t = torch.cos(phase_t)
            sin_t = torch.sin(phase_t)
            log_tempo_t = lognormal_sample_logspace(
                post_t["tempo_mu"], post_t["tempo_sigma"],
            )
            meter_soft_t = gumbel_softmax_sample(
                post_t["meter_logits"], temperature=temperature, hard=False,
            )

            # ---- Prior at t (uses sampled ẑ_{t-1}) ----
            tempo_prev_lin = torch.exp(log_tempo_prev.clamp(max=10.0))
            prior_phase_mu_t = torch.remainder(phase_prev + tempo_prev_lin, TWO_PI)
            prior_phase_kappa_t = prior_phase_kappa_all[:, t]
            prior_tempo_mu_t = log_tempo_prev
            prior_tempo_sigma_t = prior_tempo_sigma_all[:, t]

            # Meter prior: PDF §3 — boundary based on PREDICTED PHASE MEAN
            # (φ̂_{t-1} + φ̇̂_{t-1}) ≥ 2π. Fed as a hard 0/1 indicator.
            boundary = ((phase_prev + tempo_prev_lin) >= TWO_PI).float()  # [B]
            meter_prior_input = torch.cat([
                meter_soft_prev,                # K
                cos_t.unsqueeze(-1),            # cos φ̂_t
                sin_t.unsqueeze(-1),            # sin φ̂_t
                cos_pp.unsqueeze(-1),           # cos φ̂_{t-1}
                sin_pp.unsqueeze(-1),           # sin φ̂_{t-1}
                boundary.unsqueeze(-1),         # boundary indicator
                h_prior[:, t],                  # D
            ], dim=-1)
            prior_meter_logits_t = self.prior_meter_trans_ffn(meter_prior_input)

            # Append
            post_meter_logits.append(post_t["meter_logits"])
            post_phase_mu.append(post_t["phase_mu"])
            post_phase_kappa.append(post_t["phase_kappa"])
            post_tempo_mu.append(post_t["tempo_mu"])
            post_tempo_sigma.append(post_t["tempo_sigma"])

            prior_meter_logits.append(prior_meter_logits_t)
            prior_phase_mu.append(prior_phase_mu_t)
            prior_phase_kappa.append(prior_phase_kappa_t)
            prior_tempo_mu.append(prior_tempo_mu_t)
            prior_tempo_sigma.append(prior_tempo_sigma_t)

            sample_phase.append(phase_t)
            sample_log_tempo.append(log_tempo_t)
            sample_meter_soft.append(meter_soft_t)

        # ---- Stack across time ----
        posterior = {
            "meter_logits": torch.stack(post_meter_logits, dim=1),  # [B, T, K]
            "phase_mu": torch.stack(post_phase_mu, dim=1),          # [B, T]
            "phase_log_kappa": torch.log(torch.stack(post_phase_kappa, dim=1) + 1e-8),
            "tempo_mu": torch.stack(post_tempo_mu, dim=1),
            "tempo_log_sigma": torch.log(torch.stack(post_tempo_sigma, dim=1) + 1e-8),
        }
        prior = {
            "meter_logits": torch.stack(prior_meter_logits, dim=1),
            "phase_mu": torch.stack(prior_phase_mu, dim=1),
            "phase_kappa": torch.stack(prior_phase_kappa, dim=1),
            "tempo_mu": torch.stack(prior_tempo_mu, dim=1),
            "tempo_sigma": torch.stack(prior_tempo_sigma, dim=1),
        }
        samples = {
            "phase": torch.stack(sample_phase, dim=1),               # [B, T]
            "log_tempo": torch.stack(sample_log_tempo, dim=1),       # [B, T]
            "meter_soft": torch.stack(sample_meter_soft, dim=1),     # [B, T, K]
        }

        # ---- Decode (parallel) ----
        h = h_prior
        if self.h_prior_bottleneck_dim > 0:
            h = self.h_prior_bottleneck_proj(h)
        decoder_input = torch.cat([
            torch.cos(samples["phase"]).unsqueeze(-1),
            torch.sin(samples["phase"]).unsqueeze(-1),
            samples["log_tempo"].unsqueeze(-1),
            samples["meter_soft"],
            h,
        ], dim=-1)
        beat_logits = self.emission_decoder(decoder_input)  # [B, T, 2]

        return {
            "beat_logits": beat_logits,
            "posterior": posterior,
            "prior": prior,
            "samples": samples,
        }

    # ---------------------------------------------------------------------
    # Inference: sample from prior alone (no beat annotations available)
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def sample_from_prior(
        self,
        activations: Tensor,
        temperature: float = 0.1,
    ) -> dict[str, Tensor]:
        """Generate latent trajectory by sampling from the prior only.

        Used at inference time when beats are unknown. Bar-pointer dynamics
        are driven by the deterministic mean update plus the learned per-step
        uncertainty (kappa, sigma); meter follows the learned transition.
        """
        T = activations.shape[1]

        h_prior = self.encode_prior(activations)
        prior_phase_kappa_all = _softplus_pos(self.prior_phase_kappa_ffn(h_prior).squeeze(-1))
        prior_tempo_sigma_all = _softplus_pos(self.prior_tempo_sigma_ffn(h_prior).squeeze(-1))

        # Initial state from prior
        h_global = h_prior.mean(dim=1)
        init_prior = self.init_prior_head(h_global)

        phase_t = von_mises_sample(init_prior["phase_mu"], init_prior["phase_kappa"])
        phase_t = torch.remainder(phase_t, TWO_PI)
        log_tempo_t = lognormal_sample_logspace(
            init_prior["tempo_mu"], init_prior["tempo_sigma"],
        )
        meter_soft_t = gumbel_softmax_sample(
            init_prior["meter_logits"], temperature=temperature, hard=False,
        )

        sample_phase = [phase_t]
        sample_log_tempo = [log_tempo_t]
        sample_meter_soft = [meter_soft_t]

        cos_t = torch.cos(phase_t)
        sin_t = torch.sin(phase_t)

        for t in range(1, T):
            phase_prev = phase_t
            log_tempo_prev = log_tempo_t
            meter_soft_prev = meter_soft_t
            cos_pp, sin_pp = cos_t, sin_t

            tempo_prev_lin = torch.exp(log_tempo_prev.clamp(max=10.0))
            prior_phase_mu_t = torch.remainder(phase_prev + tempo_prev_lin, TWO_PI)

            phase_t = von_mises_sample(prior_phase_mu_t, prior_phase_kappa_all[:, t])
            phase_t = torch.remainder(phase_t, TWO_PI)
            cos_t = torch.cos(phase_t)
            sin_t = torch.sin(phase_t)
            log_tempo_t = lognormal_sample_logspace(
                log_tempo_prev, prior_tempo_sigma_all[:, t],
            )

            boundary = ((phase_prev + tempo_prev_lin) >= TWO_PI).float()
            meter_prior_input = torch.cat([
                meter_soft_prev,
                cos_t.unsqueeze(-1),
                sin_t.unsqueeze(-1),
                cos_pp.unsqueeze(-1),
                sin_pp.unsqueeze(-1),
                boundary.unsqueeze(-1),
                h_prior[:, t],
            ], dim=-1)
            meter_logits_t = self.prior_meter_trans_ffn(meter_prior_input)
            meter_soft_t = gumbel_softmax_sample(
                meter_logits_t, temperature=temperature, hard=False,
            )

            sample_phase.append(phase_t)
            sample_log_tempo.append(log_tempo_t)
            sample_meter_soft.append(meter_soft_t)

        phase_traj = torch.stack(sample_phase, dim=1)
        log_tempo_traj = torch.stack(sample_log_tempo, dim=1)
        meter_soft_traj = torch.stack(sample_meter_soft, dim=1)
        # Hard one-hot: argmax of soft + lazy one-hot (no extra Gumbel sampling).
        meter_hard_traj = F.one_hot(
            meter_soft_traj.argmax(dim=-1),
            num_classes=self.num_meter_classes,
        ).to(meter_soft_traj.dtype)

        # Decode for diagnostic comparison
        h = h_prior
        if self.h_prior_bottleneck_dim > 0:
            h = self.h_prior_bottleneck_proj(h)
        decoder_input = torch.cat([
            torch.cos(phase_traj).unsqueeze(-1),
            torch.sin(phase_traj).unsqueeze(-1),
            log_tempo_traj.unsqueeze(-1),
            meter_soft_traj,
            h,
        ], dim=-1)
        beat_logits = self.emission_decoder(decoder_input)  # [B, T, 2]

        return {
            "phase": phase_traj,
            "log_tempo": log_tempo_traj,
            "meter_soft": meter_soft_traj,
            "meter_onehot": meter_hard_traj,
            "beat_logits": beat_logits,
        }
