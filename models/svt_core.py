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

# ---------------------------------------------------------------------------
# Audio-driven prior-mean correction (P0 fix)
# ---------------------------------------------------------------------------
# The generative prior keeps the bar-pointer recursion as its inductive bias
#   μ^p_φ,t = wrap(φ_{t-1} + φ̇_{t-1} + g^φ_ψ(h_t))
#   μ^p_φ̇,t = log φ̇_{t-1} + g^φ̇_ψ(h_t)
# where g_ψ are small audio-conditioned correction heads off h_prior. Without
# the g_ψ terms (the original design) the rolled-out prior means are audio-blind
# and inference free-runs an uninformed random walk; the heads let audio nudge
# the dynamics (a learned Kalman/filter-style correction) while the KL keeps the
# correction from drifting away from the posterior's beat-informed belief.
#
# PHASE_CORR_SCALE = π gives the correction full reach around the circle so the
# prior phase mean can match ANY absolute posterior phase — this is what makes
# the phase-KL reducible to ~0 (no structural floor). The tempo correction is a
# generous ±1.0 in log-space (adjacent-frame log-tempo gaps are << 1).
PHASE_CORR_SCALE = math.pi
TEMPO_CORR_SCALE = 1.0

# Sensible starting tempo so the freshly-initialised prior rolls out near a real
# musical tempo (~120 BPM) instead of exp(0)=1 rad/frame (~820 BPM). This is an
# initialisation only — the (unbounded) tempo head can represent any BPM.
_INIT_LOG_TEMPO = math.log(120.0 / 60.0 * TWO_PI / 86.1328125)  # ≈ -1.93

# Musical bounds on the prior log-tempo recursion. WITHOUT these the prior tempo
# is an UNANCHORED random walk (μ^p_τ,t = log φ̇_{t-1} + corr) that diverges when
# free-running at inference (sample_from_prior): it explodes to exp(+1000), the
# phase advance becomes garbage, and the phase-wrap read-out degrades to noise.
# During training the audio-conditioned posterior re-anchors tempo each frame so
# divergence is hidden — a textbook exposure-bias gap. Clamping the carried prior
# tempo state to a generous musical range (~35–290 BPM) keeps the free-running
# rollout on a coherent sawtooth. Bounds are in log(rad/frame) at fps=86.13.
LOG_TEMPO_MIN = math.log(35.0 / 60.0 * TWO_PI / 86.1328125)   # ≈ -3.16  (35 BPM)
LOG_TEMPO_MAX = math.log(290.0 / 60.0 * TWO_PI / 86.1328125)  # ≈ -1.04  (290 BPM)


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

def _split_head(
    out: Tensor,
    K: int,
    phi_prev: Tensor | None = None,
    phase_delta_scale: float = math.pi,
) -> dict[str, Tensor]:
    """Slice fused [..., K+4] head output into the five distribution params.

    Layout: [meter_logits (K) | phase_mu_raw | phase_kappa_raw | tempo_mu | tempo_sigma_raw]

    Phase mean parameterization:
      - absolute (default, ``phi_prev is None``): μ_φ = π·tanh(raw) — a free
        per-frame angle.
      - recursive (``phi_prev`` given): μ_φ = wrap(φ_{t-1} + δ·tanh(raw)) with
        δ = ``phase_delta_scale`` — continuity by construction (smooth ramp),
        which removes the free-absolute jitter degree of freedom.
    """
    if phi_prev is None:
        phase_mu = math.pi * torch.tanh(out[..., K])
    else:
        phase_mu = torch.remainder(
            phi_prev + phase_delta_scale * torch.tanh(out[..., K]), TWO_PI,
        )
    return {
        "meter_logits": out[..., :K],
        "phase_mu": phase_mu,
        "phase_kappa": _softplus_pos(out[..., K + 1]),
        "tempo_mu": out[..., K + 2],
        "tempo_sigma": _softplus_pos(out[..., K + 3]),
    }


def _init_fused_head(head: nn.Linear, K: int) -> None:
    """Bias the tempo_mu output (slot K+2) toward a real musical tempo.

    Removes the absurd exp(0)=1 rad/frame (~820 BPM) starting point WITHOUT
    baking in a bounded tempo window (the head output stays unbounded).
    """
    with torch.no_grad():
        head.bias[K + 2] = _INIT_LOG_TEMPO


class _TransPosteriorHead(nn.Module):
    """Posterior FFN at t>=1: q_phi(z_t | h_post_t, ẑ_{t-1})."""

    def __init__(
        self,
        hidden_dim: int,
        K: int,
        recursive_phase: bool = False,
        phase_delta_scale: float = math.pi / 4,
    ) -> None:
        super().__init__()
        # z_prev_feat = [cos φ̂, sin φ̂, log τ̂, m̂_soft (K)]  -> 3 + K
        in_dim = hidden_dim + 3 + K
        self.K = K
        self.recursive_phase = recursive_phase
        self.phase_delta_scale = phase_delta_scale
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        # Fused head: meter_logits (K) + 4 scalar params concatenated.
        self.fused_head = nn.Linear(hidden_dim, K + 4)
        _init_fused_head(self.fused_head, K)

    def forward(
        self, h_post_t: Tensor, z_prev_feat: Tensor, phi_prev: Tensor | None = None,
    ) -> dict[str, Tensor]:
        h = self.shared(torch.cat([h_post_t, z_prev_feat], dim=-1))
        out = self.fused_head(h)
        if self.recursive_phase and phi_prev is not None:
            return _split_head(out, self.K, phi_prev=phi_prev,
                               phase_delta_scale=self.phase_delta_scale)
        return _split_head(out, self.K)


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
        _init_fused_head(self.fused_head, K)

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
        _init_fused_head(self.fused_head, K)

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
        phase_corr_scale: float = PHASE_CORR_SCALE,
        tempo_corr_scale: float = TEMPO_CORR_SCALE,
        decoder_use_h_prior: bool = True,
        posterior_phase_recursive: bool = False,
        **kwargs,  # absorb extras for backward compat
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_meter_classes = num_meter_classes
        # Audio-driven prior-mean correction magnitudes. Defaults reproduce the
        # original behaviour; a smaller phase_corr_scale makes audio a *nudge*
        # on the bar-pointer recursion (cleaner sawtooth, more faithful dynamics)
        # at the cost of some phase-KL reducibility.
        self.phase_corr_scale = float(phase_corr_scale)
        self.tempo_corr_scale = float(tempo_corr_scale)
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

        # ---- Audio-driven prior-MEAN correction heads (P0 fix) ----
        # g^φ_ψ(h_t), g^φ̇_ψ(h_t): small per-frame corrections that couple audio
        # to WHERE the latent state is at rollout time. Without these the prior
        # means are a pure recursion of the previous sample (audio-blind), so
        # inference free-runs an uninformed random walk. See PHASE/TEMPO_CORR_SCALE.
        self.prior_phase_corr_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )
        self.prior_tempo_corr_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )
        # Start the correction near zero (model begins as the pure bar-pointer and
        # learns to nudge), but keep weights nonzero so the audio→mean gradient is
        # nonzero from step 0 (Gate 1). Scale down the final layer, zero its bias.
        for _ffn in (self.prior_phase_corr_ffn, self.prior_tempo_corr_ffn):
            last = _ffn[-1]
            assert isinstance(last, nn.Linear)
            with torch.no_grad():
                last.weight.mul_(0.01)
                last.bias.zero_()

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
        # recursive_phase makes q's phase mean μ_q,t = wrap(φ_{t-1}+δ·tanh(·)),
        # a smooth ramp by construction (vs a free per-frame absolute angle that
        # jitters and drags the prior with it via the KL).
        self.posterior_phase_recursive = posterior_phase_recursive
        self.trans_post_head = _TransPosteriorHead(
            hidden_dim, K, recursive_phase=posterior_phase_recursive,
        )

        # ---- Decoder p_theta(b_t | z_t, h) ----
        # decoder_use_h_prior=False makes the decoder LATENT-ONLY
        # ([cos φ, sin φ, log τ, m]) — reconstruction can no longer shortcut
        # through audio (h_prior), so the phase MUST wrap on beats for the
        # decoder to fire. This is what forces the bar-pointer dynamics to be
        # real (a clean beat-locked sawtooth) rather than decorative, lifting
        # the phase-wrap inference read-out.
        self.decoder_use_h_prior = decoder_use_h_prior
        self.h_prior_bottleneck_dim = h_prior_bottleneck
        if not decoder_use_h_prior:
            h_dim_for_decoder = 0
        elif h_prior_bottleneck > 0:
            self.h_prior_bottleneck_proj = nn.Linear(hidden_dim, h_prior_bottleneck)
            h_dim_for_decoder = h_prior_bottleneck
        else:
            h_dim_for_decoder = hidden_dim
        decoder_input_dim = 3 + K + h_dim_for_decoder  # [cos φ, sin φ, log τ, m, (h)]
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
    # Audio-driven prior-mean corrections (P0 fix)
    # ---------------------------------------------------------------------

    def prior_mean_corrections(self, h_prior: Tensor) -> tuple[Tensor, Tensor]:
        """Audio-conditioned corrections g^φ_ψ(h), g^φ̇_ψ(h) for the prior means.

        Accepts ``h_prior`` of shape ``[B, T, D]`` (parallel over time) or
        ``[B, D]`` (single step) and returns ``(phase_corr, tempo_corr)`` with
        the trailing feature dim removed. ``phase_corr`` is bounded to ±π so the
        prior phase mean can reach any point on the circle (making the phase-KL
        reducible); ``tempo_corr`` is a generous ±1.0 nudge in log-space.
        """
        phase_corr = self.phase_corr_scale * torch.tanh(
            self.prior_phase_corr_ffn(h_prior).squeeze(-1)
        )
        tempo_corr = self.tempo_corr_scale * torch.tanh(
            self.prior_tempo_corr_ffn(h_prior).squeeze(-1)
        )
        return phase_corr, tempo_corr

    # ---------------------------------------------------------------------
    # Decoder p_theta(b_t | z_t, h)
    # ---------------------------------------------------------------------

    def _decode(
        self,
        phase: Tensor,        # [B, T]
        log_tempo: Tensor,    # [B, T]
        meter_soft: Tensor,   # [B, T, K]
        h_prior: Tensor,      # [B, T, D]
    ) -> Tensor:
        """Emit beat/downbeat logits [B, T, 2] from the latent (+ optional audio).

        When ``decoder_use_h_prior`` is False the decoder is LATENT-ONLY, so
        reconstruction must flow through the phase/tempo/meter latent — forcing
        the phase to wrap on beats.
        """
        feats = [
            torch.cos(phase).unsqueeze(-1),
            torch.sin(phase).unsqueeze(-1),
            log_tempo.unsqueeze(-1),
            meter_soft,
        ]
        if self.decoder_use_h_prior:
            h = h_prior
            if self.h_prior_bottleneck_dim > 0:
                h = self.h_prior_bottleneck_proj(h)
            feats.append(h)
        return self.emission_decoder(torch.cat(feats, dim=-1))

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
        # Pre-compute per-frame audio-driven prior-mean corrections (P0 fix)
        prior_phase_corr_all, prior_tempo_corr_all = self.prior_mean_corrections(h_prior)  # [B, T] each

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

            post_t = self.trans_post_head(h_post[:, t], z_prev_feat, phi_prev=phase_prev)

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

            # ---- Prior at t (uses sampled ẑ_{t-1} + audio correction g_ψ(h_t)) ----
            # μ^p_φ,t = wrap(φ_{t-1} + φ̇_{t-1} + g^φ_ψ(h_t))
            # μ^p_φ̇,t = log φ̇_{t-1} + g^φ̇_ψ(h_t)
            tempo_prev_lin = torch.exp(log_tempo_prev.clamp(max=10.0))
            prior_phase_mu_t = torch.remainder(
                phase_prev + tempo_prev_lin + prior_phase_corr_all[:, t], TWO_PI
            )
            prior_phase_kappa_t = prior_phase_kappa_all[:, t]
            # Bound the prior tempo mean to a musical range so the free-running
            # rollout cannot diverge (see LOG_TEMPO_MIN/MAX). The KL then pulls the
            # posterior tempo toward this bounded prior.
            prior_tempo_mu_t = (log_tempo_prev + prior_tempo_corr_all[:, t]).clamp(
                LOG_TEMPO_MIN, LOG_TEMPO_MAX
            )
            prior_tempo_sigma_t = prior_tempo_sigma_all[:, t]

            # Meter prior: PDF §3 — boundary based on PREDICTED PHASE MEAN
            # (φ̂_{t-1} + φ̇̂_{t-1}) ≥ 2π. Fed as a hard 0/1 indicator. The
            # bar-boundary test uses the deterministic advance (NOT the audio
            # correction or the noisy sample), per the PDF.
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
        beat_logits = self._decode(
            samples["phase"], samples["log_tempo"], samples["meter_soft"], h_prior,
        )  # [B, T, 2]

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
        # Audio-driven prior-mean corrections (P0 fix): this is what couples the
        # held-out audio to WHERE beats land during the inference rollout.
        prior_phase_corr_all, prior_tempo_corr_all = self.prior_mean_corrections(h_prior)

        # Initial state from prior
        h_global = h_prior.mean(dim=1)
        init_prior = self.init_prior_head(h_global)

        phase_t = von_mises_sample(init_prior["phase_mu"], init_prior["phase_kappa"])
        phase_t = torch.remainder(phase_t, TWO_PI)
        log_tempo_t = lognormal_sample_logspace(
            init_prior["tempo_mu"].clamp(LOG_TEMPO_MIN, LOG_TEMPO_MAX),
            init_prior["tempo_sigma"],
        ).clamp(LOG_TEMPO_MIN, LOG_TEMPO_MAX)
        meter_soft_t = gumbel_softmax_sample(
            init_prior["meter_logits"], temperature=temperature, hard=False,
        )

        sample_phase = [phase_t]
        sample_log_tempo = [log_tempo_t]
        sample_meter_soft = [meter_soft_t]

        # Deterministic prior-MEAN chain (no per-frame noise) for the phase-wrap
        # read-out. The stochastic `phase_traj` jitters the inter-beat intervals
        # (von Mises + accumulating log-tempo random walk) so its wraps are ragged
        # (low CMLt); the mean trajectory follows the bar-pointer recursion exactly
        # -> clean sawtooth. Built from the previous MEAN (not the previous sample)
        # so no jitter leaks in. Zero extra compute; decoder still reads the sample.
        phase_mu_t = torch.remainder(init_prior["phase_mu"], TWO_PI)
        log_tempo_mu_t = init_prior["tempo_mu"].clamp(LOG_TEMPO_MIN, LOG_TEMPO_MAX)
        sample_phase_mu = [phase_mu_t]

        cos_t = torch.cos(phase_t)
        sin_t = torch.sin(phase_t)

        for t in range(1, T):
            phase_prev = phase_t
            log_tempo_prev = log_tempo_t
            meter_soft_prev = meter_soft_t
            cos_pp, sin_pp = cos_t, sin_t

            tempo_prev_lin = torch.exp(log_tempo_prev.clamp(max=10.0))
            prior_phase_mu_t = torch.remainder(
                phase_prev + tempo_prev_lin + prior_phase_corr_all[:, t], TWO_PI
            )

            # Deterministic mean advance (uses the deterministic prev, not samples).
            tempo_mu_prev_lin = torch.exp(log_tempo_mu_t.clamp(max=10.0))
            phase_mu_t = torch.remainder(
                phase_mu_t + tempo_mu_prev_lin + prior_phase_corr_all[:, t], TWO_PI
            )
            log_tempo_mu_t = (log_tempo_mu_t + prior_tempo_corr_all[:, t]).clamp(
                LOG_TEMPO_MIN, LOG_TEMPO_MAX
            )
            sample_phase_mu.append(phase_mu_t)

            phase_t = von_mises_sample(prior_phase_mu_t, prior_phase_kappa_all[:, t])
            phase_t = torch.remainder(phase_t, TWO_PI)
            cos_t = torch.cos(phase_t)
            sin_t = torch.sin(phase_t)
            log_tempo_t = lognormal_sample_logspace(
                (log_tempo_prev + prior_tempo_corr_all[:, t]).clamp(
                    LOG_TEMPO_MIN, LOG_TEMPO_MAX
                ),
                prior_tempo_sigma_all[:, t],
            ).clamp(LOG_TEMPO_MIN, LOG_TEMPO_MAX)

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
        phase_mu_traj = torch.stack(sample_phase_mu, dim=1)
        log_tempo_traj = torch.stack(sample_log_tempo, dim=1)
        meter_soft_traj = torch.stack(sample_meter_soft, dim=1)
        # Hard one-hot: argmax of soft + lazy one-hot (no extra Gumbel sampling).
        meter_hard_traj = F.one_hot(
            meter_soft_traj.argmax(dim=-1),
            num_classes=self.num_meter_classes,
        ).to(meter_soft_traj.dtype)

        # Decode (latent-only when decoder_use_h_prior=False)
        beat_logits = self._decode(phase_traj, log_tempo_traj, meter_soft_traj, h_prior)

        return {
            "phase": phase_traj,
            "phase_mu": phase_mu_traj,
            "log_tempo": log_tempo_traj,
            "meter_soft": meter_soft_traj,
            "meter_onehot": meter_hard_traj,
            "beat_logits": beat_logits,
        }
