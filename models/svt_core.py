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


def _delta_committed_ratio(delta: float) -> float:
    """delta-VAE committed rate (Razavi 2019): the under-dispersed std ratio r in (0,1)
    solving h(r) = -ln r + r^2/2 - 0.5 = delta. Forcing the posterior std to
    sigma_q = r·sigma_p (r <= this root) makes the variance-only KL >= delta, so the
    full Gaussian KL(q||p) = h(r) + (mu_q-mu_p)^2/(2 sigma_p^2) >= delta by construction.
    h is strictly decreasing on (0,1) (h'(r) = r - 1/r < 0), so a Newton/bisection root
    is unique. Solved once at init (no autograd needed)."""
    if delta <= 0.0:
        return 1.0
    lo, hi = 1e-6, 1.0  # h(lo)->+inf, h(1)=0; root where h=delta lies in (lo, 1)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        h = -math.log(mid) + 0.5 * mid * mid - 0.5
        if h > delta:   # too concentrated -> move toward 1 (larger r, smaller h)
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


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
        tempo_anchor_mode: str = "none",
        tempo_reversion_alpha: float = 0.0,
        tempo_anchor_ema_beta: float = 0.02,
        audio_emission: bool = False,
        bar_phase: bool = False,
        meter_ste: bool = False,
        delta_vae: bool = False,
        delta_vae_rate: float = 0.1,
        dvbf: bool = False,
        **kwargs,  # absorb extras for backward compat
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_meter_classes = num_meter_classes
        # Straight-Through Gumbel-Softmax for the meter latent: hard one-hot forward
        # (crisp state to the decoder), soft gradient backward (STE). Tests whether the
        # categorical collapse is a differentiability/blurriness problem (ST-GS fixes it)
        # or an information/lazy-decoder problem (ST-GS won't).
        self.meter_hard = bool(meter_ste)
        # Audio-driven prior-mean correction magnitudes. Defaults reproduce the
        # original behaviour; a smaller phase_corr_scale makes audio a *nudge*
        # on the bar-pointer recursion (cleaner sawtooth, more faithful dynamics)
        # at the cost of some phase-KL reducibility.
        self.phase_corr_scale = float(phase_corr_scale)
        self.tempo_corr_scale = float(tempo_corr_scale)
        # ---- Mean-reverting (Ornstein-Uhlenbeck) tempo prior ----
        # The paper's tempo prior μ^p_φ̇,t = log φ̇_{t-1} is a pure random walk: each
        # step's tempo change is penalised (Gaussian) but the LEVEL is uncontrolled,
        # so cumulative variance grows ∝ t and the free-running rollout diverges.
        # A weak reversion μ^p_φ̇,t = logφ̇_{t-1} + α(τ_anchor − logφ̇_{t-1}) + corr
        # keeps within-bar fluctuation possible (rubato) while making sustained
        # drift low-probability (bounded stationary variance). α=0 ⇒ pure paper
        # random walk. Anchor modes: 'init' (t=1 audio tempo), 'global' (learned
        # head on the clip summary), 'ema' (slow EMA of the tempo trajectory).
        self.tempo_anchor_mode = str(tempo_anchor_mode)
        self.tempo_reversion_alpha = float(tempo_reversion_alpha)
        self.tempo_anchor_ema_beta = float(tempo_anchor_ema_beta)
        if self.tempo_anchor_mode == "global":
            self.tempo_anchor_head = nn.Linear(hidden_dim, 1)
            with torch.no_grad():
                self.tempo_anchor_head.weight.mul_(0.01)
                self.tempo_anchor_head.bias.fill_(_INIT_LOG_TEMPO)
        # ---- Hierarchical per-sequence global-tempo latent τ_bar (anchor='latent') ----
        # τ_bar ~ LogNormal is a SINGLE per-clip tempo latent: prior p(τ_bar|h) from the
        # clip summary, posterior q(τ_bar|h,b) from the posterior-encoder summary. τ_bar
        # is the OU anchor in BOTH the training recursion and the free-running rollout, so
        # a correct-octave anchor halves the wrap rate in the exact tensor scored at
        # inference — a clean, exact-ELBO attack on the double-time peg. Its KL is a 4th
        # closed-form term (loss.py), β-annealed, no free-bits.
        if self.tempo_anchor_mode == "latent":
            self.tempo_bar_prior_head = nn.Linear(hidden_dim, 2)  # [mu_p, log_sigma_p]
            self.tempo_bar_post_head = nn.Linear(hidden_dim, 2)   # [mu_q, log_sigma_q]
            with torch.no_grad():
                for _h in (self.tempo_bar_prior_head, self.tempo_bar_post_head):
                    _h.weight.mul_(0.01)
                    _h.bias.data[0] = _INIT_LOG_TEMPO
                    _h.bias.data[1] = -1.0  # σ ≈ 0.37 in log-space at init
        # Scheduled-sampling probability (exposure-bias curriculum). Set per-epoch from
        # the training loop (0 = pure posterior-anchored prior, the original behaviour);
        # >0 trains the prior partly on its OWN free-running predecessor.
        self.scheduled_sampling_eps = 0.0
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
        # DVBF (Karl 2017, ported from paper -- the one repo is unlicensed + image/gym).
        # The posterior emits a SMALL innovation around the prior TRANSITION mean (built
        # in forward), forcing z_t to obey the bar-pointer dynamics so the free-run is
        # faithful by construction and reconstruction must flow through g_phase (waking
        # it). Reuses the recursive posterior head; the anchor (phi_prev) becomes the
        # transition mean, and the innovation budget is tightened (pi/8 vs pi/4).
        self.dvbf = bool(dvbf)
        self.trans_post_head = _TransPosteriorHead(
            hidden_dim, K, recursive_phase=(posterior_phase_recursive or self.dvbf),
            phase_delta_scale=(math.pi / 8 if self.dvbf else math.pi / 4),
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
        # Bar-phase latent (Mode-4 fix): a SECOND wrapped-phase variable tracking
        # within-BAR position (wraps at downbeats) — the dense continuous analog of
        # the categorical meter counter. cos/sin(φ^bar) are appended to the decoder
        # so the downbeat channel reads them; with a latent-only decoder the downbeat
        # MUST flow through φ^bar, forcing it to encode bar position.
        self.bar_phase = bool(bar_phase)
        if not decoder_use_h_prior:
            h_dim_for_decoder = 0
        elif h_prior_bottleneck > 0:
            self.h_prior_bottleneck_proj = nn.Linear(hidden_dim, h_prior_bottleneck)
            h_dim_for_decoder = h_prior_bottleneck
        else:
            h_dim_for_decoder = hidden_dim
        bar_dim = 2 if self.bar_phase else 0       # [cos φ^bar, sin φ^bar]
        decoder_input_dim = 3 + K + bar_dim + h_dim_for_decoder  # [cos φ, sin φ, log τ, m, (bar), (h)]
        self.emission_decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # beat, downbeat
        )

        # ---- Audio-emission head p(h_t | z_t) (Dir 1) ----
        # Predict the input activations (the WaveBeat onset channels) from the LATENT
        # ALONE, so the state must EXPLAIN the audio — madmom's observation model. At
        # inference the particle filter weights trajectories by how well their predicted
        # onsets match the observed ones, which selects the right phase/tempo/level and
        # self-corrects the free-running rollout (closed-loop, sidesteps exposure bias).
        self.audio_emission = bool(audio_emission)
        # delta-VAE committed rate (Razavi 2019, ported from paper -- no official repo).
        # sigma_q = r*·sigma_p with r* = under-dispersed root of h(r)=-ln r + r^2/2 - 0.5
        # = delta guarantees KL(q||p) >= delta for the log-tempo Gaussian BY CONSTRUCTION
        # (structural minimum rate, not a free-bits loss clamp). See _delta_tempo_sigma.
        self.delta_vae = bool(delta_vae)
        self._delta_rstar = _delta_committed_ratio(delta_vae_rate) if self.delta_vae else None
        if self.audio_emission:
            self.audio_emission_head = nn.Sequential(
                nn.Linear(3 + K, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        # ---- Bar-phase latent heads (Mode-4 fix; mirror the beat-phase machinery) ----
        # Posterior/init heads emit [cos μ, sin μ, log κ] (angle via atan2). The prior
        # advances φ^bar by a learned, audio-modulated bar increment δ^bar (≈ beat-rate /
        # beats-per-bar) plus an audio-driven mean correction — identical in spirit to the
        # beat-phase recursion, just K× slower so it wraps once per bar.
        if self.bar_phase:
            self.bar_init_post_head = nn.Linear(hidden_dim, 3)
            self.bar_init_prior_head = nn.Linear(hidden_dim, 3)
            self.bar_post_head = nn.Linear(hidden_dim, 3)
            self.prior_bar_incr_ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
            )
            self.prior_bar_kappa_ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
            )
            self.prior_bar_corr_ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
            )
            with torch.no_grad():
                # Bar increment ≈ 0.036 rad/frame (beat-rate exp(_INIT_LOG_TEMPO)≈0.145 ÷ 4):
                # softplus(b)=0.036 ⇒ b≈-3.3. Wraps ≈ every 2π/0.036 ≈ 174 frames ≈ one
                # 4/4 bar at ~120 BPM @ 86 fps.
                self.prior_bar_incr_ffn[-1].weight.mul_(0.01)
                self.prior_bar_incr_ffn[-1].bias.fill_(-3.3)
                # Audio correction starts near zero (φ^bar begins as a clean recursion).
                self.prior_bar_corr_ffn[-1].weight.mul_(0.01)
                self.prior_bar_corr_ffn[-1].bias.zero_()

    # ---------------------------------------------------------------------
    # Encoders
    # ---------------------------------------------------------------------

    def _tempo_anchor(self, h_global: Tensor, init_prior: dict[str, Tensor]) -> Tensor:
        """Per-sequence OU anchor (log-tempo) for the mean-reverting tempo prior.

        'global' is a learned head on the whole-clip summary; 'init' / 'ema' seed
        from the audio-conditioned t=1 tempo mean ('ema' then drifts it in-loop).
        Returns a [B] tensor (the anchor τ_anchor in log-space)."""
        if self.tempo_anchor_mode == "global":
            return self.tempo_anchor_head(h_global).squeeze(-1)
        return init_prior["tempo_mu"]

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

    @staticmethod
    def _parse_phase_head(out3: Tensor) -> tuple[Tensor, Tensor]:
        """[..., 3] head -> (wrapped angle μ via atan2(sin, cos), κ via softplus)."""
        mu = torch.remainder(torch.atan2(out3[..., 1], out3[..., 0]), TWO_PI)
        kappa = _softplus_pos(out3[..., 2])
        return mu, kappa

    def _delta_tempo_sigma(self, post_sigma: Tensor, prior_sigma: Tensor) -> Tensor:
        """delta-VAE committed rate for the log-tempo Gaussian: sigma_q = rho·sigma_p with
        rho in (0, r*], guaranteeing KL(q||p) >= delta_vae_rate by construction (no
        free-bits clamp). The posterior's own softplus sigma output is repurposed as the
        committed-ratio control via the saturating map s/(1+s): (0,inf) -> (0,1), so the
        encoder still freely chooses rho in (0, r*] but can never reach the prior (rho=1,
        zero KL). r* is the under-dispersed root solved once at init (_delta_rstar)."""
        rho = self._delta_rstar * (post_sigma / (1.0 + post_sigma))
        return prior_sigma * rho

    def _decode(
        self,
        phase: Tensor,        # [B, T]
        log_tempo: Tensor,    # [B, T]
        meter_soft: Tensor,   # [B, T, K]
        h_prior: Tensor,      # [B, T, D]
        bar_phase: Tensor | None = None,  # [B, T]
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
        if self.bar_phase:
            assert bar_phase is not None, "bar_phase decode needs the φ^bar trajectory"
            feats.append(torch.cos(bar_phase).unsqueeze(-1))
            feats.append(torch.sin(bar_phase).unsqueeze(-1))
        if self.decoder_use_h_prior:
            h = h_prior
            if self.h_prior_bottleneck_dim > 0:
                h = self.h_prior_bottleneck_proj(h)
            feats.append(h)
        return self.emission_decoder(torch.cat(feats, dim=-1))

    def _emit_audio(
        self, phase: Tensor, log_tempo: Tensor, meter_soft: Tensor,
    ) -> Tensor:
        """Audio-emission p(h_t | z_t): predict input activations from the latent ALONE.

        Returns ``[B, T, input_dim]``. Latent-only by construction (the state must
        explain the audio); used by the particle filter to score trajectories.
        """
        feats = torch.cat([
            torch.cos(phase).unsqueeze(-1),
            torch.sin(phase).unsqueeze(-1),
            log_tempo.unsqueeze(-1),
            meter_soft,
        ], dim=-1)
        return self.audio_emission_head(feats)

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

        # Bar-phase prior per-frame: learned bar increment δ^bar, uncertainty κ^bar, and
        # audio-driven mean correction (reuses phase_corr_scale).
        if self.bar_phase:
            prior_bar_incr_all = _softplus_pos(self.prior_bar_incr_ffn(h_prior).squeeze(-1))   # [B, T]
            prior_bar_kappa_all = _softplus_pos(self.prior_bar_kappa_ffn(h_prior).squeeze(-1)) # [B, T]
            prior_bar_corr_all = torch.tanh(
                self.prior_bar_corr_ffn(h_prior).squeeze(-1)
            ) * self.phase_corr_scale

        # ---- Initial state (t = 0) ----
        h_global = h_prior.mean(dim=1)  # [B, D]
        init_prior = self.init_prior_head(h_global)
        init_post = self.init_post_head(h_post[:, 0])
        tempo_anchor = self._tempo_anchor(h_global, init_prior)  # [B] OU reference

        # Hierarchical global-tempo latent τ_bar (anchor='latent'): sample from the
        # posterior q(τ_bar|h,b) (reparam) and use it as the OU anchor; its KL is added
        # in the loss. A correct-octave per-clip tempo attacks the double-time peg.
        tempo_bar_params = None
        if self.tempo_anchor_mode == "latent":
            _pri = self.tempo_bar_prior_head(h_global)            # [B, 2]
            _pos = self.tempo_bar_post_head(h_post.mean(dim=1))   # [B, 2]
            _mu_q, _sig_q = _pos[:, 0], _pos[:, 1].exp()
            _mu_p, _sig_p = _pri[:, 0], _pri[:, 1].exp()
            tau_bar = (_mu_q + _sig_q * torch.randn_like(_mu_q)).clamp(LOG_TEMPO_MIN, LOG_TEMPO_MAX)
            tempo_anchor = tau_bar                                # OU anchor = τ_bar sample
            tempo_bar_params = {"mu_q": _mu_q, "sigma_q": _sig_q, "mu_p": _mu_p, "sigma_p": _sig_p}

        # Scheduled-sampling state: the prior's OWN (detached) free-running predecessor,
        # applied to an ε-fraction of sequences so the prior heads train on their own
        # rollout (exposure-bias curriculum). ε=0 ⇒ original posterior-anchored behaviour.
        ss_eps = float(self.scheduled_sampling_eps)
        if ss_eps > 0.0:
            ss_phase_prev = torch.remainder(
                von_mises_sample(init_prior["phase_mu"], init_prior["phase_kappa"]), TWO_PI
            ).detach()
            ss_log_tempo_prev = lognormal_sample_logspace(
                init_prior["tempo_mu"].clamp(LOG_TEMPO_MIN, LOG_TEMPO_MAX),
                init_prior["tempo_sigma"],
            ).clamp(LOG_TEMPO_MIN, LOG_TEMPO_MAX).detach()
            ss_mask = torch.rand(B, device=device) < ss_eps      # [B] per-sequence

        phase_t = von_mises_sample(init_post["phase_mu"], init_post["phase_kappa"])
        phase_t = torch.remainder(phase_t, TWO_PI)
        # delta-VAE: commit the posterior tempo std to r*·sigma_p (>= delta rate).
        init_post_tempo_sigma = (
            self._delta_tempo_sigma(init_post["tempo_sigma"], init_prior["tempo_sigma"])
            if self.delta_vae else init_post["tempo_sigma"]
        )
        log_tempo_t = lognormal_sample_logspace(
            init_post["tempo_mu"], init_post_tempo_sigma,
        )
        meter_soft_t = gumbel_softmax_sample(
            init_post["meter_logits"], temperature=temperature, hard=self.meter_hard,
        )

        post_meter_logits = [init_post["meter_logits"]]
        post_phase_mu = [init_post["phase_mu"]]
        post_phase_kappa = [init_post["phase_kappa"]]
        post_tempo_mu = [init_post["tempo_mu"]]
        post_tempo_sigma = [init_post_tempo_sigma]

        prior_meter_logits = [init_prior["meter_logits"]]
        prior_phase_mu = [init_prior["phase_mu"]]
        prior_phase_kappa = [init_prior["phase_kappa"]]
        prior_tempo_mu = [init_prior["tempo_mu"]]
        prior_tempo_sigma = [init_prior["tempo_sigma"]]

        sample_phase = [phase_t]
        sample_log_tempo = [log_tempo_t]
        sample_meter_soft = [meter_soft_t]

        # ---- Bar-phase initial state (t = 0) ----
        if self.bar_phase:
            bar_mu_post0, bar_kappa_post0 = self._parse_phase_head(self.bar_init_post_head(h_post[:, 0]))
            bar_mu_prior0, bar_kappa_prior0 = self._parse_phase_head(self.bar_init_prior_head(h_global))
            bar_phase_t = torch.remainder(von_mises_sample(bar_mu_post0, bar_kappa_post0), TWO_PI)
            post_bar_mu = [bar_mu_post0]
            post_bar_kappa = [bar_kappa_post0]
            prior_bar_mu = [bar_mu_prior0]
            prior_bar_kappa = [bar_kappa_prior0]
            sample_bar_phase = [bar_phase_t]

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

            # DVBF: anchor the recursive posterior to the FULL prior transition mean
            # (phi_{t-1} + tempo advance + audio correction g_phi(h)), not just phi_{t-1},
            # so the innovation is around the bar-pointer dynamics. Else anchor to phi_{t-1}.
            if self.dvbf:
                _tempo_adv = torch.exp(log_tempo_prev.clamp(max=10.0))
                _post_anchor = torch.remainder(
                    phase_prev + _tempo_adv + prior_phase_corr_all[:, t], TWO_PI)
            else:
                _post_anchor = phase_prev
            post_t = self.trans_post_head(h_post[:, t], z_prev_feat, phi_prev=_post_anchor)

            # Sample phase first (needed by meter prior)
            phase_t = von_mises_sample(post_t["phase_mu"], post_t["phase_kappa"])
            phase_t = torch.remainder(phase_t, TWO_PI)
            cos_t = torch.cos(phase_t)
            sin_t = torch.sin(phase_t)
            # delta-VAE: commit posterior tempo std to r*·sigma_p (>= delta rate). The
            # prior std is precomputed (prior_tempo_sigma_all[:, t]).
            post_tempo_sigma_t = (
                self._delta_tempo_sigma(post_t["tempo_sigma"], prior_tempo_sigma_all[:, t])
                if self.delta_vae else post_t["tempo_sigma"]
            )
            log_tempo_t = lognormal_sample_logspace(
                post_t["tempo_mu"], post_tempo_sigma_t,
            )
            meter_soft_t = gumbel_softmax_sample(
                post_t["meter_logits"], temperature=temperature, hard=self.meter_hard,
            )

            # ---- Prior at t (uses sampled ẑ_{t-1} + audio correction g_ψ(h_t)) ----
            # μ^p_φ,t = wrap(φ_{t-1} + φ̇_{t-1} + g^φ_ψ(h_t))
            # μ^p_φ̇,t = log φ̇_{t-1} + g^φ̇_ψ(h_t)
            # Scheduled sampling: for an ε-fraction of sequences (ss_mask) the recursion
            # is based on the prior's OWN free-running predecessor (ss_*), not the
            # posterior sample, so the prior heads see their inference-time inputs.
            if ss_eps > 0.0:
                base_phase = torch.where(ss_mask, ss_phase_prev, phase_prev)
                base_log_tempo = torch.where(ss_mask, ss_log_tempo_prev, log_tempo_prev)
            else:
                base_phase = phase_prev
                base_log_tempo = log_tempo_prev
            base_tempo_lin = torch.exp(base_log_tempo.clamp(max=10.0))
            prior_phase_mu_t = torch.remainder(
                base_phase + base_tempo_lin + prior_phase_corr_all[:, t], TWO_PI
            )
            prior_phase_kappa_t = prior_phase_kappa_all[:, t]
            # Tempo prior mean: random walk + weak OU reversion toward τ_anchor (= τ_bar
            # when anchor='latent'), then a wide musical clamp as backstop.
            if self.tempo_anchor_mode == "ema":
                tempo_anchor = (1.0 - self.tempo_anchor_ema_beta) * tempo_anchor \
                    + self.tempo_anchor_ema_beta * base_log_tempo
            tempo_reversion = self.tempo_reversion_alpha * (tempo_anchor - base_log_tempo)
            prior_tempo_mu_t = (
                base_log_tempo + tempo_reversion + prior_tempo_corr_all[:, t]
            ).clamp(LOG_TEMPO_MIN, LOG_TEMPO_MAX)
            prior_tempo_sigma_t = prior_tempo_sigma_all[:, t]
            # Advance the prior's own free-running predecessor (detached; no BPTT).
            if ss_eps > 0.0:
                ss_phase_prev = torch.remainder(
                    von_mises_sample(prior_phase_mu_t, prior_phase_kappa_t), TWO_PI
                ).detach()
                ss_log_tempo_prev = lognormal_sample_logspace(
                    prior_tempo_mu_t, prior_tempo_sigma_t
                ).clamp(LOG_TEMPO_MIN, LOG_TEMPO_MAX).detach()

            # Meter prior: PDF §3 — boundary based on PREDICTED PHASE MEAN
            # (φ̂_{t-1} + φ̇̂_{t-1}) ≥ 2π. Fed as a hard 0/1 indicator. The
            # bar-boundary test uses the deterministic advance (NOT the audio
            # correction or the noisy sample), per the PDF.
            boundary = ((base_phase + base_tempo_lin) >= TWO_PI).float()  # [B]
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

            # ---- Bar-phase transition (Mode-4): φ^bar advances K× slower, wraps on bars ----
            if self.bar_phase:
                bar_phase_prev = sample_bar_phase[-1]  # φ^bar_{t-1}
                bar_mu_t, bar_kappa_t = self._parse_phase_head(self.bar_post_head(h_post[:, t]))
                bar_phase_t = torch.remainder(von_mises_sample(bar_mu_t, bar_kappa_t), TWO_PI)
                prior_bar_mu_t = torch.remainder(
                    bar_phase_prev + prior_bar_incr_all[:, t] + prior_bar_corr_all[:, t], TWO_PI
                )
                post_bar_mu.append(bar_mu_t)
                post_bar_kappa.append(bar_kappa_t)
                prior_bar_mu.append(prior_bar_mu_t)
                prior_bar_kappa.append(prior_bar_kappa_all[:, t])
                sample_bar_phase.append(bar_phase_t)

            # Append
            post_meter_logits.append(post_t["meter_logits"])
            post_phase_mu.append(post_t["phase_mu"])
            post_phase_kappa.append(post_t["phase_kappa"])
            post_tempo_mu.append(post_t["tempo_mu"])
            post_tempo_sigma.append(post_tempo_sigma_t)

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

        if self.bar_phase:
            samples["bar_phase"] = torch.stack(sample_bar_phase, dim=1)        # [B, T]
            posterior["barphase_mu"] = torch.stack(post_bar_mu, dim=1)         # [B, T]
            posterior["barphase_log_kappa"] = torch.log(torch.stack(post_bar_kappa, dim=1) + 1e-8)
            prior["barphase_mu"] = torch.stack(prior_bar_mu, dim=1)
            prior["barphase_kappa"] = torch.stack(prior_bar_kappa, dim=1)

        # ---- Decode (parallel) ----
        beat_logits = self._decode(
            samples["phase"], samples["log_tempo"], samples["meter_soft"], h_prior,
            bar_phase=samples.get("bar_phase"),
        )  # [B, T, 2]

        # Audio emission p(h|z) (Dir 1): predicted activations from the latent.
        audio_recon = None
        if self.audio_emission:
            audio_recon = self._emit_audio(
                samples["phase"], samples["log_tempo"], samples["meter_soft"],
            )

        return {
            "beat_logits": beat_logits,
            "posterior": posterior,
            "prior": prior,
            "samples": samples,
            "tempo_bar": tempo_bar_params,
            "audio_recon": audio_recon,
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

        if self.bar_phase:
            prior_bar_incr_all = _softplus_pos(self.prior_bar_incr_ffn(h_prior).squeeze(-1))
            prior_bar_kappa_all = _softplus_pos(self.prior_bar_kappa_ffn(h_prior).squeeze(-1))
            prior_bar_corr_all = torch.tanh(
                self.prior_bar_corr_ffn(h_prior).squeeze(-1)
            ) * self.phase_corr_scale

        # Initial state from prior
        h_global = h_prior.mean(dim=1)
        init_prior = self.init_prior_head(h_global)
        tempo_anchor = self._tempo_anchor(h_global, init_prior)      # stochastic chain
        tempo_anchor_mu = self._tempo_anchor(h_global, init_prior)   # mean chain (separate EMA state)
        if self.tempo_anchor_mode == "latent":
            # Free-running inference uses the PRIOR τ_bar mean as the OU anchor.
            _taubar = self.tempo_bar_prior_head(h_global)[:, 0].clamp(LOG_TEMPO_MIN, LOG_TEMPO_MAX)
            tempo_anchor = _taubar
            tempo_anchor_mu = _taubar

        phase_t = von_mises_sample(init_prior["phase_mu"], init_prior["phase_kappa"])
        phase_t = torch.remainder(phase_t, TWO_PI)
        log_tempo_t = lognormal_sample_logspace(
            init_prior["tempo_mu"].clamp(LOG_TEMPO_MIN, LOG_TEMPO_MAX),
            init_prior["tempo_sigma"],
        ).clamp(LOG_TEMPO_MIN, LOG_TEMPO_MAX)
        meter_soft_t = gumbel_softmax_sample(
            init_prior["meter_logits"], temperature=temperature, hard=self.meter_hard,
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

        # ---- Bar-phase free-run init (stochastic sample + deterministic mean chain) ----
        if self.bar_phase:
            bar_mu0, bar_kappa0 = self._parse_phase_head(self.bar_init_prior_head(h_global))
            bar_phase_t = torch.remainder(von_mises_sample(bar_mu0, bar_kappa0), TWO_PI)
            bar_phase_mu_t = bar_mu0
            sample_bar_phase = [bar_phase_t]
            sample_bar_phase_mu = [bar_phase_mu_t]

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
            if self.tempo_anchor_mode == "ema":
                tempo_anchor_mu = (1.0 - self.tempo_anchor_ema_beta) * tempo_anchor_mu \
                    + self.tempo_anchor_ema_beta * log_tempo_mu_t
            log_tempo_mu_t = (
                log_tempo_mu_t
                + self.tempo_reversion_alpha * (tempo_anchor_mu - log_tempo_mu_t)
                + prior_tempo_corr_all[:, t]
            ).clamp(LOG_TEMPO_MIN, LOG_TEMPO_MAX)
            sample_phase_mu.append(phase_mu_t)

            phase_t = von_mises_sample(prior_phase_mu_t, prior_phase_kappa_all[:, t])
            phase_t = torch.remainder(phase_t, TWO_PI)
            cos_t = torch.cos(phase_t)
            sin_t = torch.sin(phase_t)
            if self.tempo_anchor_mode == "ema":
                tempo_anchor = (1.0 - self.tempo_anchor_ema_beta) * tempo_anchor \
                    + self.tempo_anchor_ema_beta * log_tempo_prev
            log_tempo_t = lognormal_sample_logspace(
                (
                    log_tempo_prev
                    + self.tempo_reversion_alpha * (tempo_anchor - log_tempo_prev)
                    + prior_tempo_corr_all[:, t]
                ).clamp(LOG_TEMPO_MIN, LOG_TEMPO_MAX),
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
                meter_logits_t, temperature=temperature, hard=self.meter_hard,
            )

            # ---- Bar-phase advance (stochastic + deterministic mean chain) ----
            if self.bar_phase:
                bar_phase_prev = sample_bar_phase[-1]
                prior_bar_mu_t = torch.remainder(
                    bar_phase_prev + prior_bar_incr_all[:, t] + prior_bar_corr_all[:, t], TWO_PI
                )
                bar_phase_t = torch.remainder(
                    von_mises_sample(prior_bar_mu_t, prior_bar_kappa_all[:, t]), TWO_PI
                )
                # Deterministic mean chain (clean sawtooth for the wrap read-out).
                bar_phase_mu_t = torch.remainder(
                    bar_phase_mu_t + prior_bar_incr_all[:, t] + prior_bar_corr_all[:, t], TWO_PI
                )
                sample_bar_phase.append(bar_phase_t)
                sample_bar_phase_mu.append(bar_phase_mu_t)

            sample_phase.append(phase_t)
            sample_log_tempo.append(log_tempo_t)
            sample_meter_soft.append(meter_soft_t)

        phase_traj = torch.stack(sample_phase, dim=1)
        phase_mu_traj = torch.stack(sample_phase_mu, dim=1)
        log_tempo_traj = torch.stack(sample_log_tempo, dim=1)
        meter_soft_traj = torch.stack(sample_meter_soft, dim=1)
        bar_phase_traj = torch.stack(sample_bar_phase, dim=1) if self.bar_phase else None
        bar_phase_mu_traj = torch.stack(sample_bar_phase_mu, dim=1) if self.bar_phase else None
        # Hard one-hot: argmax of soft + lazy one-hot (no extra Gumbel sampling).
        meter_hard_traj = F.one_hot(
            meter_soft_traj.argmax(dim=-1),
            num_classes=self.num_meter_classes,
        ).to(meter_soft_traj.dtype)

        # Decode (latent-only when decoder_use_h_prior=False)
        beat_logits = self._decode(
            phase_traj, log_tempo_traj, meter_soft_traj, h_prior, bar_phase=bar_phase_traj,
        )

        out = {
            "phase": phase_traj,
            "phase_mu": phase_mu_traj,
            "log_tempo": log_tempo_traj,
            "meter_soft": meter_soft_traj,
            "meter_onehot": meter_hard_traj,
            "beat_logits": beat_logits,
        }
        if self.bar_phase:
            out["bar_phase"] = bar_phase_traj
            out["bar_phase_mu"] = bar_phase_mu_traj
        return out

    # ---------------------------------------------------------------------
    # Inference: particle filter weighted by the audio-emission model (Dir 1B)
    # ---------------------------------------------------------------------

    @staticmethod
    def _systematic_resample(weights: Tensor) -> Tensor:
        """Systematic resampling: normalized weights ``[N]`` -> ancestor idx ``[N]``.

        One uniform offset, evenly spaced positions — low variance, O(N)."""
        N = weights.shape[0]
        device = weights.device
        positions = (
            torch.arange(N, device=device, dtype=weights.dtype)
            + torch.rand((), device=device)
        ) / N
        cumsum = torch.cumsum(weights, dim=0)
        cumsum = cumsum / cumsum[-1].clamp(min=1e-12)
        cumsum[-1] = 1.0
        return torch.searchsorted(cumsum, positions).clamp(max=N - 1)

    @torch.no_grad()
    def sample_from_prior_pf(
        self,
        activations: Tensor,
        n_particles: int = 400,
        obs_sigma: float = 0.3,
        temperature: float = 0.1,
        ess_frac: float = 0.5,
    ) -> dict[str, Tensor]:
        """Bootstrap particle filter inference (Dir 1B).

        Free-runs the EXACT prior dynamics of ``sample_from_prior`` over ``N``
        particles, but at every frame reweights them by the audio-emission
        likelihood ``p(h_t | z_t) = N(h_t; emit(z_t), σ²I)`` and systematically
        resamples when the effective sample size drops. This is a closed-loop
        observation model (madmom-style): particles whose latent state fails to
        EXPLAIN the observed onsets die, so the surviving trajectory locks onto the
        right phase / tempo / metrical level — self-correcting at inference and
        sidestepping the open-loop exposure bias that caps ``sample_from_prior``.

        Returns the MAP (highest-weight) trajectory, drop-in compatible with the
        gate4 read-out (``phase`` / ``phase_mu`` / ``beat_logits``). Requires
        ``audio_emission=True`` and a single sequence (B=1).
        """
        assert self.audio_emission, "sample_from_prior_pf needs an audio_emission head"
        B, T, _ = activations.shape
        assert B == 1, "particle filter runs one sequence at a time (B=1)"
        device = activations.device
        N = int(n_particles)
        inv_2sig2 = 1.0 / (2.0 * obs_sigma * obs_sigma)

        # ---- Shared (audio-only) quantities: computed once, broadcast to particles ----
        h_prior = self.encode_prior(activations)                                   # [1, T, D]
        D = h_prior.shape[-1]
        prior_phase_kappa_all = _softplus_pos(self.prior_phase_kappa_ffn(h_prior).squeeze(-1))  # [1, T]
        prior_tempo_sigma_all = _softplus_pos(self.prior_tempo_sigma_ffn(h_prior).squeeze(-1))  # [1, T]
        prior_phase_corr_all, prior_tempo_corr_all = self.prior_mean_corrections(h_prior)        # [1, T] each
        if self.bar_phase:
            prior_bar_incr_all = _softplus_pos(self.prior_bar_incr_ffn(h_prior).squeeze(-1))     # [1, T]
            prior_bar_kappa_all = _softplus_pos(self.prior_bar_kappa_ffn(h_prior).squeeze(-1))   # [1, T]
            prior_bar_corr_all = torch.tanh(self.prior_bar_corr_ffn(h_prior).squeeze(-1)) * self.phase_corr_scale
        obs = activations[0]                                                        # [T, input_dim]

        h_global = h_prior.mean(dim=1)                                              # [1, D]
        init_prior = self.init_prior_head(h_global)

        # OU anchor (τ_bar prior mean when latent; else the configured anchor) -> [N].
        if self.tempo_anchor_mode == "latent":
            anchor = self.tempo_bar_prior_head(h_global)[:, 0].clamp(LOG_TEMPO_MIN, LOG_TEMPO_MAX)
        else:
            anchor = self._tempo_anchor(h_global, init_prior)
        anchor = anchor.expand(N).contiguous()                                     # [N]

        # ---- Initialise N particles from the initial prior ----
        phase_t = torch.remainder(
            von_mises_sample(init_prior["phase_mu"].expand(N), init_prior["phase_kappa"].expand(N)),
            TWO_PI,
        )                                                                          # [N]
        log_tempo_t = lognormal_sample_logspace(
            init_prior["tempo_mu"].clamp(LOG_TEMPO_MIN, LOG_TEMPO_MAX).expand(N),
            init_prior["tempo_sigma"].expand(N),
        ).clamp(LOG_TEMPO_MIN, LOG_TEMPO_MAX)                                       # [N]
        meter_soft_t = gumbel_softmax_sample(
            init_prior["meter_logits"].expand(N, -1), temperature=temperature, hard=self.meter_hard,
        )                                                                          # [N, K]
        if self.bar_phase:
            _bm0, _bk0 = self._parse_phase_head(self.bar_init_prior_head(h_global))
            bar_phase_t = torch.remainder(von_mises_sample(_bm0.expand(N), _bk0.expand(N)), TWO_PI)  # [N]

        # Genealogical trajectory store (reindexed on every resample so the stored
        # path of each particle is consistent with its ancestry).
        traj_phase = torch.empty(N, T, device=device)
        traj_log_tempo = torch.empty(N, T, device=device)
        traj_meter = torch.empty(N, T, self.num_meter_classes, device=device)
        traj_phase[:, 0] = phase_t
        traj_log_tempo[:, 0] = log_tempo_t
        traj_meter[:, 0] = meter_soft_t
        if self.bar_phase:
            traj_bar_phase = torch.empty(N, T, device=device)
            traj_bar_phase[:, 0] = bar_phase_t

        # Accumulating log-weights, seeded by the t=0 emission.
        pred0 = self._emit_audio(phase_t, log_tempo_t, meter_soft_t)               # [N, input_dim]
        log_w = -inv_2sig2 * ((pred0 - obs[0]) ** 2).sum(dim=-1)                   # [N]

        # Bayesian beat read-out: per-frame weighted fraction of particles whose
        # bar-pointer wrapped INTO this frame (a beat in the DBN). Smoother and
        # better-aligned than wrap-detecting a single sampled trajectory, since it
        # marginalises over the whole filtering posterior (madmom-style).
        beat_activation = torch.zeros(T, device=device)

        for t in range(1, T):
            phase_prev = phase_t
            log_tempo_prev = log_tempo_t
            meter_soft_prev = meter_soft_t
            cos_pp = torch.cos(phase_prev)
            sin_pp = torch.sin(phase_prev)

            kappa_t = prior_phase_kappa_all[0, t].expand(N)
            sigma_t = prior_tempo_sigma_all[0, t].expand(N)
            pcorr_t = prior_phase_corr_all[0, t]   # scalar, broadcasts over [N]
            tcorr_t = prior_tempo_corr_all[0, t]   # scalar

            # Phase: bar-pointer advance + audio nudge, then von Mises sample.
            tempo_prev_lin = torch.exp(log_tempo_prev.clamp(max=10.0))
            prior_phase_mu_t = torch.remainder(phase_prev + tempo_prev_lin + pcorr_t, TWO_PI)
            phase_t = torch.remainder(von_mises_sample(prior_phase_mu_t, kappa_t), TWO_PI)
            cos_t = torch.cos(phase_t)
            sin_t = torch.sin(phase_t)

            # Tempo: random walk + weak OU reversion toward the anchor (τ_bar).
            if self.tempo_anchor_mode == "ema":
                anchor = (1.0 - self.tempo_anchor_ema_beta) * anchor \
                    + self.tempo_anchor_ema_beta * log_tempo_prev
            prior_tempo_mu_t = (
                log_tempo_prev
                + self.tempo_reversion_alpha * (anchor - log_tempo_prev)
                + tcorr_t
            ).clamp(LOG_TEMPO_MIN, LOG_TEMPO_MAX)
            log_tempo_t = lognormal_sample_logspace(prior_tempo_mu_t, sigma_t).clamp(
                LOG_TEMPO_MIN, LOG_TEMPO_MAX
            )

            # Meter transition (PDF §3 boundary from the deterministic advance).
            boundary = ((phase_prev + tempo_prev_lin) >= TWO_PI).float()
            meter_in = torch.cat([
                meter_soft_prev,
                cos_t.unsqueeze(-1), sin_t.unsqueeze(-1),
                cos_pp.unsqueeze(-1), sin_pp.unsqueeze(-1),
                boundary.unsqueeze(-1),
                h_prior[0, t].expand(N, D),
            ], dim=-1)
            meter_logits_t = self.prior_meter_trans_ffn(meter_in)
            meter_soft_t = gumbel_softmax_sample(
                meter_logits_t, temperature=temperature, hard=self.meter_hard,
            )

            # Bar-phase: prior recursion (advance + audio nudge), von Mises sample.
            if self.bar_phase:
                prior_bar_mu_t = torch.remainder(
                    bar_phase_t + prior_bar_incr_all[0, t] + prior_bar_corr_all[0, t], TWO_PI
                )
                bar_phase_t = torch.remainder(
                    von_mises_sample(prior_bar_mu_t, prior_bar_kappa_all[0, t].expand(N)), TWO_PI
                )

            traj_phase[:, t] = phase_t
            traj_log_tempo[:, t] = log_tempo_t
            traj_meter[:, t] = meter_soft_t
            if self.bar_phase:
                traj_bar_phase[:, t] = bar_phase_t

            # ---- Reweight by the audio-emission likelihood p(h_t | z_t) ----
            pred = self._emit_audio(phase_t, log_tempo_t, meter_soft_t)            # [N, input_dim]
            log_w = log_w - inv_2sig2 * ((pred - obs[t]) ** 2).sum(dim=-1)

            # Filtering posterior at frame t (uses ALL evidence since last resample).
            w = torch.softmax(log_w, dim=0)
            # Weighted beat probability: particles wrapping into frame t (PDF §3
            # boundary indicator), weighted by the current posterior.
            beat_activation[t] = (w * boundary).sum()

            # ---- Resample when ESS drops (systematic; reindex the trajectories) ----
            # Skip on the final frame so log_w carries a meaningful MAP signal.
            ess = 1.0 / (w * w).sum().clamp(min=1e-12)
            if ess < ess_frac * N and t < T - 1:
                idx = self._systematic_resample(w)
                traj_phase = traj_phase[idx]
                traj_log_tempo = traj_log_tempo[idx]
                traj_meter = traj_meter[idx]
                phase_t = phase_t[idx]
                log_tempo_t = log_tempo_t[idx]
                meter_soft_t = meter_soft_t[idx]
                if self.bar_phase:
                    traj_bar_phase = traj_bar_phase[idx]
                    bar_phase_t = bar_phase_t[idx]
                anchor = anchor[idx]
                log_w = torch.zeros(N, device=device)

        # ---- MAP trajectory = highest final-weight particle (genealogical path) ----
        best = torch.argmax(log_w)
        phase_map = traj_phase[best]                 # [T]
        log_tempo_map = traj_log_tempo[best]         # [T]
        meter_map = traj_meter[best]                 # [T, K]
        meter_onehot = F.one_hot(
            meter_map.argmax(dim=-1), self.num_meter_classes,
        ).to(meter_map.dtype)

        bar_phase_map = traj_bar_phase[best] if self.bar_phase else None  # [T] or None
        beat_logits = self._decode(
            phase_map.unsqueeze(0), log_tempo_map.unsqueeze(0), meter_map.unsqueeze(0), h_prior,
            bar_phase=(bar_phase_map.unsqueeze(0) if bar_phase_map is not None else None),
        )                                            # [1, T, 2]

        return {
            "phase": phase_map.unsqueeze(0),
            "phase_mu": phase_map.unsqueeze(0),      # PF selection denoises; MAP path IS the read-out
            "log_tempo": log_tempo_map.unsqueeze(0),
            "meter_soft": meter_map.unsqueeze(0),
            "meter_onehot": meter_onehot.unsqueeze(0),
            "beat_logits": beat_logits,
            "beat_activation": beat_activation.unsqueeze(0),  # [1, T] Bayesian wrap read-out
            **({"bar_phase": bar_phase_map.unsqueeze(0),
                "bar_phase_mu": bar_phase_map.unsqueeze(0)} if self.bar_phase else {}),
        }
