"""Core Sequential Variational Transformer (SVT) model for the bar pointer VAE.

Architecture per the pseudocode documents:

Prior (transition_model_pseudocode.pdf):
- Bidirectional Encoder Transformer on x_{1:T} → h_prior_{1:T}
- FFN heads with Softplus for kappa/sigma, delta*Sigmoid for meter epsilon
- All prior params computed in parallel from h_prior
- Means are fixed (bar pointer dynamics), uncertainties are learned

Posterior:
- Transformer Encoder on [h_{1:T}, b_{1:T}] concatenated
- Audio features provide position-varying content, beats mark locations
- FFN heads with Softplus for kappa/sigma, pi*tanh for phase mu
- All posterior params computed in parallel (no z_{t-1} dependency)
- Does NOT condition on z_{t-1} — purely from (h, b)

Training (Algorithm 1):
- Prior means depend on z_{t-1} (sequential), but prior uncertainties
  and all posterior params are pre-computed in parallel.
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
    """Sinusoidal positional encoding supporting batch-first tensors."""

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


class SVTModel(nn.Module):
    """Sequential Variational Transformer for the bar pointer VAE.

    Args:
        hidden_dim: Transformer / MLP hidden dimensionality.
        nhead: Number of attention heads.
        num_layers: Number of Transformer layers per encoder.
        num_meter_classes: Number of discrete meter categories (K).
        meter_delta: Upper bound for meter transition epsilon_ij.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_meter_classes: int = 8,
        meter_delta: float = 0.001,
        h_prior_bottleneck: int = 0,
        input_dim: int = 2,
        **kwargs,  # absorb extra args like z_context for backward compat
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_meter_classes = num_meter_classes
        self.meter_delta = meter_delta
        K = num_meter_classes

        # ================================================================
        # PRIOR: Encoder Transformer on x_{1:T} → h_prior_{1:T}
        # (transition_model_pseudocode.pdf, Step 0)
        # ================================================================
        self.prior_input_proj = nn.Linear(input_dim, hidden_dim)
        self.prior_pos_enc = PositionalEncoding(d_model=hidden_dim)
        self.prior_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=nhead, batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Prior FFN heads (applied position-wise to h_prior)
        # Meter: epsilon_ij matrix — delta * sigmoid(FFN) → (0, delta)
        self.prior_meter_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, K * K),
        )
        # Phase: concentration kappa — Softplus(FFN) > 0
        self.prior_phase_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Tempo: std sigma — Softplus(FFN) > 0
        self.prior_tempo_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Learnable initial tempo mean (for t=1)
        # Initialize to ~120 BPM at 86fps: log(2π * 120 / (60 * 86)) ≈ -1.92
        self.prior_tempo_mu_init = nn.Parameter(torch.tensor(-1.9))

        # ================================================================
        # POSTERIOR: Transformer Encoder on [h_{1:T}, d_beat, d_downbeat]
        # q_φ(m_{1:T}, φ_{1:T}, φ̇_{1:T} | h_{1:T}, d_beat, d_downbeat)
        # Input: audio features h_{1:T} concatenated with
        #        beat-distance (1-dim) and downbeat-distance (1-dim)
        # ================================================================
        self.post_proj = nn.Linear(input_dim + 2, hidden_dim)  # [h_t, d_beat, d_db] → D
        self.post_pos_enc = PositionalEncoding(d_model=hidden_dim)
        self.post_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=nhead, batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Posterior FFN heads (applied position-wise to h_post)
        # Meter: softmax(FFN)
        self.post_meter_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, K),
        )
        # Phase mu: pi * tanh(FFN) → [-pi, pi]
        self.post_phase_mu_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Phase kappa: Softplus(FFN) > 0
        self.post_phase_kappa_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Tempo mu: FFN (unconstrained, log-space)
        self.post_tempo_mu_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Tempo sigma: Softplus(FFN) > 0
        self.post_tempo_sigma_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # ================================================================
        # DECODER (emission model)
        # p_θ(b_t | z_t, h_{1:T}) = Bern(σ(NN_θ(z_t, h_{1:T})))
        # ================================================================
        self.h_prior_bottleneck_dim = h_prior_bottleneck
        if h_prior_bottleneck > 0:
            self.h_prior_bottleneck_proj = nn.Linear(hidden_dim, h_prior_bottleneck)
            h_dim_for_decoder = h_prior_bottleneck
        else:
            h_dim_for_decoder = hidden_dim

        decoder_input_dim = 3 + K + h_dim_for_decoder  # [cos(φ), sin(φ), log_tempo, meter, h_prior]
        self.emission_decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # channel 0: beat, channel 1: downbeat
        )

    # ------------------------------------------------------------------
    # Prior: encode and compute uncertainty params (parallel)
    # ------------------------------------------------------------------

    def encode_prior(self, activations: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Encode audio and compute all prior uncertainty params in parallel.

        Per transition_model_pseudocode.pdf Step 0.

        Args:
            activations: [B, T, 2] acoustic activations.

        Returns:
            h_prior: [B, T, D] prior hidden states.
            prior_params: Dict with pre-computed uncertainty params:
                - epsilon_ij: [B, T, K, K] meter transition probs (0, delta)
                - phase_kappa: [B, T] concentration
                - tempo_sigma: [B, T] std in log-space
        """
        x = self.prior_input_proj(activations)
        x = self.prior_pos_enc(x)
        h_prior = self.prior_encoder(x)  # [B, T, D]

        K = self.num_meter_classes
        B, T, _ = h_prior.shape

        # Meter: delta * sigmoid(FFN) → epsilon_ij in (0, delta)
        epsilon_ij = self.meter_delta * torch.sigmoid(
            self.prior_meter_ffn(h_prior).view(B, T, K, K)
        )

        # Phase: kappa in [100, 300] — prior should be confident about
        # its phase prediction (phi_prev + tempo is a good estimate)
        phase_kappa = 100.0 + 200.0 * torch.sigmoid(
            self.prior_phase_ffn(h_prior).squeeze(-1)
        )

        # Tempo: sigma in [0.001, 0.05]
        tempo_sigma = 0.001 + 0.049 * torch.sigmoid(
            self.prior_tempo_ffn(h_prior).squeeze(-1)
        )

        return h_prior, {
            "epsilon_ij": epsilon_ij,
            "phase_kappa": phase_kappa,
            "tempo_sigma": tempo_sigma,
        }

    def compute_prior_at_t(
        self,
        prior_params: dict[str, Tensor],
        t: int,
        phase_prev: Tensor,
        log_tempo_prev: Tensor,
        meter_onehot_prev: Tensor,
    ) -> dict[str, Tensor]:
        """Compute prior distribution params at time t using z_{t-1}.

        Per transition_model_pseudocode.pdf Steps 1-2.

        Args:
            prior_params: Pre-computed uncertainty params from encode_prior.
            t: Current timestep (0-indexed).
            phase_prev: [B, 1] previous phase.
            log_tempo_prev: [B, 1] previous log-tempo.
            meter_onehot_prev: [B, K] previous meter one-hot.
        """
        K = self.num_meter_classes

        if t == 0:
            # Step 1: Initial state
            B = phase_prev.size(0)
            device = phase_prev.device
            meter_logits = torch.zeros(B, K, device=device)  # uniform
            phase_mu = torch.zeros(B, device=device)
            phase_kappa = prior_params["phase_kappa"][:, 0]
            tempo_mu = self.prior_tempo_mu_init.expand(B)
            tempo_sigma = prior_params["tempo_sigma"][:, 0]
        else:
            # Step 2: Transition
            # Phase: mu = (phi_{t-1} + phidot_{t-1}) mod 2pi
            tempo_prev = torch.exp(log_tempo_prev.clamp(max=10.0))
            phase_mu = torch.remainder(
                phase_prev.squeeze(-1) + tempo_prev.squeeze(-1), TWO_PI
            )

            # Phase kappa from pre-computed
            phase_kappa = prior_params["phase_kappa"][:, t]

            # Tempo: mu = log(phidot_{t-1}) = log_tempo_prev
            tempo_mu = log_tempo_prev.squeeze(-1)
            tempo_sigma = prior_params["tempo_sigma"][:, t]

            # Meter: boundary-gated transition
            wrap_t = (phase_prev.squeeze(-1) + tempo_prev.squeeze(-1)) >= TWO_PI  # [B]
            eps_t = prior_params["epsilon_ij"][:, t]  # [B, K, K]

            # Build transition: pi[i,j] = eps[i,j] * wrap, pi[i,i] = 1 - sum(eps[i,:]) * wrap
            # For soft meter_onehot_prev, compute expected transition
            diag_mask = torch.eye(K, device=eps_t.device).unsqueeze(0)  # [1, K, K]
            off_diag_eps = eps_t * (1.0 - diag_mask)  # zero diagonal
            row_sum = off_diag_eps.sum(dim=-1, keepdim=True)  # [B, K, 1]
            stay_prob = 1.0 - row_sum  # [B, K, 1]
            trans_matrix = off_diag_eps + stay_prob * diag_mask  # [B, K, K]

            # Gate by wrap: if no wrap, stay in same meter
            wrap_mask = wrap_t.unsqueeze(-1).unsqueeze(-1).float()  # [B, 1, 1]
            trans_matrix = wrap_mask * trans_matrix + (1.0 - wrap_mask) * diag_mask.expand_as(trans_matrix)

            # Apply: pi_t = meter_prev @ trans_matrix
            meter_prev_prob = meter_onehot_prev + 1e-6
            meter_prev_prob = meter_prev_prob / meter_prev_prob.sum(dim=-1, keepdim=True)
            meter_logits = torch.log(
                torch.bmm(meter_prev_prob.unsqueeze(1), trans_matrix).squeeze(1) + 1e-10
            )

        return {
            "meter_logits": meter_logits,       # [B, K]
            "phase_mu": phase_mu,               # [B]
            "phase_kappa": phase_kappa,          # [B]
            "tempo_mu": tempo_mu,               # [B]
            "tempo_sigma": tempo_sigma,          # [B]
        }

    # ------------------------------------------------------------------
    # Posterior: encode-decode and compute all params (parallel)
    # ------------------------------------------------------------------

    def encode_posterior(
        self,
        activations: Tensor,
        beat_distance: Tensor,
        downbeat_distance: Tensor,
    ) -> dict[str, Tensor]:
        """Compute all posterior params in parallel via Transformer Encoder.

        Args:
            activations: [B, T, D_feat] acoustic features (h_{1:T}).
            beat_distance: [B, T] fraction since last beat (0→1).
            downbeat_distance: [B, T] fraction since last downbeat (0→1).

        Returns:
            Dict with posterior params for all T timesteps.
        """
        # Concatenate: [B, T, D_feat+2]
        x = torch.cat([
            activations,
            beat_distance.unsqueeze(-1),
            downbeat_distance.unsqueeze(-1),
        ], dim=-1)
        x = self.post_proj(x)           # [B, T, D]
        x = self.post_pos_enc(x)
        h_post = self.post_encoder(x)   # [B, T, D]

        # FFN heads (position-wise, shared weights)
        return {
            "meter_logits": self.post_meter_ffn(h_post),                              # [B, T, K]
            "phase_mu": math.pi * torch.tanh(self.post_phase_mu_ffn(h_post).squeeze(-1)),  # [B, T]
            "phase_kappa": 200.0 + 300.0 * torch.sigmoid(                                # [B, T] in [200, 500]
                self.post_phase_kappa_ffn(h_post).squeeze(-1)),
            "tempo_mu": -2.1 + 0.9 * torch.tanh(                                      # [B, T] in [-3.0, -1.2]
                self.post_tempo_mu_ffn(h_post).squeeze(-1)),                           # ~40-250 BPM at 86fps (madmom range)
            "tempo_sigma": 0.01 + 1.99 * torch.sigmoid(                               # [B, T] in [0.01, 2.0]
                self.post_tempo_sigma_ffn(h_post).squeeze(-1)),
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_latent_at_t(
        self,
        posterior: dict[str, Tensor],
        t: int,
        temperature: float = 1.0,
    ) -> dict[str, Tensor]:
        """Sample z_t from posterior at timestep t.

        Args:
            posterior: Full posterior params [B, T, ...] from encode_posterior.
            t: Timestep index.
            temperature: Gumbel-Softmax temperature.

        Returns:
            Dict with samples at time t (each [B] or [B, K]).
        """
        meter_logits_t = posterior["meter_logits"][:, t, :]  # [B, K]
        meter_soft = gumbel_softmax_sample(meter_logits_t, temperature=temperature, hard=False)
        meter_hard = gumbel_softmax_sample(meter_logits_t, temperature=temperature, hard=True)

        phase = von_mises_sample(posterior["phase_mu"][:, t], posterior["phase_kappa"][:, t])
        phase = torch.remainder(phase, TWO_PI)

        log_tempo = lognormal_sample_logspace(
            posterior["tempo_mu"][:, t], posterior["tempo_sigma"][:, t],
        )

        return {
            "meter_soft": meter_soft,      # [B, K]
            "meter_onehot": meter_hard,    # [B, K]
            "phase": phase,                # [B]
            "log_tempo": log_tempo,        # [B]
        }

    # ------------------------------------------------------------------
    # Emission decoder
    # ------------------------------------------------------------------

    def decode_at_t(
        self,
        samples: dict[str, Tensor],
        h_prior_t: Tensor,
    ) -> Tensor:
        """Decode beat logit at time t."""
        h = h_prior_t
        if self.h_prior_bottleneck_dim > 0:
            h = self.h_prior_bottleneck_proj(h)
        decoder_input = torch.cat([
            torch.cos(samples["phase"]).unsqueeze(-1),  # [B, 1]
            torch.sin(samples["phase"]).unsqueeze(-1),  # [B, 1]
            samples["log_tempo"].unsqueeze(-1),          # [B, 1]
            samples["meter_soft"],                       # [B, K]
            h,                                           # [B, D'] or [B, D]
        ], dim=-1)
        return self.emission_decoder(decoder_input)  # [B, 2]

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Forward: fully parallel (posterior independent of z_{t-1})
    # ------------------------------------------------------------------

    def _compute_prior_parallel(
        self,
        prior_params: dict[str, Tensor],
        samples: dict[str, Tensor],
        z_prev_gt: dict[str, Tensor],
        beat_targets: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute all prior params in parallel using GT z_prev.

        Prior means use ground-truth phase/tempo/meter (teacher-forced),
        anchoring the prior to correct bar pointer dynamics. The KL then
        forces the posterior to predict values consistent with the GT
        trajectory, not with its own garbage.
        """
        B, T = samples["phase"].shape
        device = samples["phase"].device
        K = self.num_meter_classes

        # Use GT z_prev directly (already shifted: z_prev[t] = GT[t-1])
        phase_prev = z_prev_gt["phase"][:, :T, 0] if z_prev_gt["phase"].ndim == 3 else z_prev_gt["phase"][:, :T]
        log_tempo_prev = z_prev_gt["log_tempo"][:, :T, 0] if z_prev_gt["log_tempo"].ndim == 3 else z_prev_gt["log_tempo"][:, :T]
        meter_prev = z_prev_gt["meter_onehot"][:, :T, :]

        # Phase prior: mu = (phi_{t-1} + phidot_{t-1}) mod 2pi
        tempo_prev = torch.exp(log_tempo_prev.clamp(max=10.0))
        phase_mu = torch.remainder(phase_prev + tempo_prev, TWO_PI)          # [B, T]

        # Override t=0 with initial state
        phase_mu[:, 0] = 0.0

        # Phase kappa: pre-computed from h_prior
        phase_kappa = prior_params["phase_kappa"]  # [B, T]

        # Tempo prior: mu = log(phidot_{t-1}) = log_tempo_prev
        # Use cat to avoid in-place op (preserves gradient to prior_tempo_mu_init)
        tempo_mu_init = self.prior_tempo_mu_init.expand(B).unsqueeze(1)       # [B, 1]
        tempo_mu = torch.cat([tempo_mu_init, log_tempo_prev[:, 1:]], dim=1)   # [B, T]

        # Tempo sigma: beat-gated (tempo can change at beats, locked between beats)
        # Uses GT beat_targets during training to gate tempo changes.
        # At inference (beat_targets=None), falls back to learned sigma everywhere.
        _TEMPO_BETWEEN_SIGMA = 0.03  # moderate lock between beats
        # No beat-gating: GT z_prev already anchors the prior to correct tempo.
        # The learned sigma controls how tightly the posterior must match.
        tempo_sigma = prior_params["tempo_sigma"]

        # Meter boundary detection (phase-based for now)
        boundary = (phase_prev + tempo_prev) >= TWO_PI  # [B, T]
        boundary[:, 0] = False

        eps_ij = prior_params["epsilon_ij"]  # [B, T, K, K]
        diag_mask = torch.eye(K, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
        off_diag_eps = eps_ij * (1.0 - diag_mask)
        row_sum = off_diag_eps.sum(dim=-1, keepdim=True)  # [B, T, K, 1]
        stay_prob = 1.0 - row_sum
        trans_matrix = off_diag_eps + stay_prob * diag_mask  # [B, T, K, K]

        # Gate by boundary
        wrap_mask = boundary.unsqueeze(-1).unsqueeze(-1).float()  # [B, T, 1, 1]
        identity = diag_mask.expand_as(trans_matrix)
        trans_matrix = wrap_mask * trans_matrix + (1.0 - wrap_mask) * identity

        # Apply: pi_t = meter_prev @ trans_matrix
        meter_prev_prob = meter_prev + 1e-6
        meter_prev_prob = meter_prev_prob / meter_prev_prob.sum(dim=-1, keepdim=True)
        meter_logits = torch.log(
            torch.bmm(
                meter_prev_prob.view(B * T, 1, K),
                trans_matrix.view(B * T, K, K),
            ).view(B, T, K) + 1e-10
        )

        # Override t=0 with uniform
        meter_logits[:, 0, :] = 0.0

        return {
            "meter_logits": meter_logits,       # [B, T, K]
            "phase_mu": phase_mu,               # [B, T]
            "phase_kappa": phase_kappa,          # [B, T]
            "tempo_mu": tempo_mu,               # [B, T]
            "tempo_sigma": tempo_sigma,          # [B, T]
        }

    @staticmethod
    def _beat_targets_to_distance(beat_targets: Tensor) -> Tensor:
        """Convert binary beat labels to continuous beat-distance in [0, 1).

        For each frame, computes the fraction of time elapsed since the
        last beat. At a beat frame d=0, then ramps linearly to ~1 just
        before the next beat. This gives the posterior a smooth
        interpolation signal instead of sparse binary input.

        Args:
            beat_targets: [B, T] binary beat indicators.
        Returns:
            beat_distance: [B, T] in [0, 1).
        """
        B, T = beat_targets.shape
        device = beat_targets.device

        # Find beat positions per batch element
        distance = torch.zeros_like(beat_targets)
        for b in range(B):
            beats = torch.where(beat_targets[b] > 0.5)[0]
            if len(beats) == 0:
                # No beats: linear ramp over full sequence
                distance[b] = torch.linspace(0, 1, T, device=device)
                continue
            # Between consecutive beats: linear ramp 0→1
            prev = 0
            for i, beat_pos in enumerate(beats):
                beat_pos = beat_pos.item()
                span = beat_pos - prev
                if span > 0:
                    distance[b, prev:beat_pos] = torch.linspace(0, 1 - 1/max(span, 1), span, device=device)
                prev = beat_pos
            # After last beat
            remaining = T - prev
            if remaining > 0:
                distance[b, prev:] = torch.linspace(0, 1 - 1/max(remaining, 1), remaining, device=device)
        return distance

    def forward(
        self,
        activations: Tensor,
        z_prev_init: dict[str, Tensor],
        temperature: float = 1.0,
        beat_targets: Tensor | None = None,
        downbeat_targets: Tensor | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """Fully parallel forward pass.

        Args:
            activations: [B, T, D_feat] acoustic features.
            z_prev_init: Dict with initial z_prev (t=0 seed).
            temperature: Gumbel-Softmax temperature.
            beat_targets: [B, T] binary beat indicators.
            downbeat_targets: [B, T] binary downbeat indicators.

        Returns:
            Dict with beat_logits, posterior, prior, samples.
        """
        B, T, _ = activations.shape
        device = activations.device

        if beat_targets is None:
            beat_targets = torch.zeros(B, T, device=device)
        if downbeat_targets is None:
            downbeat_targets = torch.zeros(B, T, device=device)

        # Convert binary labels to continuous distances for posterior
        beat_distance = self._beat_targets_to_distance(beat_targets)
        downbeat_distance = self._beat_targets_to_distance(downbeat_targets)

        # Step 1: Prior encoder — pre-compute h_prior and uncertainties (parallel)
        h_prior, prior_params = self.encode_prior(activations)

        # Step 2: Posterior encoder — pre-compute all params (parallel)
        posterior = self.encode_posterior(activations, beat_distance, downbeat_distance)

        # Step 3: Sample ALL timesteps from posterior at once (parallel)
        meter_soft = gumbel_softmax_sample(posterior["meter_logits"], temperature=temperature, hard=False)
        meter_hard = gumbel_softmax_sample(posterior["meter_logits"], temperature=temperature, hard=True)
        kappa_q = posterior["phase_kappa"]
        phase = von_mises_sample(posterior["phase_mu"], kappa_q)
        phase = torch.remainder(phase, TWO_PI)
        sigma_q = posterior["tempo_sigma"]
        log_tempo = lognormal_sample_logspace(posterior["tempo_mu"], sigma_q)

        samples = {
            "meter_soft": meter_soft,       # [B, T, K]
            "meter_onehot": meter_hard,     # [B, T, K]
            "phase": phase,                 # [B, T]
            "log_tempo": log_tempo,         # [B, T]
        }

        # Step 4: Compute prior params using GT z_prev (teacher-forced)
        prior_out = self._compute_prior_parallel(prior_params, samples, z_prev_init, beat_targets)

        # Step 5: Decode ALL timesteps at once (parallel)
        h = h_prior
        if self.h_prior_bottleneck_dim > 0:
            h = self.h_prior_bottleneck_proj(h)
        decoder_input = torch.cat([
            torch.cos(phase).unsqueeze(-1),  # [B, T, 1]
            torch.sin(phase).unsqueeze(-1),  # [B, T, 1]
            log_tempo.unsqueeze(-1),         # [B, T, 1]
            meter_soft,                      # [B, T, K]
            h,                               # [B, T, D]
        ], dim=-1)
        beat_logits = self.emission_decoder(decoder_input)  # [B, T, 2] (beat, downbeat)

        # Reformat posterior for KL compatibility
        posterior_for_loss = {
            "meter_logits": posterior["meter_logits"],
            "phase_mu": posterior["phase_mu"],
            "phase_log_kappa": torch.log(posterior["phase_kappa"] + 1e-8),
            "tempo_mu": posterior["tempo_mu"],
            "tempo_log_sigma": torch.log(posterior["tempo_sigma"] + 1e-8),
        }

        return {
            "beat_logits": beat_logits,
            "posterior": posterior_for_loss,
            "prior": prior_out,
            "samples": samples,
        }
