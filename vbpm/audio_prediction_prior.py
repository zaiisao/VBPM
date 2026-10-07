"""Sohn-style prior feedback from a baseline audio prediction.

The frozen feature frontend's original output head provides an initial output
prediction. No annotation is accepted by this module. A trainable conditional
prior corrects its initial phase and tempo trajectory using audio and prediction
features; the observation network remains a separate, trainable module.
"""

from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from .constants import KAPPA_MIN
from .nets import PriorModel
from .specs import PriorSpec, WalkSpec


class AudioPredictionPrior(PriorModel):
    """Generate initial phase and tempo proposals from frozen audio predictions."""

    def __init__(
        self,
        weight,
        bias,
        meters=(3, 4),
        walk=None,
        spec=None,
        fps=50.0,
        proposal_meter=4,
        residual_scale=0.1,
        trajectory_basis="linear",
    ):
        spec = spec or PriorSpec()
        super().__init__(weight.shape[1] + 5, meters, walk or WalkSpec(), spec)
        self.register_buffer("prediction_weight", weight.detach().clone())
        self.register_buffer("prediction_bias", bias.detach().clone())
        self.fps = float(fps)
        self.proposal_meter = int(proposal_meter)
        self.residual_scale = float(residual_scale)
        if trajectory_basis not in ("linear", "framewise"):
            raise ValueError("unknown tempo trajectory basis")
        self.trajectory_basis = trajectory_basis
        # Log concentration, rather than a large softplus raw value, gives
        # initial-phase uncertainty an effective multiplicative learning scale.
        with torch.no_grad():
            self.phase0_head.bias[2] = math.log(spec.phase0_kappa - KAPPA_MIN)
            self.meter0_head.weight.zero_()
            self.meter0_head.bias.copy_(self.log_meter0_table)

    def initial_prediction(
        self, h, mask, max_bpm=None, min_probability=None, recover_missing_beats=False
    ):
        """Audio-only beat prediction and interpolated phase/velocity proposal."""
        if max_bpm is not None and max_bpm <= 0:
            raise ValueError("Proposal maximum BPM must be positive")
        if min_probability is not None and not 0 <= min_probability < 1:
            raise ValueError("Proposal probability threshold must be in [0,1)")
        logits = F.linear(h, self.prediction_weight, self.prediction_bias)
        beat_logits = logits[..., 0] + logits[..., 1]
        down_logits = logits[..., 1]
        B, T = mask.shape
        phi = h.new_zeros(B, T)
        velocity = h.new_full((B, T), 2 * math.pi * 2 / (self.proposal_meter * self.fps))
        bar_confidence = h.new_zeros(B, self.proposal_meter)
        # The baseline predictor is fixed. Event extraction only constructs
        # features/initial coordinates; gradients still reach h through the
        # trainable encoder and through its raw prediction-logit inputs.
        with torch.no_grad():
            probability = beat_logits.sigmoid()
            maxima = F.max_pool1d(probability[:, None], 7, stride=1, padding=3)[:, 0]
            frames = torch.arange(T, device=h.device, dtype=h.dtype)
            for b in range(B):
                confident = (
                    probability[b] >= 0.3
                    if min_probability is None
                    else probability[b] > min_probability
                )
                peaks = torch.nonzero(
                    (probability[b] >= maxima[b]) & confident & (mask[b] > 0)
                ).flatten()
                if peaks.numel() < 2:
                    phi[b] = frames * velocity[b, 0]
                    continue
                positions = peaks.to(h.dtype)
                # Refine the baseline peak using its logit curvature, without
                # looking at beat annotations or using true tempo.
                inside = (peaks > 0) & (peaks < T - 1)
                k = peaks[inside]
                left, center, right = (
                    beat_logits[b, k - 1],
                    beat_logits[b, k],
                    beat_logits[b, k + 1],
                )
                denom = left - 2 * center + right
                shift = torch.where(
                    denom.abs() > 1e-6, 0.5 * (left - right) / denom, torch.zeros_like(denom)
                )
                positions[inside] += shift.clamp(-0.5, 0.5)
                if max_bpm is not None:
                    # Optional staged control: remove contradictory close
                    # peaks by audio confidence, after subframe refinement.
                    selected = []
                    for candidate in (
                        probability[b, peaks].argsort(descending=True, stable=True).tolist()
                    ):
                        if all(
                            abs(float(positions[candidate] - positions[j]))
                            >= 60 * self.fps / max_bpm
                            for j in selected
                        ):
                            selected.append(candidate)
                    selected = sorted(selected)
                    peaks = peaks[selected]
                    positions = positions[selected]
                    if len(selected) < 2:
                        phi[b] = frames * velocity[b, 0]
                        continue
                jumps = torch.ones(len(peaks) - 1, device=h.device, dtype=torch.long)
                if recover_missing_beats:
                    # An isolated near-integer long gap, bounded by consistent
                    # periods, can be a missed audio peak. Preserve ordinal
                    # count rather than slowing the entire clock in that gap.
                    gaps = positions[1:] - positions[:-1]
                    for j in range(1, len(gaps) - 1):
                        neighbors = torch.cat(
                            (gaps[max(0, j - 2) : j], gaps[j + 1 : min(len(gaps), j + 3)])
                        )
                        period = neighbors.median()
                        ratio = gaps[j] / period.clamp_min(1.0)
                        count = int(ratio.round())
                        adjacent_consistent = bool(
                            ((gaps[j - 1 : j + 2 : 2] / period - 1).abs() <= 0.25).all()
                        )
                        if (
                            2 <= count <= 3
                            and abs(float(ratio / count - 1)) <= 0.15
                            and adjacent_consistent
                        ):
                            jumps[j] = count
                ordinal = torch.cat((jumps.new_zeros(1), jumps.cumsum(0)))
                scores = torch.stack(
                    [
                        down_logits[b, peaks][ordinal % self.proposal_meter == g].sum()
                        for g in range(self.proposal_meter)
                    ]
                )
                bar_confidence[b] = scores.softmax(0)
                offset = scores.argmax()
                interval = (
                    torch.searchsorted(positions.contiguous(), frames.contiguous()) - 1
                ).clamp(0, len(peaks) - 2)
                duration = (positions[interval + 1] - positions[interval]).clamp(min=1.0)
                beat_position = ordinal[interval].to(h.dtype) + (
                    frames - positions[interval]
                ) / duration * jumps[interval].to(h.dtype)
                phi[b] = (beat_position - offset.to(h.dtype)) * (2 * math.pi / self.proposal_meter)
                velocity[b] = (
                    (2 * math.pi / self.proposal_meter) / duration * jumps[interval].to(h.dtype)
                )
        return {
            "phase": phi,
            "physical_velocity": velocity,
            "beat_logits": beat_logits,
            "downbeat_logits": down_logits,
            "bar_confidence": bar_confidence,
        }

    def forward(self, h, mask=None, velocity_ref=None):
        """Evaluate the network on its input tensors."""
        if velocity_ref is None:
            raise ValueError("AudioPredictionPrior requires positive log-velocity units.")
        if mask is None:
            mask = h.new_ones(h.shape[:2])
        guess = self.initial_prediction(h, mask)
        guess_log_velocity = (guess["physical_velocity"] / velocity_ref).log()
        cues = torch.stack(
            [
                guess["beat_logits"],
                guess["downbeat_logits"],
                guess["phase"].cos(),
                guess["phase"].sin(),
                guess_log_velocity,
            ],
            -1,
        )
        p = super().forward(torch.cat([h, cues], -1), mask, velocity_ref)
        weights = mask / mask.sum(1, keepdim=True).clamp(min=1)
        proposal_anchor = (guess["phase"] * weights).sum(1)
        pooled = (p["feats"] * weights[..., None]).sum(1)
        # An unbounded residual angle can correct an incorrect baseline bar
        # offset; multiplying an atan2 output would cap that correction.
        correction = self.phase0_head(pooled)[:, 1]
        p["phase_anchor"] = proposal_anchor + self.residual_scale * correction
        residual = p["velocity_mean"]
        if self.trajectory_basis == "linear":
            time = torch.linspace(-1.0, 1.0, h.shape[1], device=h.device, dtype=h.dtype)[None]
            center = time - (time * weights).sum(1, keepdim=True)
            level = (residual * weights).sum(1, keepdim=True)
            slope = (residual * center * weights).sum(1, keepdim=True) / (
                center.square() * weights
            ).sum(1, keepdim=True).clamp(min=1e-6)
            residual = level + slope * center
        p["velocity_mean"] = guess_log_velocity + self.residual_scale * residual
        p["velocity0"] = (p["velocity_mean"][:, 0], p["velocity0"][1])
        physical = velocity_ref * p["velocity_mean"].exp()
        relative = torch.cat(
            [torch.zeros_like(physical[:, :1]), (physical[:, :-1] * mask[:, 1:]).cumsum(1)], 1
        )
        concentration = KAPPA_MIN + self.phase0_head(pooled)[:, 2].exp()
        p["phase0"] = (p["phase_anchor"] - (relative * weights).sum(1), concentration)
        p["initial_prediction"] = guess["phase"]
        return p
