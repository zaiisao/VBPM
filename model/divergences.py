"""Optional modules implementing ablations -- deliberate departures from the VBPM model.

None of these are part of the derived ELBO. They are toggled by config flags so we can measure, in
isolation, how each one changes the model's behaviour:
  * AutocorrelationTempoHead -- computes the tempo from features (instead of inferring it as a latent).
  * GeometricEmission        -- a fixed-form likelihood beat~cos(M*phi), downbeat~cos(phi).
  * blend_phase              -- the Kalman-style predict/correct phase update.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TWO_PI = 2.0 * math.pi

# Candidate beat periods (in frames) the autocorrelation head searches over: ~40-258 BPM at 86 fps.
AUTOCORR_LAG_FRAMES = np.arange(20, 140)


class AutocorrelationTempoHead(nn.Module):
    """Differentiable tempogram: learned onset strength -> autocorrelation over lags -> tempo.

    Tempo is a *rate* (a periodicity), which a pointwise operation can't read; this measures the period
    of a learned onset signal via windowed autocorrelation -- the correct operator. Returns a softmax-able
    score over candidate lags; the argmax lag gives the beat period (and thus the bar-phase advance).
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.onset_projection = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.score_scale = nn.Parameter(torch.tensor(5.0))  # sharpens the lag distribution

    def lag_scores(self, features: torch.Tensor) -> torch.Tensor:
        """features [batch, num_frames, feature_dim] -> autocorrelation score per lag [batch, num_lags]."""
        onset = F.softplus(self.onset_projection(features).squeeze(-1))  # [batch, num_frames]
        onset = onset - onset.mean(dim=1, keepdim=True)
        num_frames = onset.shape[1]
        onset_energy = onset.pow(2).mean(dim=1) + 1e-6
        per_lag_autocorrelation = []
        for lag in AUTOCORR_LAG_FRAMES:
            # normalized autocorrelation at this lag = <onset[:-lag], onset[lag:]> / energy
            per_lag_autocorrelation.append((onset[:, : num_frames - lag] * onset[:, lag:]).mean(dim=1) / onset_energy)
        autocorrelation = torch.stack(per_lag_autocorrelation, dim=1)  # [batch, num_lags]
        return autocorrelation * F.softplus(self.score_scale)

    def bar_phase_advance(self, features: torch.Tensor, beats_per_bar: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-clip bar-phase advance (radians/frame) from the best lag, plus the lag scores (for CE loss).

        The advance is detached: the tempo head is trained by its own cross-entropy against the ground-truth
        period (see losses.py), not by gradients flowing back from the phase rollout.
        """
        scores = self.lag_scores(features)
        best_lag_index = scores.argmax(dim=1)
        beat_period_frames = torch.tensor(AUTOCORR_LAG_FRAMES, device=features.device,
                                          dtype=features.dtype)[best_lag_index]
        # bar advances 2*pi per bar = per (beats_per_bar) beats; so per frame it is 2*pi/(beats_per_bar*period).
        bar_phase_advance_per_frame = TWO_PI / (beats_per_bar * beat_period_frames)
        return bar_phase_advance_per_frame.detach(), scores


class GeometricEmission(nn.Module):
    """Fixed-form likelihood: beat logit = a_b*cos(beats_per_bar*phi)+c_b, downbeat = a_d*cos(phi)+c_d.

    VBPM beats are geometrically determined by the phase; this makes the *likelihood*
    reflect that (vs a free MLP decoder). a_* are kept positive so the cosines peak at the event phases.
    """

    def __init__(self):
        super().__init__()
        self.beat_amplitude_raw = nn.Parameter(torch.tensor(1.0))
        self.downbeat_amplitude_raw = nn.Parameter(torch.tensor(1.0))
        self.beat_bias = nn.Parameter(torch.tensor(-1.0))
        self.downbeat_bias = nn.Parameter(torch.tensor(-1.0))

    def logits(self, phase: torch.Tensor, beats_per_bar: int) -> torch.Tensor:
        """phase [...] -> [..., 2] logits for (beat, downbeat)."""
        beat_logit = F.softplus(self.beat_amplitude_raw) * torch.cos(beats_per_bar * phase) + self.beat_bias
        downbeat_logit = F.softplus(self.downbeat_amplitude_raw) * torch.cos(phase) + self.downbeat_bias
        return torch.stack([beat_logit, downbeat_logit], dim=-1)


def blend_phase(predicted_phase: torch.Tensor, data_phase: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """Kalman-style circular update: combine a prediction and a data reading on the unit circle.

    gain=0 -> trust the prediction entirely (pure integrator); gain=1 -> trust the data entirely (free
    posterior). A learned gain in between predicts via the tempo then corrects toward the audio, which is
    what avoids the pure integrator's drift while still using the tempo.
    """
    cos_component = (1.0 - gain) * torch.cos(predicted_phase) + gain * torch.cos(data_phase)
    sin_component = (1.0 - gain) * torch.sin(predicted_phase) + gain * torch.sin(data_phase)
    return torch.atan2(sin_component, cos_component) % TWO_PI
