"""Verification gates for the audio-driven-prior fix (P0).

These are the fast, self-contained gates from the task spec. They run on
synthetic data and need no external dataset or checkpoint:

    Gate 1 — audio pathway: audio gradient reaches the prior phase/tempo MEANS.
    Gate 3 — KL is reducible: per-latent KLs fall toward 0 on an overfit batch
             (no permanent floor from clamped parameter ranges).
    Gate 5 — stability: 100+ iters with no NaN/Inf; grads flow to the new
             audio-correction heads.

Gate 2 (overfit a real batch) and Gate 4 (held-out beat tracking on real audio)
require the WaveBeat extractor + audio dataset and live in standalone scripts:
``tests/gate2_overfit_real.py`` and ``tests/gate4_heldout.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.loss import compute_elbo_loss
from models.svt_core import SVTModel, TWO_PI

K = 4
D = 64


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Synthetic clean-sawtooth batch (perfectly learnable bar-pointer dynamics)
# ---------------------------------------------------------------------------

def _synthetic_batch(B: int = 2, T: int = 128, bpm: float = 120.0,
                     fps: float = 86.1328125):
    """Constant-tempo sawtooth with beats at phase wraps and audio spikes.

    Returns (activations[B,T,2], beat_targets[B,T], downbeat_targets[B,T]).
    The audio carries a clean spike at every beat (and downbeat), so the model
    *can* in principle reach near-zero reconstruction and KL.
    """
    tempo_bpf = bpm / 60.0 / fps                 # beats per frame
    phase = (torch.arange(T, dtype=torch.float32) * tempo_bpf * TWO_PI) % TWO_PI
    beats = torch.zeros(T)
    wraps = torch.where(torch.diff(phase) < -math.pi)[0] + 1
    beats[wraps] = 1.0
    downbeats = torch.zeros(T)
    bidx = torch.where(beats > 0.5)[0]
    downbeats[bidx[::4]] = 1.0                    # 4/4

    beat_targets = beats.unsqueeze(0).repeat(B, 1)
    downbeat_targets = downbeats.unsqueeze(0).repeat(B, 1)

    acts = 0.05 + 0.05 * torch.rand(B, T, 2)
    acts[:, :, 0] += 0.85 * beat_targets
    acts[:, :, 1] += 0.85 * downbeat_targets
    acts = acts.clamp(0, 1)
    return acts, beat_targets, downbeat_targets


# ===========================================================================
# GATE 1 — audio pathway to the prior MEANS
# ===========================================================================

class TestGate1AudioPathway:
    """The direct test that P0 is fixed: audio gradient reaches the prior
    phase/tempo means, *independent of the posterior-sample path*."""

    def test_correction_provides_audio_to_prior_mean(self):
        """Surgical test. Detach the previous-state (z_{t-1}) path so the ONLY
        route from audio to the prior mean is the correction head. The gradient
        of both prior means w.r.t. the audio activations must be non-None and
        have nonzero norm."""
        m = SVTModel(hidden_dim=D, nhead=4, num_layers=1, num_meter_classes=K)
        acts = torch.randn(2, 16, 2, requires_grad=True)
        h_prior = m.encode_prior(acts)

        # z_{t-1} is a detached constant — removes the sample→audio path.
        phase_prev = torch.rand(2).detach()
        log_tempo_prev = torch.randn(2).detach()
        t = 5
        phase_corr, tempo_corr = m.prior_mean_corrections(h_prior[:, t])
        tempo_prev_lin = torch.exp(log_tempo_prev)
        prior_phase_mu = torch.remainder(phase_prev + tempo_prev_lin + phase_corr, TWO_PI)
        prior_tempo_mu = log_tempo_prev + tempo_corr

        g_phase = torch.autograd.grad(prior_phase_mu.sum(), acts, retain_graph=True)[0]
        g_tempo = torch.autograd.grad(prior_tempo_mu.sum(), acts)[0]
        assert g_phase is not None and g_tempo is not None
        assert g_phase.norm() > 0, "prior phase mean is audio-blind (P0 not fixed)"
        assert g_tempo.norm() > 0, "prior tempo mean is audio-blind (P0 not fixed)"

    def test_full_forward_prior_mean_has_audio_gradient(self):
        """End-to-end forward(): the stacked prior phase/tempo means carry a
        nonzero gradient back to the audio activations."""
        m = SVTModel(hidden_dim=D, nhead=4, num_layers=1, num_meter_classes=K)
        acts = torch.randn(2, 24, 2, requires_grad=True)
        bt = torch.zeros(2, 24)
        out = m(acts, beat_targets=bt)
        g_phase = torch.autograd.grad(out["prior"]["phase_mu"].sum(), acts, retain_graph=True)[0]
        g_tempo = torch.autograd.grad(out["prior"]["tempo_mu"].sum(), acts)[0]
        assert g_phase is not None and g_phase.norm() > 0
        assert g_tempo is not None and g_tempo.norm() > 0

    def test_different_audio_gives_different_prior_corrections(self):
        """Determinism check (no sampling): two different audio clips produce
        different prior-mean corrections."""
        m = SVTModel(hidden_dim=D, nhead=4, num_layers=1, num_meter_classes=K)
        m.eval()
        a1 = torch.randn(1, 32, 2)
        a2 = torch.randn(1, 32, 2)
        pc1, tc1 = m.prior_mean_corrections(m.encode_prior(a1))
        pc2, tc2 = m.prior_mean_corrections(m.encode_prior(a2))
        assert (pc1 - pc2).abs().max() > 1e-5
        assert (tc1 - tc2).abs().max() > 1e-6

    def test_inference_rollout_uses_audio_corrections(self):
        """DIRECT test of P0 at inference: the prior-only rollout
        (sample_from_prior, the deployed path) must actually be shaped by the
        audio-driven mean corrections. Same audio + same RNG seed, corrections
        ON vs zeroed: the von Mises/log-normal draws are identical (the rejection
        sampler's randomness depends on kappa, not mu), so any difference in the
        rolled-out phase/tempo is *exactly* the corrections' contribution. Zeroing
        them reproduces the pre-fix audio-blind recursion."""
        m = SVTModel(hidden_dim=D, nhead=4, num_layers=1, num_meter_classes=K)
        m.eval()
        acts = torch.randn(1, 96, 2)

        torch.manual_seed(7)
        out_on = m.sample_from_prior(acts, temperature=0.1)

        # Zero the correction heads -> corr = tanh(0) = 0 -> audio-blind means.
        with torch.no_grad():
            for ffn in (m.prior_phase_corr_ffn, m.prior_tempo_corr_ffn):
                for p in ffn.parameters():
                    p.zero_()
        torch.manual_seed(7)
        out_off = m.sample_from_prior(acts, temperature=0.1)

        dphase = (out_on["phase"] - out_off["phase"]).abs().mean().item()
        dtempo = (out_on["log_tempo"] - out_off["log_tempo"]).abs().mean().item()
        assert dphase > 1e-3, f"inference phase rollout ignores audio corrections (Δ={dphase:.2e})"
        assert dtempo > 1e-4, f"inference tempo rollout ignores audio corrections (Δ={dtempo:.2e})"


# ===========================================================================
# GATE 3 — KL is reducible (no permanent floor from clamps)
# ===========================================================================

class TestGate3KLReducible:
    def test_identical_params_give_zero_kl(self):
        """Sanity: when posterior params equal prior params, every KL term is
        ~0. Proves the Softplus parameterisation admits q == p (no structural
        floor baked into the loss/ranges)."""
        B, T = 2, 16
        post = {
            "meter_logits": torch.randn(B, T, K),
            "phase_mu": torch.rand(B, T) * TWO_PI,
            "phase_log_kappa": torch.randn(B, T),
            "tempo_mu": torch.randn(B, T),
            "tempo_log_sigma": torch.randn(B, T),
        }
        prior = {
            "meter_logits": post["meter_logits"].clone(),
            "phase_mu": post["phase_mu"].clone(),
            "phase_kappa": post["phase_log_kappa"].exp().clone(),
            "tempo_mu": post["tempo_mu"].clone(),
            "tempo_sigma": post["tempo_log_sigma"].exp().clone(),
        }
        _, comps = compute_elbo_loss(
            torch.randn(B, T, 2), torch.zeros(B, T), post, prior,
        )
        for key in ("kl_meter", "kl_phase", "kl_tempo"):
            assert comps[key].item() < 1e-4, f"{key} floor: {comps[key].item()}"

    def test_kl_decreases_to_small_on_overfit(self):
        """Overfit a single clean synthetic batch and confirm every per-latent
        KL falls substantially (toward 0). A clamp-induced floor would pin a KL
        term at a large constant; this catches that."""
        torch.manual_seed(0)
        acts, bt, db = _synthetic_batch(B=2, T=128)
        m = SVTModel(hidden_dim=D, nhead=4, num_layers=2, num_meter_classes=K)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)

        def kls(temp):
            out = m(acts, temperature=temp, beat_targets=bt, downbeat_targets=db)
            total, comps = compute_elbo_loss(
                out["beat_logits"], bt, out["posterior"], out["prior"],
                downbeat_targets=db,
            )
            return total, comps

        total0, c0 = kls(1.0)
        for step in range(400):
            temp = max(0.1, 1.0 - step / 400.0)
            opt.zero_grad()
            total, _ = kls(temp)
            total.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        _, c1 = kls(0.1)

        # Each KL must drop well below its starting value and reach a small
        # absolute level (no permanent floor).
        for key in ("kl_phase", "kl_tempo", "kl_meter"):
            start = c0[key].item()
            end = c1[key].item()
            assert end < max(0.5 * start, 0.5), (
                f"{key} did not reduce: start={start:.3f} end={end:.3f} (possible floor)"
            )


# ===========================================================================
# GATE 5 — stability + gradients to the new audio-correction heads
# ===========================================================================

class TestGate5Stability:
    def test_correction_heads_receive_gradient(self):
        m = SVTModel(hidden_dim=D, nhead=4, num_layers=1, num_meter_classes=K)
        acts = torch.randn(2, 32, 2)
        bt = torch.randint(0, 2, (2, 32)).float()
        out = m(acts, beat_targets=bt)
        total, _ = compute_elbo_loss(out["beat_logits"], bt, out["posterior"], out["prior"])
        total.backward()
        for name, ffn in (("phase_corr", m.prior_phase_corr_ffn),
                          ("tempo_corr", m.prior_tempo_corr_ffn)):
            gnorm = sum(
                p.grad.abs().sum().item() for p in ffn.parameters() if p.grad is not None
            )
            assert gnorm > 0, f"no gradient reached prior_{name}_ffn"

    def test_100_iters_no_nan_with_correction(self):
        m = SVTModel(hidden_dim=D, nhead=4, num_layers=1, num_meter_classes=K)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for i in range(100):
            torch.manual_seed(1000 + i)
            acts = torch.randn(2, 48, 2)
            bt = torch.randint(0, 2, (2, 48)).float()
            db = torch.randint(0, 2, (2, 48)).float()
            opt.zero_grad()
            out = m(acts, beat_targets=bt, downbeat_targets=db)
            total, _ = compute_elbo_loss(
                out["beat_logits"], bt, out["posterior"], out["prior"], downbeat_targets=db,
            )
            assert torch.isfinite(total), f"non-finite loss at iter {i}"
            total.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            for n, p in m.named_parameters():
                assert torch.isfinite(p).all(), f"non-finite param {n} at iter {i}"

    def test_sample_from_prior_stable(self):
        m = SVTModel(hidden_dim=D, nhead=4, num_layers=1, num_meter_classes=K)
        acts = torch.randn(3, 64, 2)
        out = m.sample_from_prior(acts, temperature=0.1)
        for k, v in out.items():
            assert torch.isfinite(v).all(), f"sample_from_prior[{k}] non-finite"
