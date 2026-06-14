"""Stability tests targeting the NaN-at-step-thousands instability.

Tier 1: boundary tests on distribution primitives at the exact arithmetic
edges the trained model is known to push against (kappa at _KAPPA_MAX,
posterior kappa at its 500 ceiling, log-tempo gap up to ~15 in log-space
when the dataset produces near-silence frames).

Tier 3: long-running soaks (marked @pytest.mark.slow) that replay the
training inner loop on synthetic-but-realistic data for many thousands of
steps. The existing test_pipeline::test_100_iterations_no_nan covers 100
steps with a small model and random inputs; the train_v7.log failure was
at step ~thousands with the real hparams. Run with `pytest -m slow`.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.distributions import (
    _KAPPA_MAX,
    categorical_kl,
    lognormal_kl,
    von_mises_kl,
    von_mises_sample,
)
from models.loss import compute_elbo_loss
from models.svt_core import SVTModel

TWO_PI = 2.0 * math.pi


# ===========================================================================
# Tier 1: boundary tests on distribution primitives
# ===========================================================================


class TestVonMisesBoundary:
    """The von Mises sampler/backward at the kappa clamps and at the
    series<->normal CDF branch seam (kappa = 10.5).
    """

    @pytest.mark.parametrize(
        "kappa_val",
        [_KAPPA_MAX - 1.0, _KAPPA_MAX, _KAPPA_MAX + 100.0],
    )
    def test_grad_finite_at_kappa_max(self, kappa_val: float) -> None:
        """At and past the _KAPPA_MAX clamp, gradients on both mu and
        kappa must be finite. This is the worst case for the
        `inv_p = exp(-kappa * cosxm1(z)) * 2pi * i0e(kappa)` cancellation
        in the implicit-reparam backward; if the cancellation breaks,
        gradient becomes Inf*0 = NaN."""
        torch.manual_seed(0)
        for trial in range(50):
            mu = torch.randn(8, 32, requires_grad=True)
            kappa = torch.full((8, 32), float(kappa_val), requires_grad=True)
            samples = von_mises_sample(mu, kappa)
            assert torch.isfinite(samples).all(), (
                f"non-finite sample at trial {trial}, kappa={kappa_val}"
            )
            (samples ** 2).mean().backward()
            assert torch.isfinite(mu.grad).all(), (
                f"non-finite mu.grad at trial {trial}, kappa={kappa_val}"
            )
            assert torch.isfinite(kappa.grad).all(), (
                f"non-finite kappa.grad at trial {trial}, kappa={kappa_val}"
            )

    @pytest.mark.parametrize("kappa_val", [10.0, 10.4, 10.5, 10.6, 11.0])
    def test_grad_finite_across_branch_seam(self, kappa_val: float) -> None:
        """Series CDF (kappa < 10.5) and normal CDF (kappa >= 10.5) are
        two different code paths. Gradient must be finite on both sides
        and at the seam."""
        torch.manual_seed(42)
        mu = torch.zeros(64)
        kappa = torch.full((64,), float(kappa_val), requires_grad=True)
        samples = von_mises_sample(mu, kappa)
        samples.mean().backward()
        assert torch.isfinite(kappa.grad).all(), (
            f"non-finite kappa.grad at branch boundary kappa={kappa_val}"
        )

    def test_kl_grad_at_model_bounds(self) -> None:
        """Posterior kappa at 500 (its sigmoid-bounded ceiling),
        prior kappa at 100 (the prior's lower sigmoid bound), with a
        pi/2 phase gap. The realistic worst-case the trained model can
        actually emit. KL must be finite, positive, and yield finite
        gradients on both posterior parameters."""
        mu_q = torch.zeros(4, 64, requires_grad=True)
        log_kappa_q = torch.full((4, 64), math.log(500.0), requires_grad=True)
        mu_p = torch.full((4, 64), math.pi / 2)
        kappa_p = torch.full((4, 64), 100.0)

        kl = von_mises_kl(mu_q, log_kappa_q.exp(), mu_p, kappa_p)
        assert torch.isfinite(kl).all(), f"non-finite KL: {kl}"
        assert (kl > 0).all(), "KL must be non-negative for non-identical distributions"

        kl.mean().backward()
        assert torch.isfinite(mu_q.grad).all()
        assert torch.isfinite(log_kappa_q.grad).all()


class TestLogNormalBoundary:
    """Tempo KL is the most likely NaN source: the dataset produces
    log_tempo as low as log(1e-8) ~ -18 in silence/pre-first-beat
    regions, but the posterior tempo_mu is bounded at [-3.0, -1.2]. The
    gap of ~15 in log-space combined with sigma_p clamped at 1e-2
    produces per-frame KL on the order of 10^6 nats — well-defined but
    enormous, with gradients in the 10^4--10^5 range.
    """

    def test_kl_finite_at_extreme_mu_gap(self) -> None:
        """Posterior at its lower bound, prior reflecting near-silence."""
        mu_q = torch.full((2, 16), -3.0, requires_grad=True)
        sigma_q = torch.full((2, 16), 0.01, requires_grad=True)
        mu_p = torch.full((2, 16), -18.0)
        sigma_p = torch.full((2, 16), 0.01)

        kl = lognormal_kl(mu_q, sigma_q, mu_p, sigma_p)
        assert torch.isfinite(kl).all(), f"non-finite KL: {kl}"
        # The whole point — KL is huge but the test must still pass.
        # If you change the clamp/bound logic this expectation may shift.
        assert kl.mean().item() > 1e5, (
            f"sanity: extreme-gap KL should be very large, got {kl.mean().item()}"
        )

        kl.mean().backward()
        assert torch.isfinite(mu_q.grad).all()
        assert torch.isfinite(sigma_q.grad).all()

    def test_kl_finite_at_sigma_clamps(self) -> None:
        """Both sigmas below their respective clamp floors (1e-4 for q,
        1e-2 for p). The clamps should kick in cleanly with finite
        gradient flow."""
        mu_q = torch.zeros(32, requires_grad=True)
        sigma_q = torch.full((32,), 1e-8, requires_grad=True)
        mu_p = torch.zeros(32)
        sigma_p = torch.full((32,), 1e-8)

        kl = lognormal_kl(mu_q, sigma_q, mu_p, sigma_p)
        assert torch.isfinite(kl).all()

        kl.mean().backward()
        assert torch.isfinite(mu_q.grad).all()
        assert torch.isfinite(sigma_q.grad).all()


class TestCategoricalBoundary:
    def test_kl_finite_at_collapsed_logits(self) -> None:
        """Post-collapse meter posterior: one class with overwhelming
        confidence, others crushed. Softmax must not overflow."""
        K = 8
        logits_q = torch.full((4, 16, K), -50.0, requires_grad=True)
        with torch.no_grad():
            logits_q[:, :, 0] = 50.0
        logits_p = torch.zeros(4, 16, K)

        kl = categorical_kl(logits_q, logits_p)
        assert torch.isfinite(kl).all(), f"non-finite KL: {kl}"

        kl.mean().backward()
        assert torch.isfinite(logits_q.grad).all()


# ===========================================================================
# Tier 3: long-soak training reproductions
# ===========================================================================


def _make_synthetic_batch(
    B: int,
    T: int,
    K: int = 8,
    bpm: float = 100.0,
    fps: float = 86.1328125,
    inject_silence: bool = False,
) -> dict[str, torch.Tensor]:
    """Synthetic batch matching the trainer's expected shapes & semantics.

    The phase trajectory is a sawtooth at the given BPM, with beat
    targets at every wrap and downbeat targets every 4th beat. When
    inject_silence=True, one batch element gets log_tempo at log(1e-8)
    -- exercising the extreme prior-vs-posterior tempo gap.
    """
    tempo_bpf = bpm / 60.0 / fps
    phase_traj = (torch.arange(T, dtype=torch.float32) * tempo_bpf * TWO_PI) % TWO_PI

    # Per-batch log_tempo trajectory (constant unless silence injected)
    log_tempo_normal = torch.full((T,), math.log(tempo_bpf * TWO_PI))
    log_tempo_rows = []
    phase_rows = []
    for b in range(B):
        if inject_silence and b == B - 1:
            log_tempo_rows.append(torch.full((T,), math.log(1e-8)))
        else:
            log_tempo_rows.append(log_tempo_normal)
        phase_rows.append(phase_traj.clone())
    log_tempo = torch.stack(log_tempo_rows)  # [B, T]
    phase = torch.stack(phase_rows)  # [B, T]

    # Beat targets at phase wraps
    beat_targets = torch.zeros(B, T)
    for b in range(B):
        wraps = torch.where(torch.diff(phase[b]) < -math.pi)[0] + 1
        beat_targets[b, wraps] = 1.0
    # Downbeats every 4th beat (4/4)
    downbeat_targets = torch.zeros(B, T)
    for b in range(B):
        beat_idx = torch.where(beat_targets[b] > 0.5)[0]
        if len(beat_idx) > 0:
            downbeat_targets[b, beat_idx[::4]] = 1.0

    # z_prev: shifted-by-1 GT trajectories
    phase_prev = torch.zeros(B, T, 1)
    phase_prev[:, 1:, 0] = phase[:, :-1]
    log_tempo_prev = torch.zeros(B, T, 1)
    log_tempo_prev[:, 1:, 0] = log_tempo[:, :-1]
    meter_prev = torch.zeros(B, T, K)
    meter_prev[:, :, 3] = 1.0  # 4/4 = class index 3 in the K=8 vocabulary

    # Activations: WaveBeat-style sigmoid output, with a spike at each beat
    activations = 0.05 + 0.05 * torch.rand(B, T, 2)
    activations[:, :, 0] += 0.8 * beat_targets
    activations[:, :, 1] += 0.8 * downbeat_targets
    activations = activations.clamp(0, 1)

    return {
        "activations": activations,
        "beat_targets": beat_targets,
        "downbeat_targets": downbeat_targets,
        "phase_prev": phase_prev,
        "log_tempo_prev": log_tempo_prev,
        "meter_onehot_prev": meter_prev,
    }


def _run_soak(
    *,
    num_steps: int,
    hidden_dim: int,
    num_layers: int,
    B: int,
    T: int,
    silence_every: int = 100,
    seed: int = 0,
    device: torch.device | None = None,
) -> int | None:
    """Run a synthetic training loop. Returns the step at which a NaN
    first appeared, or None if no NaN occurred."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    K = 8
    model = SVTModel(
        hidden_dim=hidden_dim,
        nhead=4,
        num_layers=num_layers,
        num_meter_classes=K,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(num_steps):
        inject = silence_every > 0 and step % silence_every == silence_every - 1
        batch_cpu = _make_synthetic_batch(B, T, K=K, inject_silence=inject)
        batch = {k: v.to(device) for k, v in batch_cpu.items()}
        optimizer.zero_grad()
        out = model(
            batch["activations"],
            beat_targets=batch["beat_targets"],
            downbeat_targets=batch["downbeat_targets"],
        )
        loss, _ = compute_elbo_loss(
            out["beat_logits"],
            batch["beat_targets"],
            out["posterior"],
            out["prior"],
            downbeat_targets=batch["downbeat_targets"],
            free_bits_meter=0.1,
            free_bits_phase=0.1,
            free_bits_tempo=0.1,
        )

        if not torch.isfinite(loss):
            return step

        loss.backward()

        # Mirror the trainer's gradient-NaN guard
        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                pytest.fail(
                    f"non-finite gradient at step {step}: {name} "
                    f"(loss={loss.item():.6g})"
                )

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Catch parameters going non-finite from optimizer step
        for name, p in model.named_parameters():
            if not torch.isfinite(p).all():
                pytest.fail(f"non-finite parameter after step {step}: {name}")

    return None


class TestLongSoak:
    def test_500_steps_small_model(self) -> None:
        """500 steps with a reduced model (~30s on CPU). Catches
        regressions visible in the first few hundred steps."""
        first_nan = _run_soak(
            num_steps=500,
            hidden_dim=64,
            num_layers=1,
            B=2,
            T=128,
            silence_every=50,
        )
        assert first_nan is None, f"NaN at step {first_nan}"

    @pytest.mark.slow
    def test_5000_steps_full_hparams(self) -> None:
        """Full production hparams (hidden_dim=128, num_layers=2,
        T=256), 5000 steps, with a silence batch every 100 steps. This
        is the closest pure-PyTorch reproduction of the train_v7.log
        failure available without the real dataloader. Runs in roughly
        5-10 min on a single GPU."""
        first_nan = _run_soak(
            num_steps=5000,
            hidden_dim=128,
            num_layers=2,
            B=2,
            T=256,
            silence_every=100,
        )
        assert first_nan is None, f"NaN at step {first_nan}"
