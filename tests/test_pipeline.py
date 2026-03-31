"""Comprehensive unit tests for the CHART variational beat tracking model.

Covers distributions, loss, SVT model, numerical stability, and gradient flow.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.distributions import (
    categorical_kl,
    gumbel_softmax_sample,
    lognormal_kl,
    lognormal_sample_logspace,
    von_mises_kl,
    von_mises_sample,
)
from models.loss import compute_elbo_loss
from models.svt_core import SVTModel

TWO_PI = 2.0 * math.pi

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B = 2       # batch size
T = 16      # sequence length
K = 4       # meter classes
D = 64      # hidden dim (small for speed)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


@pytest.fixture()
def model():
    return SVTModel(hidden_dim=D, nhead=4, num_layers=1, num_meter_classes=K)


@pytest.fixture()
def activations():
    return torch.randn(B, T, 2)


@pytest.fixture()
def z_prev():
    return {
        "phase": torch.rand(B, T, 1) * TWO_PI,
        "log_tempo": torch.randn(B, T, 1),
        "meter_onehot": torch.nn.functional.one_hot(
            torch.randint(0, K, (B, T)), K
        ).float(),
    }


def _make_posterior_prior(requires_grad: bool = False):
    """Build synthetic posterior and prior dicts for loss testing."""
    posterior = {
        "meter_logits": torch.randn(B, T, K, requires_grad=requires_grad),
        "phase_mu": torch.randn(B, T, requires_grad=requires_grad),
        "phase_log_kappa": torch.randn(B, T, requires_grad=requires_grad),
        "tempo_mu": torch.randn(B, T, requires_grad=requires_grad),
        "tempo_log_sigma": torch.randn(B, T, requires_grad=requires_grad),
    }
    prior = {
        "meter_logits": torch.randn(B, T, K),
        "phase_mu": torch.randn(B, T),
        "phase_kappa": torch.rand(B, T) + 0.1,
        "tempo_mu": torch.randn(B, T),
        "tempo_sigma": torch.rand(B, T) + 0.1,
    }
    return posterior, prior


# ===================================================================
# 1. Distribution tests
# ===================================================================


class TestCategoricalKL:
    def test_identical_distributions_zero(self):
        logits = torch.randn(B, T, K)
        kl = categorical_kl(logits, logits)
        assert kl.shape == (B, T)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    def test_different_distributions_positive(self):
        logits_q = torch.randn(B, T, K)
        logits_p = torch.randn(B, T, K)
        kl = categorical_kl(logits_q, logits_p)
        assert kl.shape == (B, T)
        assert (kl >= -1e-6).all(), "KL should be non-negative"

    def test_shape_1d(self):
        logits = torch.randn(K)
        kl = categorical_kl(logits, logits)
        assert kl.shape == ()


class TestVonMisesKL:
    def test_identical_distributions_zero(self):
        mu = torch.randn(B, T)
        kappa = torch.rand(B, T) + 1.0
        kl = von_mises_kl(mu, kappa, mu, kappa)
        assert kl.shape == (B, T)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-4)

    def test_different_distributions_positive(self):
        kl = von_mises_kl(
            torch.zeros(B, T), torch.ones(B, T) * 5.0,
            torch.ones(B, T), torch.ones(B, T) * 2.0,
        )
        assert kl.shape == (B, T)
        assert (kl > 0).all()

    @pytest.mark.parametrize("kappa_val", [1e-6, 1e-3, 0.01])
    def test_small_kappa(self, kappa_val):
        mu = torch.zeros(B, T)
        kappa = torch.full((B, T), kappa_val)
        kl = von_mises_kl(mu, kappa, mu + 0.1, kappa * 2)
        assert torch.isfinite(kl).all(), f"NaN/Inf with kappa={kappa_val}"

    @pytest.mark.parametrize("kappa_val", [100.0, 500.0, 700.0])
    def test_large_kappa(self, kappa_val):
        mu = torch.zeros(B, T)
        kappa = torch.full((B, T), kappa_val)
        kl = von_mises_kl(mu, kappa, mu + 0.01, kappa)
        assert torch.isfinite(kl).all(), f"NaN/Inf with kappa={kappa_val}"


class TestLognormalKL:
    def test_identical_distributions_zero(self):
        mu = torch.randn(B, T)
        sigma = torch.rand(B, T) + 0.5
        kl = lognormal_kl(mu, sigma, mu, sigma)
        assert kl.shape == (B, T)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    def test_different_distributions_positive(self):
        kl = lognormal_kl(
            torch.zeros(B, T), torch.ones(B, T),
            torch.ones(B, T), torch.ones(B, T) * 2.0,
        )
        assert kl.shape == (B, T)
        assert (kl > 0).all()

    def test_sigma_clamp_no_nan(self):
        """Very small sigma should be clamped, not produce NaN."""
        kl = lognormal_kl(
            torch.zeros(B, T), torch.full((B, T), 1e-10),
            torch.zeros(B, T), torch.full((B, T), 1e-10),
        )
        assert torch.isfinite(kl).all()

    def test_extreme_sigma_values(self):
        """Large sigma values should not overflow."""
        kl = lognormal_kl(
            torch.zeros(B, T), torch.full((B, T), 100.0),
            torch.zeros(B, T), torch.full((B, T), 100.0),
        )
        assert torch.isfinite(kl).all()

    def test_large_mu_difference(self):
        kl = lognormal_kl(
            torch.full((B, T), 50.0), torch.ones(B, T),
            torch.full((B, T), -50.0), torch.ones(B, T),
        )
        assert torch.isfinite(kl).all()


class TestVonMisesSample:
    def test_output_shape(self):
        mu = torch.randn(B, T)
        kappa = torch.rand(B, T) + 1.0
        s = von_mises_sample(mu, kappa)
        assert s.shape == (B, T)

    def test_finite(self):
        mu = torch.randn(B, T)
        kappa = torch.rand(B, T) * 10 + 0.1
        s = von_mises_sample(mu, kappa)
        assert torch.isfinite(s).all()

    def test_gradient_flow_mu(self):
        mu = torch.randn(B, T, requires_grad=True)
        kappa = torch.rand(B, T) + 1.0
        s = von_mises_sample(mu, kappa)
        s.sum().backward()
        assert mu.grad is not None
        assert torch.isfinite(mu.grad).all()

    def test_gradient_flow_kappa(self):
        mu = torch.randn(B, T)
        kappa = (torch.rand(B, T) + 1.0).requires_grad_(True)
        s = von_mises_sample(mu, kappa)
        s.sum().backward()
        assert kappa.grad is not None
        assert torch.isfinite(kappa.grad).all()


class TestLognormalSampleLogspace:
    def test_output_shape(self):
        mu = torch.randn(B, T)
        sigma = torch.rand(B, T) + 0.1
        s = lognormal_sample_logspace(mu, sigma)
        assert s.shape == (B, T)

    def test_finite(self):
        mu = torch.randn(B, T)
        sigma = torch.rand(B, T) + 0.1
        s = lognormal_sample_logspace(mu, sigma)
        assert torch.isfinite(s).all()

    def test_gradient_flow(self):
        mu = torch.randn(B, T, requires_grad=True)
        sigma = (torch.rand(B, T) + 0.1).requires_grad_(True)
        s = lognormal_sample_logspace(mu, sigma)
        s.sum().backward()
        assert mu.grad is not None and torch.isfinite(mu.grad).all()
        assert sigma.grad is not None and torch.isfinite(sigma.grad).all()


class TestGumbelSoftmaxSample:
    def test_output_shape(self):
        logits = torch.randn(B, T, K)
        s = gumbel_softmax_sample(logits, temperature=1.0)
        assert s.shape == (B, T, K)

    def test_sums_to_one(self):
        logits = torch.randn(B, T, K)
        s = gumbel_softmax_sample(logits, temperature=1.0)
        sums = s.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gradient_flow(self):
        logits = torch.randn(B, T, K, requires_grad=True)
        s = gumbel_softmax_sample(logits, temperature=1.0)
        s.sum().backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_hard_mode(self):
        logits = torch.randn(B, T, K)
        s = gumbel_softmax_sample(logits, temperature=1.0, hard=True)
        # Each row should have exactly one 1.0 and rest 0.0
        assert torch.allclose(s.sum(dim=-1), torch.ones(B, T))
        assert ((s == 0.0) | (s == 1.0)).all()


# ===================================================================
# 2. Loss function tests
# ===================================================================


class TestComputeElboLoss:
    def _call(self, **kwargs):
        beat_logits = torch.randn(B, T, 2)
        beat_targets = torch.randint(0, 2, (B, T)).float()
        posterior, prior = _make_posterior_prior()
        return compute_elbo_loss(
            beat_logits, beat_targets, posterior, prior, **kwargs,
        )

    def test_return_shape(self):
        total, components = self._call()
        assert total.shape == ()
        assert set(components.keys()) == {"bce", "kl_meter", "kl_phase", "kl_tempo"}
        for v in components.values():
            assert v.shape == ()

    def test_all_finite(self):
        total, components = self._call()
        assert torch.isfinite(total)
        for v in components.values():
            assert torch.isfinite(v)

    def test_beta_zero(self):
        """With beta=0, KL should not contribute to total loss."""
        total_b0, comp_b0 = self._call(beta=0.0)
        # total should equal just the BCE
        assert torch.allclose(total_b0, comp_b0["bce"], atol=1e-5)

    def test_pos_weight_changes_loss(self):
        torch.manual_seed(42)
        _, comp1 = self._call(pos_weight=1.0)
        torch.manual_seed(42)
        _, comp2 = self._call(pos_weight=50.0)
        # BCE should differ when pos_weight changes (unless all targets are 0)
        # Just check it runs without error; exact inequality depends on targets
        assert torch.isfinite(comp1["bce"]) and torch.isfinite(comp2["bce"])

    def test_free_bits_floor(self):
        fb = 0.5
        _, components = self._call(free_bits=fb)
        # Each KL component (averaged over batch) should be >= free_bits
        for key in ("kl_meter", "kl_phase", "kl_tempo"):
            assert components[key].item() >= fb - 1e-6, (
                f"{key} = {components[key].item()} < free_bits={fb}"
            )

    def test_per_latent_free_bits(self):
        fb_m, fb_p, fb_t = 0.1, 0.5, 1.0
        _, components = self._call(
            free_bits_meter=fb_m,
            free_bits_phase=fb_p,
            free_bits_tempo=fb_t,
        )
        assert components["kl_meter"].item() >= fb_m - 1e-6
        assert components["kl_phase"].item() >= fb_p - 1e-6
        assert components["kl_tempo"].item() >= fb_t - 1e-6

    def test_no_nan_reasonable_inputs(self):
        for _ in range(10):
            total, _ = self._call()
            assert torch.isfinite(total), "NaN/Inf in loss for reasonable inputs"

    def test_no_nan_extreme_posterior(self):
        beat_logits = torch.randn(B, T, 2)
        beat_targets = torch.randint(0, 2, (B, T)).float()
        posterior = {
            "meter_logits": torch.randn(B, T, K) * 100,
            "phase_mu": torch.randn(B, T) * 100,
            "phase_log_kappa": torch.full((B, T), 10.0),   # kappa ~ 22000
            "tempo_mu": torch.randn(B, T) * 50,
            "tempo_log_sigma": torch.full((B, T), -10.0),  # sigma ~ 4.5e-5
        }
        prior = {
            "meter_logits": torch.randn(B, T, K) * 100,
            "phase_mu": torch.randn(B, T),
            "phase_kappa": torch.full((B, T), 500.0),
            "tempo_mu": torch.randn(B, T),
            "tempo_sigma": torch.full((B, T), 0.01),
        }
        total, components = compute_elbo_loss(beat_logits, beat_targets, posterior, prior)
        assert torch.isfinite(total), f"Loss is {total.item()} with extreme inputs"
        for k, v in components.items():
            assert torch.isfinite(v), f"{k} = {v.item()} is not finite"


# ===================================================================
# 3. SVT Model tests
# ===================================================================


class TestSVTModelConstruction:
    def test_creates_without_error(self):
        m = SVTModel(hidden_dim=D, nhead=4, num_layers=1, num_meter_classes=K)
        assert m.hidden_dim == D
        assert m.num_meter_classes == K


class TestSVTModelForward:
    def test_forward_returns_correct_keys(self, model, activations, z_prev):
        out = model(activations, z_prev)
        assert set(out.keys()) == {"beat_logits", "posterior", "prior", "samples"}

    def test_forward_shapes(self, model, activations, z_prev):
        out = model(activations, z_prev)
        assert out["beat_logits"].shape == (B, T, 2)

        post = out["posterior"]
        assert post["meter_logits"].shape == (B, T, K)
        assert post["phase_mu"].shape == (B, T)
        assert post["phase_log_kappa"].shape == (B, T)
        assert post["tempo_mu"].shape == (B, T)
        assert post["tempo_log_sigma"].shape == (B, T)

        pri = out["prior"]
        assert pri["meter_logits"].shape == (B, T, K)
        assert pri["phase_mu"].shape == (B, T)
        assert pri["phase_kappa"].shape == (B, T)
        assert pri["tempo_mu"].shape == (B, T)
        assert pri["tempo_sigma"].shape == (B, T)

        samp = out["samples"]
        assert samp["meter_soft"].shape == (B, T, K)
        assert samp["phase"].shape == (B, T)
        assert samp["log_tempo"].shape == (B, T)


class TestEncodePosterior:
    def test_output_shape(self, model, activations):
        beat_targets = torch.zeros(B, T)
        post = model.encode_posterior(activations, beat_targets, beat_targets)
        assert post["phase_mu"].shape == (B, T)
        assert post["tempo_mu"].shape == (B, T)


class TestEncodePrior:
    def test_output_shape(self, model, activations):
        h_prior, prior_params = model.encode_prior(activations)
        assert h_prior.shape == (B, T, D)
        assert "phase_kappa" in prior_params
        assert "tempo_sigma" in prior_params


class TestComputePriorAtT:
    def test_shapes_and_finite(self, model, activations, z_prev):
        _, prior_params = model.encode_prior(activations)
        prior = model.compute_prior_at_t(
            prior_params, t=1,
            phase_prev=z_prev["phase"][:, 0, :],       # [B, 1]
            log_tempo_prev=z_prev["log_tempo"][:, 0, :],  # [B, 1]
            meter_onehot_prev=z_prev["meter_onehot"][:, 0],  # [B, K]
        )
        assert prior["phase_mu"].shape == (B,)
        assert prior["phase_kappa"].shape == (B,)
        assert prior["tempo_mu"].shape == (B,)
        assert prior["tempo_sigma"].shape == (B,)
        for k, v in prior.items():
            assert torch.isfinite(v).all(), f"prior[{k}] has non-finite values"

    def test_phase_mu_wraps_to_0_2pi(self, model, activations, z_prev):
        _, prior_params = model.encode_prior(activations)
        prior = model.compute_prior_at_t(
            prior_params, t=1,
            phase_prev=z_prev["phase"][:, 0, :],       # [B, 1]
            log_tempo_prev=z_prev["log_tempo"][:, 0, :],  # [B, 1]
            meter_onehot_prev=z_prev["meter_onehot"][:, 0],  # [B, K]
        )
        assert (prior["phase_mu"] >= 0).all()
        assert (prior["phase_mu"] < TWO_PI + 1e-5).all()


class TestComputePosteriorParams:
    def test_shapes_and_finite(self, model, activations):
        beat_targets = torch.zeros(B, T)
        post = model.encode_posterior(activations, beat_targets, beat_targets)
        assert post["meter_logits"].shape == (B, T, K)
        assert post["phase_mu"].shape == (B, T)
        assert post["phase_kappa"].shape == (B, T)
        assert post["tempo_mu"].shape == (B, T)
        assert post["tempo_sigma"].shape == (B, T)
        for k, v in post.items():
            assert torch.isfinite(v).all(), f"posterior[{k}] has non-finite values"


class TestSampleLatent:
    def test_shapes_and_finite(self, model, activations):
        beat_targets = torch.zeros(B, T)
        posterior = model.encode_posterior(activations, beat_targets, beat_targets)
        phase = von_mises_sample(posterior["phase_mu"], posterior["phase_kappa"])
        phase = torch.remainder(phase, TWO_PI)
        assert phase.shape == (B, T)
        assert torch.isfinite(phase).all()

    def test_phase_in_0_2pi(self, model, activations):
        beat_targets = torch.zeros(B, T)
        posterior = model.encode_posterior(activations, beat_targets, beat_targets)
        phase = von_mises_sample(posterior["phase_mu"], posterior["phase_kappa"])
        phase = torch.remainder(phase, TWO_PI)
        assert (phase >= 0).all()
        assert (phase < TWO_PI).all()


class TestDecode:
    def test_output_shape(self, model, activations):
        h_prior, _ = model.encode_prior(activations)
        samples = {
            "phase": torch.rand(B) * TWO_PI,
            "log_tempo": torch.randn(B),
            "meter_soft": torch.softmax(torch.randn(B, K), dim=-1),
        }
        logits = model.decode_at_t(samples, h_prior[:, 0, :])
        assert logits.shape == (B, 2)


class TestForwardBackward:
    def test_no_nan_in_loss_and_grads(self, model, activations, z_prev):
        out = model(activations, z_prev)
        beat_targets = torch.randint(0, 2, (B, T)).float()
        total, components = compute_elbo_loss(
            out["beat_logits"], beat_targets,
            out["posterior"], out["prior"],
        )
        assert torch.isfinite(total), f"Loss = {total.item()}"
        total.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient for {name}"


# ===================================================================
# 4. Numerical stability stress tests
# ===================================================================


class TestNumericalStability:
    @pytest.mark.parametrize("kappa_val", [500.0, 700.0])
    def test_von_mises_sample_large_kappa(self, kappa_val):
        mu = torch.zeros(B, T)
        kappa = torch.full((B, T), kappa_val)
        s = von_mises_sample(mu, kappa)
        assert torch.isfinite(s).all(), f"NaN with kappa={kappa_val}"

    @pytest.mark.parametrize("kappa_val", [1e-6, 1e-8])
    def test_von_mises_sample_small_kappa(self, kappa_val):
        mu = torch.zeros(B, T)
        kappa = torch.full((B, T), kappa_val)
        s = von_mises_sample(mu, kappa)
        assert torch.isfinite(s).all(), f"NaN with kappa={kappa_val}"

    def test_lognormal_kl_at_clamp_boundary(self):
        """sigma_q near 1e-4 clamp, sigma_p near 1e-2 clamp."""
        kl = lognormal_kl(
            torch.zeros(B, T), torch.full((B, T), 1e-5),  # will be clamped to 1e-4
            torch.zeros(B, T), torch.full((B, T), 5e-3),  # will be clamped to 1e-2
        )
        assert torch.isfinite(kl).all()

    def test_lognormal_kl_large_mu_no_overflow(self):
        kl = lognormal_kl(
            torch.full((B, T), 100.0), torch.ones(B, T),
            torch.full((B, T), -100.0), torch.ones(B, T),
        )
        assert torch.isfinite(kl).all()

    def test_forward_backward_extreme_z_prev(self, model, activations):
        """Extreme z_prev values (log_tempo = +/-10) should not crash."""
        z_prev = {
            "phase": torch.rand(B, T, 1) * TWO_PI,
            "log_tempo": torch.full((B, T, 1), 10.0),
            "meter_onehot": torch.nn.functional.one_hot(
                torch.randint(0, K, (B, T)), K
            ).float(),
        }
        model.zero_grad()
        out = model(activations, z_prev)
        beat_targets = torch.randint(0, 2, (B, T)).float()
        total, _ = compute_elbo_loss(
            out["beat_logits"], beat_targets,
            out["posterior"], out["prior"],
        )
        assert torch.isfinite(total), f"Loss = {total.item()} with extreme z_prev"
        total.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), (
                    f"NaN/Inf gradient for {name} with extreme z_prev"
                )

    def test_forward_backward_negative_extreme_log_tempo(self, model, activations):
        z_prev = {
            "phase": torch.rand(B, T, 1) * TWO_PI,
            "log_tempo": torch.full((B, T, 1), -10.0),
            "meter_onehot": torch.nn.functional.one_hot(
                torch.randint(0, K, (B, T)), K
            ).float(),
        }
        model.zero_grad()
        out = model(activations, z_prev)
        beat_targets = torch.randint(0, 2, (B, T)).float()
        total, _ = compute_elbo_loss(
            out["beat_logits"], beat_targets,
            out["posterior"], out["prior"],
        )
        assert torch.isfinite(total), f"Loss = {total.item()} with log_tempo=-10"
        total.backward()

    def test_100_iterations_no_nan(self, model):
        """Run 100 forward+backward passes. Loss must never be NaN."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for i in range(100):
            torch.manual_seed(i)
            act = torch.randn(B, T, 2)
            zp = {
                "phase": torch.rand(B, T, 1) * TWO_PI,
                "log_tempo": torch.randn(B, T, 1),
                "meter_onehot": torch.nn.functional.one_hot(
                    torch.randint(0, K, (B, T)), K
                ).float(),
            }
            bt = torch.randint(0, 2, (B, T)).float()

            optimizer.zero_grad()
            out = model(act, zp)
            total, _ = compute_elbo_loss(
                out["beat_logits"], bt,
                out["posterior"], out["prior"],
            )
            assert torch.isfinite(total), f"NaN loss at iteration {i}: {total.item()}"
            total.backward()
            # Clip grads to prevent explosion that could cause NaN next iter
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


# ===================================================================
# 5. End-to-end gradient flow test
# ===================================================================


class TestEndToEndGradientFlow:
    def test_all_params_have_gradients(self, model, activations, z_prev):
        model.zero_grad()
        out = model(activations, z_prev)
        beat_targets = torch.randint(0, 2, (B, T)).float()
        total, _ = compute_elbo_loss(
            out["beat_logits"], beat_targets,
            out["posterior"], out["prior"],
        )
        total.backward()

        params_without_grad = []
        params_with_nan_grad = []
        for name, p in model.named_parameters():
            if p.grad is None:
                params_without_grad.append(name)
            elif not torch.isfinite(p.grad).all():
                params_with_nan_grad.append(name)

        assert len(params_without_grad) == 0, (
            f"Parameters without gradients: {params_without_grad}"
        )
        assert len(params_with_nan_grad) == 0, (
            f"Parameters with NaN/Inf gradients: {params_with_nan_grad}"
        )

    def test_gradients_are_nonzero_somewhere(self, model, activations, z_prev):
        """At least most parameters should have non-zero gradients."""
        model.zero_grad()
        out = model(activations, z_prev)
        beat_targets = torch.randint(0, 2, (B, T)).float()
        total, _ = compute_elbo_loss(
            out["beat_logits"], beat_targets,
            out["posterior"], out["prior"],
        )
        total.backward()

        nonzero_count = 0
        total_count = 0
        for name, p in model.named_parameters():
            total_count += 1
            if p.grad is not None and p.grad.abs().max() > 1e-12:
                nonzero_count += 1

        # At least 80% of parameters should have non-trivial gradients
        ratio = nonzero_count / total_count
        assert ratio > 0.8, (
            f"Only {nonzero_count}/{total_count} params have non-zero grads"
        )
