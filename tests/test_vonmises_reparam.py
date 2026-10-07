"""The tutorial's test_vonmises_reparam, run against vbpm's implicit von Mises node."""

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import i0, i0e

from vbpm.util.vonmises import _VonMisesInvCDF, mean_resultant


def _q0(t, k):
    return np.exp(k * (np.cos(t) - 1.0)) / (2 * np.pi * i0e(k))


def _inverse_cdf(e, k):
    width = min(np.pi, 12.0 / np.sqrt(k))

    def cdf(p):
        return quad(lambda t: _q0(t, k), -np.pi, p, points=[0.0], limit=200)[0]

    return brentq(lambda p: cdf(p) - e, -width, width, xtol=1e-13)


@pytest.mark.parametrize("kappa", [0.5, 2.0, 5.0, 10.0, 100.0, 383.0])
def test_kappa_gradient_matches_finite_differences(kappa):
    torch.manual_seed(0)
    eps = torch.rand(6, dtype=torch.float64)
    k = torch.full((6,), kappa, dtype=torch.float64, requires_grad=True)
    _VonMisesInvCDF.apply(k, eps).sum().backward()
    h = 1e-5 * max(1.0, kappa)
    fd = np.array(
        [
            (_inverse_cdf(float(e), kappa + h) - _inverse_cdf(float(e), kappa - h)) / (2 * h)
            for e in eps
        ]
    )
    scale = max(1.0, float(np.abs(fd).max()))
    assert np.abs(k.grad.numpy() - fd).max() < 1e-4 * scale


def test_mean_derivative_is_exactly_one():
    torch.manual_seed(0)
    eps = torch.rand(6, dtype=torch.float64)
    mu = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    (mu + _VonMisesInvCDF.apply(torch.full((6,), 2.0, dtype=torch.float64), eps)).sum().backward()
    assert (mu.grad - 1.0).abs().max() < 1e-6


@pytest.mark.parametrize("kappa", [0.5, 2.0, 5.0, 10.0])
def test_mean_resultant_identity(kappa):
    a = float(mean_resultant(torch.tensor(kappa, dtype=torch.float64)))
    val = quad(
        lambda t: (np.cos(t) - a) * np.exp(kappa * np.cos(t)) / (2 * np.pi * i0(kappa)),
        -np.pi,
        np.pi,
    )[0]
    assert abs(val) < 1e-9


def test_gaussian_location_scale_gradients_match_equation_eight():
    from vbpm.nets import LatentParameters, LatentSampler

    torch.manual_seed(33)
    noise = torch.randn(4, dtype=torch.float64)
    torch.manual_seed(33)
    mean = torch.tensor([-2., 0., 1., 3.], dtype=torch.float64, requires_grad=True)
    scale = torch.tensor([0.2, 0.5, 1., 2.], dtype=torch.float64, requires_grad=True)
    parameters = LatentParameters(torch.zeros(4), torch.ones(4), mean, scale.log())
    velocity = LatentSampler()(parameters).velocity
    torch.testing.assert_close(velocity, mean + scale * noise)
    velocity.sum().backward()
    torch.testing.assert_close(mean.grad, torch.ones_like(mean))
    torch.testing.assert_close(scale.grad, noise)


def test_actual_sampler_phase_mean_gradient_is_one_across_wrap():
    from vbpm.nets import LatentParameters, LatentSampler

    torch.manual_seed(34)
    mean = torch.tensor([-3.14, 3.14], dtype=torch.float64, requires_grad=True)
    kappa = torch.tensor([2., 5.], dtype=torch.float64, requires_grad=True)
    parameters = LatentParameters(mean, kappa, torch.zeros(2), torch.zeros(2))
    phase = LatentSampler()(parameters).phase
    phase.sum().backward()
    torch.testing.assert_close(mean.grad, torch.ones_like(mean))
    assert torch.isfinite(kappa.grad).all()
    assert kappa.grad.abs().sum() > 0


@pytest.mark.parametrize('use_posterior', [False, True])
def test_final_frame_reconstruction_reaches_initial_draw_and_heads(use_posterior):
    from types import SimpleNamespace
    from vbpm.model import VBPM

    torch.manual_seed(35)
    model = VBPM(SimpleNamespace(num_channels=5, output_fps=50), d_model=8, samples=2)
    h = torch.randn(2, 6, 5)
    labels = torch.tensor([[0, 1, 0, 2, 0, 1], [1, 0, 2, 0, 1, 0]])
    draws = []

    def capture(module, args, state):
        if state.phase.requires_grad:
            state.phase.retain_grad()
        state.velocity.retain_grad()
        draws.append(state)

    hook = model.latent_sampler.register_forward_hook(capture)
    try:
        if use_posterior:
            logits, _, _ = model.posterior_rollout(h, labels, samples=2)
            heads = (model.posterior_model.phase_head, model.posterior_model.velocity_head)
        else:
            logits, _ = model.rollout(h, samples=2)
            heads = (model.prior_model.concentration_head, model.prior_model.velocity_head)
    finally:
        hook.remove()
    targets = labels[:, -1].expand(2, -1).reshape(-1)
    # Likelihood only: KL gradients cannot hide a detached sampler.
    loss = torch.nn.functional.cross_entropy(logits[:, :, -1].reshape(-1, 3), targets)
    loss.backward()
    assert len(draws) == h.shape[1] + 1
    # The initial prior phase is a parameter-free uniform draw.
    phase_draw = draws[0].phase if use_posterior else draws[1].phase
    for value in (phase_draw, draws[0].velocity):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
    for head in heads:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        assert all(p.grad.abs().sum() > 0 for p in head.parameters())
