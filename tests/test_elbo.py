"""Independent checks of tutorial Section 3, with meter fixed at four."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import vonmises
from torch.distributions import Normal, kl_divergence

from vbpm.model import VBPM
from vbpm.util.kl import gaussian_kl
from vbpm.util.vonmises import kl_vonmises


def circular_kl_oracle(mu_q, kappa_q, mu_p, kappa_p):
    """Integrate q(log q - log p) around the circle independently of our Bessel code."""
    def integrand(angle):
        log_q = vonmises.logpdf(angle, kappa_q, loc=mu_q)
        log_p = vonmises.logpdf(angle, kappa_p, loc=mu_p)
        return np.exp(log_q) * (log_q - log_p)
    return quad(integrand, -np.pi, np.pi, epsabs=1e-10)[0]


@pytest.mark.parametrize('parameters', [
    (0., 0., 1., 0.), (1., 2., 0., 0.), (3.1, 4., -3.1, 2.),
    (-1., 50., 1., 30.), (0.2, 7., 0.2, 7.), (0., 0., 1., 3.),
])
def test_phase_kl_matches_numerical_density_integral(parameters):
    values = [torch.tensor(x, dtype=torch.float64) for x in parameters]
    actual = kl_vonmises(*values).item()
    expected = circular_kl_oracle(*parameters)
    assert actual == pytest.approx(expected, abs=1e-9)
    assert actual >= -1e-12


def test_velocity_kl_matches_independent_distribution_implementation():
    mu_q = torch.tensor([-2., 0., 4.], dtype=torch.float64)
    std_q = torch.tensor([0.3, 1., 2.], dtype=torch.float64)
    mu_p = torch.tensor([1., 0., -1.], dtype=torch.float64)
    std_p = torch.tensor([2., 1., 0.5], dtype=torch.float64)
    actual = gaussian_kl(mu_q, std_q, mu_p, std_p)
    expected = kl_divergence(Normal(mu_q, std_q), Normal(mu_p, std_p))
    torch.testing.assert_close(actual, expected)
    assert actual[1] == 0
    assert (actual >= 0).all()


def test_elbo_matches_independent_likelihood_and_initial_plus_transition_kls():
    torch.manual_seed(21)
    model = VBPM(SimpleNamespace(num_channels=5, output_fps=50), d_model=8, samples=2)
    h = torch.randn(2, 3, 5)
    labels = torch.tensor([[0, 1, 2], [2, 0, 1]])
    mask = torch.tensor([[1., 1., 0.], [0., 0., 0.]])
    prior, posterior, emitted = [], [], []
    hooks = [
        model.prior_model.register_forward_hook(lambda module, args, output: prior.append(output)),
        model.posterior_model.register_forward_hook(
            lambda module, args, output: posterior.append(output)
        ),
        model.emission_model.register_forward_hook(
            lambda module, args, output: emitted.append(output)
        ),
    ]
    try:
        result = model(h, mask, labels, gsnn_only=False)
    finally:
        for hook in hooks:
            hook.remove()
    assert len(posterior) == h.shape[1] + 1
    assert len(prior) == 2 * len(posterior)
    assert len(emitted) == 2 * h.shape[1]
    # The second branch uses posterior draws. There is no emission for z_0.
    logits = torch.stack(emitted[h.shape[1]:], dim=1).reshape(2, 2, 3, 3)
    log_prob = logits.log_softmax(-1).gather(
        -1, labels[None, ..., None].expand(2, -1, -1, -1)
    ).squeeze(-1)
    expected_recon = (log_prob * mask).sum(-1).mean(0)
    torch.testing.assert_close(result['recon'], expected_recon)
    expected_terms = {'phase': torch.zeros(2), 'velocity': torch.zeros(2)}
    for step, (q, p) in enumerate(zip(posterior, prior[len(posterior):])):
        phase_kl = torch.tensor([
            circular_kl_oracle(*(float(v[i].detach()) for v in (
                q.phase_mean, q.phase_concentration, p.phase_mean, p.phase_concentration
            ))) for i in range(4)
        ]).reshape(2, 2)
        vel_kl = kl_divergence(
            Normal(q.velocity_mean, q.velocity_log_std.exp()),
            Normal(p.velocity_mean, p.velocity_log_std.exp()),
        ).reshape(2, 2)
        weight = mask.any(-1) if step == 0 else mask[:, step - 1]
        expected_terms['phase'] += (phase_kl * weight).mean(0)
        expected_terms['velocity'] += (vel_kl * weight).mean(0)
    for name, value in expected_terms.items():
        torch.testing.assert_close(result['kl_terms'][name], value, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(result['elbo'], expected_recon - sum(expected_terms.values()))
    assert result['elbo'][1] == 0
