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
