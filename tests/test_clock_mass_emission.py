import math

import pytest
import torch
from torch.nn import functional as F

from diagnostics.emissions import ClockMassDecoder


def test_extra_clock_cycles_cannot_disappear_from_event_mass():
    decoder = ClockMassDecoder()
    phase = (torch.arange(200, dtype=torch.float64) * (2 * math.pi / 200))[None]
    velocity = torch.full((1, 199), 2 * math.pi / 200, dtype=torch.float64)
    labels = torch.zeros_like(phase, dtype=torch.long)
    labels[:, [50, 100, 150]] = 1
    labels[:, 0] = 2
    losses = []
    for factor in (1, 3):
        logp = decoder(phase * factor, velocity * factor)
        # -log P(N) is the integrated event intensity. Timing uncertainty
        # cannot reduce its total from twelve landmarks to four.
        assert float(-logp[..., 0].sum()) == pytest.approx(4 * factor, abs=1e-6)
        losses.append(F.cross_entropy(logp.reshape(-1, 3), labels.reshape(-1)))
    assert losses[1] > losses[0]


@pytest.mark.parametrize("raw_width", [-12.0, 0.0, 12.0])
def test_tail_probabilities_and_gradients_remain_finite(raw_width):
    decoder = ClockMassDecoder()
    with torch.no_grad():
        decoder.raw_width.fill_(raw_width)
    phase = torch.linspace(-4.0, 4.0, 100, dtype=torch.float64)[None].requires_grad_()
    velocity = torch.full((1, 99), 0.08, dtype=torch.float64, requires_grad=True)
    logp = decoder(phase, velocity)
    torch.testing.assert_close(logp.logsumexp(-1), torch.zeros_like(phase), atol=1e-9, rtol=0)
    loss = F.cross_entropy(logp.reshape(-1, 3), torch.arange(100) % 3)
    loss.backward()
    assert torch.isfinite(loss)
    for tensor in (phase, velocity, decoder.raw_width):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize("raw_width", [-12.0, 0.0, 12.0])
def test_bernoulli_clock_normalization_and_finite_tail_gradients(raw_width):
    from diagnostics.emissions import BernoulliClockDecoder

    decoder = BernoulliClockDecoder()
    with torch.no_grad():
        decoder.raw_width.fill_(raw_width)
    phase = (torch.arange(200, dtype=torch.float64) * (2 * math.pi / 200))[None].requires_grad_()
    velocity = torch.full((1, 199), 2 * math.pi / 200, dtype=torch.float64, requires_grad=True)
    logp = decoder(phase, velocity)
    torch.testing.assert_close(logp.exp().sum(-1), torch.ones_like(phase), rtol=1e-9, atol=1e-9)
    labels = torch.zeros_like(phase, dtype=torch.long)
    labels[:, [50, 100, 150]] = 1
    labels[:, 0] = 2
    loss = F.cross_entropy(logp.reshape(-1, 3), labels.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    for gradient in (phase.grad, velocity.grad, decoder.raw_width.grad):
        assert torch.isfinite(gradient).all()
    if raw_width == -12.0:
        assert float(logp[0, 0, 2].exp()) > 0.999
        expected = float(logp[..., 1:].exp().sum())
        assert expected == pytest.approx(4.0, abs=1e-5)
