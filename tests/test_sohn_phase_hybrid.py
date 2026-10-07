"""Check recognition initialization and the exact conditional phase KL."""

import sys
from pathlib import Path
import torch
from torch.distributions import VonMises

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics"))
from diagnostics.audio_generator import AudioPhaseFirstGSNN, _VonMisesInvCDF
from diagnostics.phase_hybrid import PhaseRecognition, noise
from vbpm.util.vonmises import kl_vonmises


def fixture():
    torch.manual_seed(23)
    p = AudioPhaseFirstGSNN(
        aligned_context=True,
        smooth_concentration=True,
        bernoulli_clock_emission=True,
        proposal_min_probability=0.5,
    )
    # The experiment starts from a learned phase head, rather than a zero-output cold head.
    with torch.no_grad():
        p.phase_head[-1].weight[0].normal_(std=0.01)
    h = torch.randn(2, 32, 512)
    phase = torch.arange(32)[None].expand(2, -1) * 0.04
    p.cached_prediction = dict(phase=phase, physical_velocity=torch.full_like(phase, 0.04))
    q = PhaseRecognition(p)
    labels = torch.zeros(2, 32, dtype=torch.long)
    labels[:, 3] = 2
    labels[:, 13] = 1
    return p, q, h, labels


def test_recognition_matches_prior_and_hybrid_matches_gsnn_at_initialization():
    p, q, h, y = fixture()
    pp = p.parameters_for(h)
    qp = q.parameters_for(p, h, y)
    assert all(torch.equal(a, b) for a, b in zip(pp, qp))
    n = noise(torch.Generator().manual_seed(4), 2, 32)
    rp = p.loss(h, y, n)
    rq, kl = q.loss(p, h, y, n)
    assert torch.equal(rp, rq) and kl.item() == 0
    assert torch.allclose(0.7 * (rq + kl) + 0.3 * rp, rp)
    (0.7 * (rq + kl) + 0.3 * rp).backward()
    assert all(
        v.grad is not None and torch.isfinite(v.grad).all()
        for v in list(p.parameters()) + list(q.parameters())
    )
    assert q.head[1].weight.grad[:, 1536:].abs().sum() > 0


def test_velocity_centering_cancels_in_joint_phase_kl():
    p, q, h, y = fixture()
    with torch.no_grad():
        q.head[-1].bias[0] += 0.4 / torch.pi
    pp = p.parameters_for(h)
    qp = q.parameters_for(p, h, y)
    count = 20000
    g = torch.Generator().manual_seed(333)
    u = torch.rand(count, 2, generator=g).clamp(1e-6, 1 - 1e-6)
    eps = torch.randn(count, 2, generator=g) * p.log_initial_sigma.exp() * ((h.shape[1] - 1) / 2)
    phase = qp[0] + _VonMisesInvCDF.apply(qp[2].expand(count, -1), u) - eps
    logq = VonMises(qp[0] - eps, qp[2]).log_prob(phase)
    logp = VonMises(pp[0] - eps, pp[2]).log_prob(phase)
    analytic = kl_vonmises(qp[0], qp[2], pp[0], pp[2])
    assert torch.allclose((logq - logp).mean(0), analytic, atol=0.06, rtol=0.06)


def test_mixture_importance_proposal_is_normalized_and_prior_identity():
    from diagnostics.likelihood import vm_logprob, importance_weights

    g = torch.Generator().manual_seed(62)
    angles = torch.rand(50000, generator=g) * 2 * torch.pi - torch.pi
    logp = vm_logprob(angles, torch.tensor(0.6), torch.tensor(3.0))
    logq = vm_logprob(angles, torch.tensor(-1.0), torch.tensor(1.2))
    assert abs(float(logp.exp().mean() * 2 * torch.pi) - 1) < 0.02
    assert abs(float(logq.exp().mean() * 2 * torch.pi) - 1) < 0.02
    logobs = torch.full_like(logp, -4.0)
    assert torch.allclose(importance_weights(logobs, logp, logp), logobs)
    # Integrating proposal_density * weight recovers a constant likelihood.
    proposal = torch.logaddexp(
        logp - torch.log(torch.tensor(2.0)), logq - torch.log(torch.tensor(2.0))
    )
    integral = (proposal + importance_weights(logobs, logp, logq)).exp().mean() * 2 * torch.pi
    assert abs(float(integral / torch.exp(torch.tensor(-4.0))) - 1) < 0.02


def test_observation_residual_preserves_initial_prediction_and_bypasses_saturation():
    p, original, h, y = fixture()
    with torch.no_grad():
        original.head[1].weight.zero_()
        original.head[1].bias.fill_(20.0)
    q = PhaseRecognition(p, observation_residual=True)
    missing = q.load_state_dict(original.state_dict(), strict=False)
    assert all(name.startswith("observation_head.") for name in missing.missing_keys)
    assert not missing.unexpected_keys
    before = original.parameters_for(p, h, y)
    after = q.parameters_for(p, h, y)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    after[0].sum().backward()
    assert q.head[1].weight.grad[:, 1536:].abs().sum() == 0
    assert q.observation_head[-1].weight.grad[0].abs().sum() > 0
