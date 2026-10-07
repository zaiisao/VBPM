"""Exercise tutorial Section 5 through the actual train.py optimizer step."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import train as training
from vbpm.model import VBPM


@pytest.mark.parametrize('alpha,beta', [(0., 1.), (0.7, 1.), (1., 1.), (1., 0.)])
def test_training_weights_and_gradient_routes(monkeypatch, alpha, beta):
    dataset = []
    for i in range(2):
        dataset.append(dict(
            input=np.arange(16, dtype=np.float32).reshape(4, 4) / 20 + i / 10,
            cls=np.array([0, 1, 0, 2], dtype=np.int64),
            mask=np.array([1., 1., 1., float(i)], dtype=np.float32),
            t0=0., beat_times=[], downbeat_times=[], dataset='test',
        ))
    frontend = SimpleNamespace(num_channels=4, output_fps=50, forward_features=lambda x: x)
    cfg = SimpleNamespace(
        batch_size=2, lr=0.001, epochs=1, gsnn_alpha=alpha,
        beta_warmup=0, beta_end=beta, clip=1e6,
    )
    recorded = {}

    def build_model(cfg, frontend):
        model = VBPM(frontend, d_model=8, samples=2)
        recorded['model'] = model
        original_forward = model.forward

        def forward(h, mask, cls, *, gsnn_only):
            output = original_forward(h, mask, cls, gsnn_only=gsnn_only)
            recorded.update(output=output, mask=mask, gsnn_only=gsnn_only)
            for name in ('prior_recon', 'recon', 'kl'):
                if output[name].requires_grad:
                    output[name].retain_grad()
            return output

        monkeypatch.setattr(model, 'forward', forward)
        return model

    training.train(
        dataset, frontend, torch.device('cpu'), cfg,
        SimpleNamespace(build_model=build_model), seed=40, workers=0,
    )
    output, mask, model = recorded['output'], recorded['mask'], recorded['model']
    denominator = mask.sum(-1) * mask.shape[0]
    torch.testing.assert_close(output['prior_recon'].grad, -(1 - alpha) / denominator)
    assert recorded['gsnn_only'] == (alpha == 0)
    if alpha == 0:
        assert all(p.grad is None for p in model.posterior_model.parameters())
    else:
        torch.testing.assert_close(output['recon'].grad, -alpha / denominator)
        torch.testing.assert_close(output['kl'].grad, alpha * beta / denominator)
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.posterior_model.parameters()
        )
        assert model.posterior_model.phase_head.weight.grad.abs().sum() > 0
        assert model.posterior_model.velocity_head.weight.grad.abs().sum() > 0
    if alpha == 1 and beta == 0:
        # A pure posterior reconstruction has no path into the independent prior.
        assert all(p.grad is not None and p.grad.abs().sum() == 0
                   for p in model.prior_model.parameters())
    else:
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.prior_model.parameters())
        assert model.prior_model.velocity_head.weight.grad.abs().sum() > 0
        assert model.prior_model.concentration_head.weight.grad.abs().sum() > 0
    assert model.emission_model.network[0].weight.grad.abs().sum() > 0


def test_hybrid_rolls_separate_chains_with_shared_prior_and_emission():
    torch.manual_seed(41)
    model = VBPM(SimpleNamespace(num_channels=4, output_fps=50), d_model=8, samples=2)
    h = torch.randn(2, 5, 4)
    labels = torch.tensor([[0, 1, 0, 2, 0], [2, 0, 1, 0, 1]])
    previous_states, emitted_states = [], []
    hooks = [
        model.prior_model.register_forward_pre_hook(
            lambda module, args: previous_states.append(args[1])
        ),
        model.emission_model.register_forward_pre_hook(
            lambda module, args: emitted_states.append(args[0])
        ),
    ]
    try:
        output = model(h, torch.ones(2, 5), labels, gsnn_only=False)
    finally:
        for hook in hooks:
            hook.remove()
    assert len(previous_states) == 2 * (h.shape[1] + 1)
    assert len(emitted_states) == 2 * h.shape[1]
    # The same modules run twice, conditioning on their branch's own draws.
    assert not torch.equal(previous_states[1].phase, previous_states[7].phase)
    assert not torch.equal(output['prior_path']['phi_path'], output['path']['phi_path'])
    for branch in range(2):
        offset = branch * (h.shape[1] + 1)
        for step in range(2, h.shape[1] + 1):
            previous = previous_states[offset + step]
            actual = emitted_states[branch * h.shape[1] + step - 2]
            assert previous is actual
