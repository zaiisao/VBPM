"""Checks for the fresh GSNN transition-prior implementation."""

from types import SimpleNamespace

import torch

from vbpm.model import VBPM
from vbpm.nets import LatentState


def small_case():
    torch.manual_seed(12)
    frontend = SimpleNamespace(num_channels=8, output_fps=50)
    model = VBPM(frontend, d_model=16, samples=2)
    h = torch.randn(2, 12, 8)
    labels = torch.randint(0, 3, (2, 12))
    return model, h, labels, torch.ones(2, 12)


def test_gsnn_trains_each_transition_head_and_emission():
    model, h, labels, mask = small_case()
    result = model(h, mask, labels)
    (-result["prior_recon"].mean()).backward()
    assert torch.isfinite(result["prior_recon"]).all()
    for module in (
        model.prior_model.context,
        model.prior_model.velocity_head,
        model.prior_model.concentration_head,
        model.emission_model,
    ):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert any(p.grad.abs().sum() > 0 for p in module.parameters())
    assert torch.equal(result["kl"], torch.zeros(2))


def test_transition_uses_previous_sample_and_frame_period():
    model, h, _, _ = small_case()
    phase = torch.tensor([0.2, 0.4], requires_grad=True)
    velocity = torch.tensor([2.0, 3.0], requires_grad=True)
    p = model.prior_model(h[:, 0], LatentState(phase, velocity))
    torch.testing.assert_close(p.phase_mean, phase + velocity / 50)
    p.velocity_mean.sum().backward()
    assert phase.grad.abs().sum() > 0
    assert velocity.grad.abs().sum() > 0


def test_initial_phase_is_uniform_and_sampling_is_differentiable():
    model, h, _, _ = small_case()
    audio = h[:1, 0].expand(2048, -1)
    zero = torch.zeros(2048)
    p = model.prior_model(audio, LatentState(zero, zero), initial=True)
    state = model.latent_sampler(p, initial=True)
    assert state.phase.min() >= -torch.pi and state.phase.max() <= torch.pi
    assert state.phase.cos().mean().abs() < 0.06
    assert state.phase.sin().mean().abs() < 0.06
    state.velocity.mean().backward()
    assert model.prior_model.velocity_head.weight.grad.abs().sum() > 0


def test_inference_padding_and_deterministic_rollout():
    model, h, _, mask = small_case()
    mask[1, 8:] = 0
    rng = torch.random.get_rng_state().clone()
    path = model.infer_path(h, mask)
    assert torch.equal(rng, torch.random.get_rng_state())
    torch.testing.assert_close(path["phi_path"], model.infer_path(h, mask)["phi_path"])
    probabilities = model.predict_label_probs(h, mask, rollouts=2)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(mask))
    assert (probabilities[1, 8:, 0] == 1).all()
    assert (path["phi_path"][1, 8:] == 0).all()


def test_sampling_reaches_previous_states_through_the_full_rollout():
    model, h, _, _ = small_case()
    h.requires_grad_()
    logits, state = model.rollout(h, samples=2)
    assert state.phase.shape == (2, 2, 12)
    logits[:, :, -1].square().sum().backward()
    assert torch.isfinite(h.grad).all()
    assert h.grad[:, 0].abs().sum() > 0


def test_hybrid_trains_posterior_prior_and_emission():
    model, h, labels, mask = small_case()
    result = model(h, mask, labels, gsnn_only=False)
    torch.testing.assert_close(result["elbo"], result["recon"] - result["kl"])
    torch.testing.assert_close(result["kl"], sum(result["kl_terms"].values()))
    assert set(result["kl_terms"]) == {"phase", "velocity"}
    assert (result["kl"] >= 0).all()
    loss = -(0.7 * result["elbo"] + 0.3 * result["prior_recon"]).mean()
    loss.backward()
    for module in (model.posterior_model, model.prior_model, model.emission_model):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert all(p.grad.abs().sum() > 0 for p in module.parameters())


def test_posterior_uses_labels_future_context_and_previous_state():
    model, h, labels, _ = small_case()
    context = model.posterior_model.encode(h, labels)
    altered = labels.clone()
    altered[:, -1] = (altered[:, -1] + 1) % 3
    changed_context = model.posterior_model.encode(h, altered)
    assert not torch.allclose(context[:, 0], changed_context[:, 0], atol=1e-8, rtol=0)
    previous = LatentState(torch.zeros(2, requires_grad=True), torch.ones(2, requires_grad=True))
    parameters = model.posterior_model(context[:, 0], previous)
    parameters.velocity_mean.sum().backward()
    assert previous.phase.grad.abs().sum() > 0
    assert previous.velocity.grad.abs().sum() > 0


def test_gsnn_does_not_use_posterior():
    model, h, labels, mask = small_case()
    (-model(h, mask, labels)["prior_recon"].mean()).backward()
    assert all(p.grad is None for p in model.posterior_model.parameters())


def test_emission_depends_only_on_latent_state():
    model, _, _, _ = small_case()
    phase = torch.tensor([0.2, 0.8], requires_grad=True)
    velocity = torch.tensor([2.0, 3.0], requires_grad=True)
    logits = model.emission_model(LatentState(phase, velocity))
    assert logits.shape == (2, 3)
    logits.square().sum().backward()
    assert phase.grad.abs().sum() > 0
    assert velocity.grad.abs().sum() > 0


def test_velocity_updates_every_frame_without_a_beat_gate():
    from vbpm.nets import LatentSampler, LatentParameters

    means = torch.tensor([10., 11.], requires_grad=True)
    parameters = LatentParameters(torch.tensor([0.2, 0.3]), torch.ones(2), means, torch.zeros(2))
    state = LatentSampler()(parameters, sample=False)
    torch.testing.assert_close(state.velocity, means)
    state.velocity.sum().backward()
    torch.testing.assert_close(means.grad, torch.ones(2))


def test_phase_parameters_match_section_two_across_the_wrap():
    model, h, labels, _ = small_case()
    previous = LatentState(torch.tensor([6.27, -0.01]), torch.tensor([3., -2.]))
    prior = model.prior_model(h[:, 0], previous)
    expected = torch.remainder(previous.phase + previous.velocity / 50, 2 * torch.pi)
    torch.testing.assert_close(torch.remainder(prior.phase_mean, 2 * torch.pi), expected)
    assert (prior.phase_concentration > 0).all()
    initial = model.prior_model(h[:, 0], previous, initial=True)
    torch.testing.assert_close(initial.phase_concentration, torch.zeros(2))
    with torch.no_grad():
        model.posterior_model.phase_head.weight.zero_()
        model.posterior_model.phase_head.bias.copy_(torch.tensor([3., 4., 0.]))
    context = model.posterior_model.encode(h, labels)
    posterior = model.posterior_model(context[:, 0], previous)
    expected_mean = torch.atan2(torch.tensor(4.), torch.tensor(3.)).expand(2)
    torch.testing.assert_close(posterior.phase_mean, expected_mean)
    assert (posterior.phase_concentration > 0).all()
    assert (prior.velocity_log_std.exp() > 0).all()
    assert (posterior.velocity_log_std.exp() > 0).all()


def test_gaussian_samples_match_predicted_mean_and_variance():
    from vbpm.nets import LatentParameters, LatentSampler

    torch.manual_seed(41)
    count = 12000
    means = torch.tensor([-2., 3.]).expand(count, 2)
    stds = torch.tensor([0.7, 1.4]).expand(count, 2)
    parameters = LatentParameters(
        torch.zeros_like(means), torch.zeros_like(means), means, stds.log()
    )
    draws = LatentSampler()(parameters, initial=True).velocity
    torch.testing.assert_close(draws.mean(0), means[0], atol=0.05, rtol=0)
    torch.testing.assert_close(draws.var(0), stds[0].square(), atol=0.07, rtol=0)


def test_vonmises_samples_match_circular_mean_and_concentration():
    from scipy.special import ive
    from vbpm.nets import LatentParameters, LatentSampler

    torch.manual_seed(42)
    count = 6000
    means = torch.tensor([2.8, -2.8]).expand(count, 2)
    kappas = torch.tensor([1., 20.]).expand(count, 2)
    parameters = LatentParameters(means, kappas, torch.zeros_like(means), torch.zeros_like(means))
    phase = LatentSampler()(parameters).phase
    deviations = phase - means
    torch.testing.assert_close(deviations.sin().mean(0), torch.zeros(2), atol=0.035, rtol=0)
    expected = torch.tensor(ive(1, kappas[0].numpy()) / ive(0, kappas[0].numpy()))
    torch.testing.assert_close(deviations.cos().mean(0), expected, atol=0.035, rtol=0)
    assert deviations.cos().mean(0)[1] > deviations.cos().mean(0)[0]


def test_posterior_parameters_respond_to_future_audio():
    model, h, labels, _ = small_case()
    changed = h.clone()
    changed[:, -1] += 10
    previous = LatentState(torch.zeros(2), torch.ones(2))
    original = model.posterior_model(model.posterior_model.encode(h, labels)[:, 0], previous)
    altered = model.posterior_model(model.posterior_model.encode(changed, labels)[:, 0], previous)
    assert not torch.allclose(original.velocity_mean, altered.velocity_mean, atol=1e-8, rtol=0)


def test_rollouts_condition_on_the_actual_previous_draw():
    model, h, labels, _ = small_case()
    for use_posterior in (False, True):
        prior_inputs, posterior_inputs, draws = [], [], []
        prior_hook = model.prior_model.register_forward_pre_hook(
            lambda module, args: prior_inputs.append(args[1])
        )
        posterior_hook = model.posterior_model.register_forward_pre_hook(
            lambda module, args: posterior_inputs.append(args[1])
        )
        sampler_hook = model.latent_sampler.register_forward_hook(
            lambda module, args, output: draws.append(output)
        )
        try:
            if use_posterior:
                _, states, _ = model.posterior_rollout(h, labels, samples=2)
            else:
                _, states = model.rollout(h, samples=2)
        finally:
            prior_hook.remove()
            posterior_hook.remove()
            sampler_hook.remove()
        assert len(prior_inputs) == h.shape[1] + 1
        torch.testing.assert_close(prior_inputs[0].phase, torch.zeros(4))
        for frame in range(1, h.shape[1] + 1):
            torch.testing.assert_close(
                prior_inputs[frame].phase, draws[frame - 1].phase
            )
            torch.testing.assert_close(
                prior_inputs[frame].velocity, draws[frame - 1].velocity
            )
        if use_posterior:
            assert len(posterior_inputs) == len(prior_inputs)
            assert all(q is p for q, p in zip(posterior_inputs, prior_inputs))
        else:
            assert not posterior_inputs
