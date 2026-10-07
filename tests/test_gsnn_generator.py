"""Regressions for the deployed generator and the pure-GSNN objective."""

import math
from unittest.mock import patch

import torch

from vbpm.config import load_config
from vbpm.model import VBPM, build_model
from vbpm.specs import EmissionSpec, PriorSpec


def model():
    torch.manual_seed(6)
    return VBPM(
        8,
        d_model=32,
        prior_spec=PriorSpec(dim=32, layers=1),
        emission=EmissionSpec(dim=32, layers=1),
        velocity_ref=math.pi / 50,
        velocity_step=0.01,
    )


def test_mode_is_deterministic_and_consumes_no_rng():
    m = model().eval()
    h = torch.randn(3, 20, 8)
    rng = torch.random.get_rng_state().clone()
    a, b = m.infer_path(h), m.infer_path(h)
    assert torch.equal(a["phi_path"], b["phi_path"])
    assert torch.equal(a["velocity_path"], b["velocity_path"])
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_centered_anchor_matches_mean_rollout_with_changing_tempo():
    m = model().eval()
    h, mask = torch.randn(3, 20, 8), torch.ones(3, 20)
    mask[1, 13:] = 0
    with torch.no_grad():
        m.prior_model.velocity_head.weight[0].normal_(std=0.03)
        p = m.prior_model(h, mask, m.velocity_ref)
        path = m.infer_path(h, mask)
    weights = mask / mask.sum(1, keepdim=True)
    assert torch.allclose((path["phi_path"] * weights).sum(1), p["phase_anchor"], atol=2e-6)
    assert torch.allclose(
        path["velocity_path"][mask.bool()], p["velocity_mean"][mask.bool()], atol=2e-6
    )
    assert path["phi_path"][1, 13:].eq(path["phi_path"][1, 12]).all()
    assert m.phase_step(path["velocity_path"]).gt(0).all()


def test_random_walk_retains_predicted_drift_and_its_gradient():
    m = model()
    h, mask = torch.randn(2, 12, 8), torch.ones(2, 12)
    with torch.no_grad():
        m.prior_model.velocity_head.weight[0].normal_(std=0.03)
    p = m.prior_model(h, mask, m.velocity_ref)
    previous = torch.tensor([0.4, -0.3])
    factor = m.prior_model.step(
        p, 7, torch.zeros(2), previous, torch.tensor([[0.0, 1.0], [0.0, 1.0]]), m.velocity_step
    )
    mu, sigma = factor["velocity"]
    expected = previous + p["velocity_mean"][:, 7] - p["velocity_mean"][:, 6]
    assert torch.allclose(mu, expected)
    assert not torch.allclose(mu, previous)
    assert torch.allclose(sigma, m.velocity_step * p["velocity_sigma"][:, 7])
    mu.sum().backward()
    assert m.prior_model.velocity_head.weight.grad[0].norm() > 0


def test_initial_tempo_noise_preserves_window_phase_anchor():
    m = model().eval()
    h, mask = torch.randn(2, 60, 8), torch.ones(2, 60)
    p = m.prior_model(h, mask, m.velocity_ref)
    draw = m.prior_model.mode(h, mask, parameters=p)
    draw["velocity"][:, 0] += torch.tensor([0.3, -0.2])
    draw["phase0"] = m.prior_model.phase0_given_velocity(
        p, draw["velocity"][:, 0], mask, m.velocity_ref, m.velocity_step
    )[0]
    path = m.draws_to_paths(draw, mask, prior=p, h=h, mode=True)
    assert torch.allclose(path["phi_path"].mean(1), p["phase_anchor"], atol=2e-6)


def test_pure_gsnn_skips_q_and_updates_phase_tempo_and_decoder():
    m = model()
    h, mask = torch.randn(3, 16, 8), torch.ones(3, 16)
    labels = torch.randint(3, (3, 16))
    groups = [
        m.prior_model.phase0_head,
        m.prior_model.velocity0_head,
        m.prior_model.velocity_head,
        m.emission_model,
    ]
    before = [[p.detach().clone() for p in group.parameters()] for group in groups]
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    with (
        patch.object(m.posterior_model, "forward", side_effect=AssertionError("q was called")),
        patch.object(m, "kl_terms", side_effect=AssertionError("KL was called")),
    ):
        out = m(h, mask, labels, gsnn_only=True)
    loss = -out["recon_prior"].mean() / mask.shape[1]
    loss.backward()
    assert torch.isfinite(loss)
    for group in groups:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in group.parameters())
        assert sum(float(p.grad.square().sum()) for p in group.parameters()) > 0
    assert all(p.grad is None for p in m.posterior_model.parameters())
    opt.step()
    for group, old in zip(groups, before):
        assert any(not torch.equal(p, q) for p, q in zip(group.parameters(), old))


def test_mixed_objective_and_sampling_still_work():
    m = model()
    h, mask = torch.randn(2, 10, 8), torch.ones(2, 10)
    labels = torch.randint(3, (2, 10))
    out = m(h, mask, labels)
    loss = -(0.7 * out["elbo"] + 0.3 * out["recon_prior"]).mean() / mask.shape[1]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(v).all() for v in out["kl_terms"].values())
    for module in (m.posterior_model, m.prior_model, m.emission_model):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.square().sum()) for g in grads) > 0
    m.eval()
    probs = m.label_probs(h, mask, rollouts=3)
    assert probs.shape == (2, 10, 3)
    assert torch.allclose(probs.sum(-1), torch.ones(2, 10), atol=1e-6)


def test_padding_does_not_change_prior_parameters_and_empty_item_is_finite():
    m = model().eval()
    h, mask = torch.randn(2, 12, 8), torch.ones(2, 12)
    mask[:, 8:] = 0
    other = h.clone()
    other[:, 8:] = torch.randn_like(other[:, 8:]) * 100
    a, b = m.prior_model(h, mask, m.velocity_ref), m.prior_model(other, mask, m.velocity_ref)
    assert torch.allclose(a["phase0"][0], b["phase0"][0], atol=1e-6)
    assert torch.allclose(a["velocity_mean"][:, :8], b["velocity_mean"][:, :8], atol=1e-6)
    mask[0] = 0
    out = m(h, mask, torch.zeros(2, 12, dtype=torch.long), gsnn_only=True)
    assert torch.isfinite(out["recon_prior"]).all()
    assert out["recon_prior"][0] == 0


def test_gsnn_recipe_builds_positive_velocity_and_trainable_decoder():
    cfg, _ = load_config("vbpm/configs/gsnn.yaml")
    m = build_model(cfg, 8)
    assert cfg.gsnn_alpha == 0
    assert m.velocity_ref > 0 and m.velocity_step > 0
    assert m.prior_model.meter_gated
    assert all(p.requires_grad for p in m.emission_model.parameters())


def test_baseline_restores_alpha_point_seven_with_the_corrected_generator():
    cfg, _ = load_config("vbpm/configs/baseline.yaml")
    m = build_model(cfg, 8)
    assert cfg.gsnn_alpha == 0.7
    assert cfg.beta_warmup == 0 and cfg.beta_end == 1
    assert m.velocity_ref > 0 and m.velocity_step > 0
    assert m.prior_model.meter_gated


def test_posterior_starts_at_prior_continuous_factors_in_their_actual_units():
    m = model()
    h, mask = torch.randn(3, 16, 8), torch.ones(3, 16)
    labels = torch.randint(3, (3, 16))
    out = m(h, mask, labels)
    for key in ("phase0", "phase", "velocity0", "velocity"):
        assert out["kl_terms"][key].abs().max() < 1e-5
    # A half-standard-deviation correction has KL = 0.5 * 0.5**2,
    # regardless of the small physical random-walk scale.
    with torch.no_grad():
        m.posterior_model.velocity_head.bias[0] = 0.5
    out = m(h, mask, labels)
    expected = torch.full((3,), 0.125 * (mask.shape[1] - 1))
    assert torch.allclose(out["kl_terms"]["velocity"], expected, atol=1e-4)


def test_gated_meter_is_audio_conditioned_and_changes_only_at_bar_crossings():
    m = model().eval()
    m.prior_model.meter_gated = True
    h, mask = torch.randn(2, 150, 8), torch.ones(2, 150)
    p = m.prior_model(h, mask, m.velocity_ref)
    assert not torch.equal(p["log_meter0"][0], p["log_meter0"][1])
    path = m.infer_path(h, mask)
    changed = (path["meter_path"][:, 1:] != path["meter_path"][:, :-1]).any(-1)
    assert not (changed & ~path["is_downbeat"][:, 1:]).any()
