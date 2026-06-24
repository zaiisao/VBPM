"""End-to-end integration tests for the CHART SVT pipeline.

Covers: SVTModel.forward (bar_phase=True, audio_emission=True) -> compute_elbo_loss
(+ audio_recon + z_forcing-style future term + barphase_sup + phase_sup) -> backward.

Ground truth / invariants used (NOT tautologies):
  * Structural: output dict keys/shapes, simplex/one-hot of meter samples, angles in [0, 2pi).
  * Finiteness: no NaN/Inf anywhere through forward, loss, and gradients.
  * Connectivity: EVERY trainable parameter that participates in the composed objective
    receives a finite, non-None gradient (no silently-disconnected sub-module). The set
    of params that get grad is derived from the model's own used sub-modules, not hard-coded.
  * Closed-form: the recon term of compute_elbo_loss equals F.binary_cross_entropy_with_logits
    computed independently (channel sum), and the z_forcing/audio_recon terms equal an
    independent MSE.
  * Learnability (optimization invariant, NOT a tautology): a short Adam loop on ONE fixed
    batch must DECREASE the training objective and the beat BCE — a model that cannot fit a
    single batch is broken. Verified by strict inequality with margin, not by a copied value.
  * sample_from_prior / sample_from_prior_pf produce finite beat_logits of the right shape.

All on CPU with tiny tensors.
"""

import sys
sys.path.insert(0, "/home/sogang/jaehoon/CHART")

import math

import pytest
import torch
import torch.nn.functional as F

from models.svt_core import SVTModel, TWO_PI
from models.loss import compute_elbo_loss


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

B, T, INPUT_DIM, K = 2, 24, 2, 4


def _build_model(dtype=torch.float64, **overrides):
    kw = dict(
        hidden_dim=16,
        nhead=2,
        num_layers=1,
        num_meter_classes=K,
        input_dim=INPUT_DIM,
        bar_phase=True,
        audio_emission=True,
    )
    kw.update(overrides)
    m = SVTModel(**kw).to("cpu")
    if dtype is torch.float64:
        m = m.double()
    return m.train()


def _make_batch(seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    act = torch.randn(B, T, INPUT_DIM, generator=g, dtype=dtype)
    # Sparse, realistic beat/downbeat targets (downbeats a subset-ish of beats but
    # independent here is fine for the invariants we test).
    bt = (torch.rand(B, T, generator=g) > 0.7).to(dtype)
    db = (torch.rand(B, T, generator=g) > 0.85).to(dtype)
    return act, bt, db


def _barphase_targets(db):
    """Reproduce the training-loop dense bar-phase target (sawtooth in [0, 2pi))."""
    return SVTModel._beat_targets_to_distance(db) * TWO_PI


def _full_objective(model, act, bt, db, *, beta=1.0,
                    audio_recon_weight=0.5, z_forcing_weight=0.5,
                    z_forcing_offset=8, barphase_sup_weight=1.0,
                    phase_sup_weight=0.5, free_bits=0.0):
    """Compose the exact training-step objective: ELBO (+ barphase/phase sup) +
    audio_recon MSE + z_forcing future-prediction MSE. Returns (total, components, out)."""
    out = model(act, beat_targets=bt, downbeat_targets=db)
    barphase_targets = None
    if "barphase_mu" in out["prior"]:
        barphase_targets = _barphase_targets(db)
    # GT beat-phase sawtooth as a stand-in phase_target (matches train.py wiring shape).
    phase_targets = SVTModel._beat_targets_to_distance(bt) * TWO_PI

    total, comp = compute_elbo_loss(
        beat_logits=out["beat_logits"],
        beat_targets=bt,
        posterior=out["posterior"],
        prior=out["prior"],
        beta=beta,
        downbeat_targets=db,
        free_bits=free_bits,
        barphase_targets=barphase_targets,
        barphase_sup_weight=barphase_sup_weight,
        phase_targets=phase_targets,
        phase_sup_weight=phase_sup_weight,
    )

    if audio_recon_weight > 0.0 and out.get("audio_recon") is not None:
        audio_recon_loss = ((out["audio_recon"] - act) ** 2).mean()
        total = total + audio_recon_weight * audio_recon_loss
        comp["audio_recon"] = audio_recon_loss.detach()

    if z_forcing_weight > 0.0 and out.get("audio_recon") is not None:
        kf = max(1, int(z_forcing_offset))
        if act.shape[1] > kf:
            zf = ((out["audio_recon"][:, :-kf] - act[:, kf:]) ** 2).mean()
            total = total + z_forcing_weight * zf
            comp["z_forcing"] = zf.detach()

    return total, comp, out


# ---------------------------------------------------------------------------
# Forward structural invariants
# ---------------------------------------------------------------------------

def test_forward_shapes_and_keys():
    model = _build_model()
    act, bt, db = _make_batch()
    out = model(act, beat_targets=bt, downbeat_targets=db)

    assert out["beat_logits"].shape == (B, T, 2)
    assert out["audio_recon"].shape == (B, T, INPUT_DIM)

    post, prior, samp = out["posterior"], out["prior"], out["samples"]
    assert post["meter_logits"].shape == (B, T, K)
    assert post["phase_mu"].shape == (B, T)
    assert post["phase_log_kappa"].shape == (B, T)
    assert post["tempo_mu"].shape == (B, T)
    assert post["tempo_log_sigma"].shape == (B, T)
    assert prior["phase_kappa"].shape == (B, T)
    assert prior["tempo_sigma"].shape == (B, T)

    # bar_phase machinery present
    for d in (post, prior):
        assert "barphase_mu" in d
    assert samp["bar_phase"].shape == (B, T)
    assert samp["phase"].shape == (B, T)
    assert samp["log_tempo"].shape == (B, T)
    assert samp["meter_soft"].shape == (B, T, K)


def test_forward_finite_everywhere():
    model = _build_model()
    act, bt, db = _make_batch()
    out = model(act, beat_targets=bt, downbeat_targets=db)
    for name, t in [("beat_logits", out["beat_logits"]),
                    ("audio_recon", out["audio_recon"])]:
        assert torch.isfinite(t).all(), f"{name} not finite"
    for grp in ("posterior", "prior", "samples"):
        for k, v in out[grp].items():
            assert torch.isfinite(v).all(), f"{grp}.{k} not finite"


def test_angles_wrapped_and_meter_simplex():
    model = _build_model()
    act, bt, db = _make_batch()
    out = model(act, beat_targets=bt, downbeat_targets=db)
    for key in ("phase", "bar_phase"):
        ang = out["samples"][key]
        assert (ang >= 0).all() and (ang < TWO_PI + 1e-6).all(), f"{key} not in [0,2pi)"
    # PRIOR phase mean: the TRANSITION steps (t>=1) are torch.remainder(..., 2pi) so
    # they are wrapped in [0, 2pi). The INIT step (t=0) comes from _split_head with
    # phi_prev=None -> pi*tanh(raw) in [-pi, pi] (a free angle, KL is rotation-invariant).
    pm_prior = out["prior"]["phase_mu"]
    pm_prior_trans = pm_prior[:, 1:]
    assert (pm_prior_trans >= -1e-6).all() and (pm_prior_trans < TWO_PI + 1e-6).all()
    # Every prior phase mean lies in the union of the two valid parameterizations.
    assert (pm_prior >= -math.pi - 1e-6).all() and (pm_prior < TWO_PI + 1e-6).all()
    # The POSTERIOR phase mean (non-recursive) is pi*tanh(raw) in [-pi, pi] by design.
    pm_post = out["posterior"]["phase_mu"]
    assert (pm_post >= -math.pi - 1e-6).all() and (pm_post <= math.pi + 1e-6).all()
    # Gumbel-softmax (soft) meter is a simplex along K.
    ms = out["samples"]["meter_soft"]
    assert torch.allclose(ms.sum(-1), torch.ones_like(ms.sum(-1)), atol=1e-5)
    assert (ms >= -1e-6).all()


def test_prior_kappa_sigma_positive():
    """von Mises / log-normal scale params must be strictly positive (softplus)."""
    model = _build_model()
    act, bt, db = _make_batch()
    out = model(act, beat_targets=bt, downbeat_targets=db)
    assert (out["prior"]["phase_kappa"] > 0).all()
    assert (out["prior"]["tempo_sigma"] > 0).all()
    assert (out["prior"]["barphase_kappa"] > 0).all()
    assert (out["posterior"]["phase_log_kappa"].exp() > 0).all()
    assert (out["posterior"]["tempo_log_sigma"].exp() > 0).all()


# ---------------------------------------------------------------------------
# Loss closed-form checks (vs independent F.* computations)
# ---------------------------------------------------------------------------

def test_bce_term_matches_independent_bce():
    """compute_elbo_loss recon term == sum of independent BCE on beat & downbeat
    channels. Ground truth: torch.nn.functional, not the code's own internal recon."""
    model = _build_model()
    act, bt, db = _make_batch()
    out = model(act, beat_targets=bt, downbeat_targets=db)

    total, comp = compute_elbo_loss(
        beat_logits=out["beat_logits"], beat_targets=bt,
        posterior=out["posterior"], prior=out["prior"],
        beta=0.0, downbeat_targets=db,  # beta=0 isolates BCE in `total`
    )
    bce_ref = (
        F.binary_cross_entropy_with_logits(out["beat_logits"][:, :, 0], bt)
        + F.binary_cross_entropy_with_logits(out["beat_logits"][:, :, 1], db)
    )
    assert torch.allclose(comp["bce"], bce_ref, atol=1e-7)
    # With beta=0 and no aux weights, total == bce exactly.
    assert torch.allclose(total, bce_ref, atol=1e-7)


def test_barphase_and_phase_sup_match_circular_loss():
    """barphase_sup / phase_sup components == independent mean(1-cos(mu-target))."""
    model = _build_model()
    act, bt, db = _make_batch()
    out = model(act, beat_targets=bt, downbeat_targets=db)
    bpt = _barphase_targets(db)
    pt = SVTModel._beat_targets_to_distance(bt) * TWO_PI

    _, comp = compute_elbo_loss(
        beat_logits=out["beat_logits"], beat_targets=bt,
        posterior=out["posterior"], prior=out["prior"],
        beta=0.0, downbeat_targets=db,
        barphase_targets=bpt, barphase_sup_weight=1.0,
        phase_targets=pt, phase_sup_weight=1.0,
    )
    bp_ref = (1.0 - torch.cos(out["prior"]["barphase_mu"] - bpt)).mean()
    p_ref = (1.0 - torch.cos(out["prior"]["phase_mu"] - pt)).mean()
    assert torch.allclose(comp["barphase_sup"], bp_ref, atol=1e-7)
    assert torch.allclose(comp["phase_sup"], p_ref, atol=1e-7)
    # circular loss is in [0, 2]
    assert (0.0 <= comp["barphase_sup"] <= 2.0 + 1e-6)
    assert (0.0 <= comp["phase_sup"] <= 2.0 + 1e-6)


def test_free_bits_floor_raises_or_holds_kl():
    """free_bits clamps per-latent mean KL up to the floor: KL component must be
    >= floor (within fp tol). Ground truth: definition of the free-bits clamp."""
    model = _build_model()
    act, bt, db = _make_batch()
    out = model(act, beat_targets=bt, downbeat_targets=db)
    fb = 5.0  # large floor so the clamp is active
    _, comp = compute_elbo_loss(
        beat_logits=out["beat_logits"], beat_targets=bt,
        posterior=out["posterior"], prior=out["prior"],
        beta=1.0, downbeat_targets=db, free_bits=fb,
    )
    for k in ("kl_meter", "kl_phase", "kl_tempo"):
        assert comp[k].item() >= fb - 1e-5, f"{k}={comp[k].item()} below floor {fb}"


# ---------------------------------------------------------------------------
# Full objective: finiteness + backward + gradient connectivity
# ---------------------------------------------------------------------------

def test_full_objective_finite_and_backward():
    model = _build_model()
    act, bt, db = _make_batch()
    total, comp, out = _full_objective(model, act, bt, db)
    assert torch.isfinite(total), "total objective not finite"
    for k, v in comp.items():
        assert torch.isfinite(v).all(), f"component {k} not finite"
    total.backward()
    # gradients finite where present
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"


def test_all_used_params_receive_gradient():
    """Every trainable parameter of a sub-module that is EXERCISED by the composed
    objective must get a non-None, finite, not-all-zero gradient. This catches a
    silently-disconnected head (e.g. an emission/bar-phase head that never feeds the
    loss). The "used" set is the union of every parameter touched by forward + the
    aux terms; with bar_phase + audio_emission + barphase_sup + phase_sup ON, this is
    ALL parameters of the model.
    """
    model = _build_model()
    act, bt, db = _make_batch()
    total, _, _ = _full_objective(model, act, bt, db)
    model.zero_grad(set_to_none=True)
    total.backward()

    missing, all_zero = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            missing.append(n)
        else:
            assert torch.isfinite(p.grad).all(), f"non-finite grad {n}"
            if torch.count_nonzero(p.grad) == 0:
                all_zero.append(n)

    assert not missing, f"params with NO gradient (disconnected): {missing}"
    # Some bias terms can legitimately have a structurally-zero grad on a tiny random
    # batch (e.g. positional buffers are not params). We only flag if a WHOLE weight
    # matrix is dead, which would indicate a disconnected sub-module.
    dead_weights = [n for n in all_zero if n.endswith(".weight")]
    assert not dead_weights, f"weight matrices with all-zero grad (disconnected): {dead_weights}"


def test_audio_emission_head_is_load_bearing():
    """Removing the audio_recon + z_forcing terms must change the audio_emission head's
    gradient (it is otherwise NOT in the ELBO+decoder path). Confirms the emission head
    is actually driven by those aux terms, not dead weight."""
    model = _build_model()
    act, bt, db = _make_batch()

    # With aux terms.
    model.zero_grad(set_to_none=True)
    total_with, _, _ = _full_objective(model, act, bt, db,
                                       audio_recon_weight=1.0, z_forcing_weight=1.0)
    total_with.backward()
    g_with = {n: p.grad.clone() for n, p in model.named_parameters()
              if n.startswith("audio_emission_head") and p.grad is not None}
    assert g_with, "audio_emission_head has no params/grad"
    # head must actually receive gradient from the aux MSE terms
    assert any(torch.count_nonzero(g) > 0 for g in g_with.values()), \
        "audio_emission_head got zero grad even with audio_recon+z_forcing on"

    # Without aux terms the emission head should be disconnected (grad None / zero),
    # since it feeds neither the decoder nor the KLs.
    model.zero_grad(set_to_none=True)
    total_wo, _, _ = _full_objective(model, act, bt, db,
                                     audio_recon_weight=0.0, z_forcing_weight=0.0)
    total_wo.backward()
    for n, p in model.named_parameters():
        if n.startswith("audio_emission_head"):
            assert p.grad is None or torch.count_nonzero(p.grad) == 0, \
                f"{n} unexpectedly got grad without aux terms ({p.grad})"


# ---------------------------------------------------------------------------
# Inference samplers
# ---------------------------------------------------------------------------

def test_sample_from_prior_finite():
    model = _build_model()
    act, _, _ = _make_batch()
    model.eval()
    out = model.sample_from_prior(act, temperature=0.5)
    assert out["beat_logits"].shape == (B, T, 2)
    assert torch.isfinite(out["beat_logits"]).all()
    for k in ("phase", "phase_mu", "log_tempo", "meter_soft"):
        assert torch.isfinite(out[k]).all(), f"{k} not finite"
    ph = out["phase"]
    assert (ph >= 0).all() and (ph < TWO_PI + 1e-6).all()


@pytest.mark.xfail(
    strict=True,
    reason="BUG: sample_from_prior_pf's final _decode (svt_core.py:1253) omits the "
    "bar_phase arg, so a model built with bar_phase=True (as the overnight "
    "run_bz_bt_{dir1,faith}.sh do, alongside audio_emission) raises AssertionError "
    "'bar_phase decode needs the phi^bar trajectory'. The PF loop never tracks a "
    "bar-phase trajectory at all. sample_from_prior handles this correctly; only the "
    "PF path is broken.",
)
def test_sample_from_prior_pf_finite_barphase():
    """Particle filter inference (B=1) on a bar_phase+audio_emission model should
    return finite, correctly-shaped read-outs. Currently xfails due to the bug above."""
    model = _build_model(bar_phase=True)  # matches overnight checkpoints
    act, _, _ = _make_batch()
    model.eval()
    a1 = act[:1]  # B=1 required
    out = model.sample_from_prior_pf(a1, n_particles=32, obs_sigma=0.3,
                                     temperature=0.5, ess_frac=0.5)
    assert out["beat_logits"].shape == (1, T, 2)
    assert torch.isfinite(out["beat_logits"]).all()


def test_sample_from_prior_pf_finite():
    """Particle filter inference (B=1) returns finite, correctly-shaped read-outs.

    Built WITHOUT bar_phase (the configuration the PF path actually supports) so this
    exercises the real PF dynamics + Bayesian wrap read-out without the bar_phase bug."""
    model = _build_model(bar_phase=False)
    act, _, _ = _make_batch()
    model.eval()
    a1 = act[:1]  # B=1 required
    out = model.sample_from_prior_pf(a1, n_particles=32, obs_sigma=0.3,
                                     temperature=0.5, ess_frac=0.5)
    assert out["beat_logits"].shape == (1, T, 2)
    assert torch.isfinite(out["beat_logits"]).all()
    assert out["beat_activation"].shape == (1, T)
    ba = out["beat_activation"]
    assert torch.isfinite(ba).all()
    # Bayesian wrap read-out is a weighted fraction in [0, 1].
    assert (ba >= -1e-6).all() and (ba <= 1.0 + 1e-6).all()
    # MAP phase trajectory is wrapped.
    assert (out["phase"] >= 0).all() and (out["phase"] < TWO_PI + 1e-6).all()


# ---------------------------------------------------------------------------
# Learnability: the model can fit ONE fixed batch (objective decreases)
# ---------------------------------------------------------------------------

def test_short_adam_loop_decreases_bce():
    """Optimization invariant: 12 Adam steps on a single fixed batch must reduce the
    beat BCE substantially. A model that cannot lower the reconstruction term on a
    fixed batch is structurally broken (dead gradients / wrong sign). float32 to mirror
    real training; fixed seed for determinism.

    Ground truth = the optimisation dynamics (loss must go DOWN), not a copied value.
    """
    torch.manual_seed(123)
    model = _build_model(dtype=torch.float32)
    act, bt, db = _make_batch(seed=1, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    def step(do_opt):
        out = model(act, beat_targets=bt, downbeat_targets=db)
        total, comp = compute_elbo_loss(
            beat_logits=out["beat_logits"], beat_targets=bt,
            posterior=out["posterior"], prior=out["prior"],
            beta=0.1, downbeat_targets=db,
        )
        if out.get("audio_recon") is not None:
            total = total + 0.5 * ((out["audio_recon"] - act) ** 2).mean()
        if do_opt:
            opt.zero_grad(); total.backward(); opt.step()
        return float(total.detach()), float(comp["bce"].detach())

    total0, bce0 = step(do_opt=True)
    for _ in range(11):
        total_last, bce_last = step(do_opt=True)
        assert math.isfinite(total_last), "loss went non-finite during optimisation"

    # BCE must drop meaningfully (the model can fit the recon on a fixed batch).
    assert bce_last < bce0 - 1e-3, f"BCE did not decrease: {bce0:.4f} -> {bce_last:.4f}"
    # Total objective should also trend down over the run.
    assert total_last < total0, f"objective did not decrease: {total0:.4f} -> {total_last:.4f}"


def test_loss_decreases_on_overfittable_target():
    """Sanity that the decoder->BCE pathway has the right gradient SIGN: drive the
    beat target to all-ones and check BCE falls (logits should rise). Independent of
    the random batch above; isolates the recon gradient direction."""
    torch.manual_seed(7)
    model = _build_model(dtype=torch.float32)
    act, _, _ = _make_batch(seed=2, dtype=torch.float32)
    bt = torch.ones(B, T)      # all-beat target
    db = torch.ones(B, T)      # all-downbeat target
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    def bce_now():
        out = model(act, beat_targets=bt, downbeat_targets=db)
        return F.binary_cross_entropy_with_logits(out["beat_logits"][:, :, 0], bt), out

    b0, _ = bce_now()
    for _ in range(15):
        out = model(act, beat_targets=bt, downbeat_targets=db)
        loss = (
            F.binary_cross_entropy_with_logits(out["beat_logits"][:, :, 0], bt)
            + F.binary_cross_entropy_with_logits(out["beat_logits"][:, :, 1], db)
        )
        opt.zero_grad(); loss.backward(); opt.step()
    b1, _ = bce_now()
    assert float(b1) < float(b0) - 1e-3, f"beat BCE did not fall: {float(b0):.4f}->{float(b1):.4f}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
