"""Unit tests for models/svt_core.py — bar_phase + meter_ste + decoder_latent_only.

Tests assert against STRUCTURAL INVARIANTS and CLOSED-FORM facts:
  - simplex / one-hot properties of the meter sample,
  - exact decoder input-dim arithmetic (3 + K + bar_dim + h_dim) re-derived
    independently from the model config and checked against the actual
    nn.Linear.in_features the module built,
  - presence/absence of barphase keys gated on the bar_phase flag,
  - shape and finiteness of all returned tensors,
  - phase samples live on the circle [0, 2π).

No value is copied from a debug run; every expected number is derived from the
documented layout in the source or from torch ground-truth (one_hot, softmax).
"""

import sys

sys.path.insert(0, "/home/sogang/jaehoon/CHART")

import math

import pytest
import torch

from models.svt_core import SVTModel, TWO_PI

torch.manual_seed(0)

DEVICE = "cpu"
HID = 16          # small hidden dim
NHEAD = 2
NLAYERS = 1
K = 4             # num_meter_classes
INPUT_DIM = 2
B, T = 2, 12


def _make_model(**overrides):
    cfg = dict(
        hidden_dim=HID,
        nhead=NHEAD,
        num_layers=NLAYERS,
        num_meter_classes=K,
        input_dim=INPUT_DIM,
    )
    cfg.update(overrides)
    m = SVTModel(**cfg).to(DEVICE)
    m.eval()
    return m


def _inputs(b=B, t=T):
    acts = torch.randn(b, t, INPUT_DIM, device=DEVICE)
    beat = (torch.rand(b, t, device=DEVICE) > 0.7).float()
    down = (torch.rand(b, t, device=DEVICE) > 0.9).float()
    return acts, beat, down


# ---------------------------------------------------------------------------
# Decoder input-dim arithmetic: 3 + K + bar_dim + h_dim  (closed form)
# ---------------------------------------------------------------------------

def _expected_decoder_in_dim(*, K, bar_phase, decoder_use_h_prior,
                             h_prior_bottleneck, hidden_dim):
    if not decoder_use_h_prior:
        h_dim = 0
    elif h_prior_bottleneck > 0:
        h_dim = h_prior_bottleneck
    else:
        h_dim = hidden_dim
    bar_dim = 2 if bar_phase else 0
    return 3 + K + bar_dim + h_dim


def _decoder_in_features(model):
    """First Linear of emission_decoder -> its in_features."""
    first = model.emission_decoder[0]
    assert isinstance(first, torch.nn.Linear)
    return first.in_features


@pytest.mark.parametrize("bar_phase", [False, True])
@pytest.mark.parametrize("decoder_use_h_prior", [True, False])
def test_decoder_input_dim_matches_closed_form(bar_phase, decoder_use_h_prior):
    m = _make_model(bar_phase=bar_phase, decoder_use_h_prior=decoder_use_h_prior)
    expected = _expected_decoder_in_dim(
        K=K, bar_phase=bar_phase, decoder_use_h_prior=decoder_use_h_prior,
        h_prior_bottleneck=0, hidden_dim=HID,
    )
    assert _decoder_in_features(m) == expected


def test_bar_phase_adds_exactly_two_to_decoder():
    """Turning on bar_phase must add EXACTLY +2 (cos/sin φ^bar) to decoder in-dim."""
    base = _make_model(bar_phase=False)
    barp = _make_model(bar_phase=True)
    assert _decoder_in_features(barp) - _decoder_in_features(base) == 2


def test_latent_only_decoder_drops_h_prior_dim():
    """decoder_use_h_prior=False removes exactly hidden_dim from the decoder in-dim."""
    with_h = _make_model(decoder_use_h_prior=True)
    latent_only = _make_model(decoder_use_h_prior=False)
    assert _decoder_in_features(with_h) - _decoder_in_features(latent_only) == HID
    # latent-only with K meter classes: 3 + K + 0(bar) + 0(h)
    assert _decoder_in_features(latent_only) == 3 + K


def test_h_prior_bottleneck_changes_decoder_dim():
    bott = 5
    m = _make_model(h_prior_bottleneck=bott)
    expected = _expected_decoder_in_dim(
        K=K, bar_phase=False, decoder_use_h_prior=True,
        h_prior_bottleneck=bott, hidden_dim=HID,
    )
    assert _decoder_in_features(m) == expected
    # bottleneck path uses bottleneck dim, NOT hidden_dim
    assert _decoder_in_features(m) == 3 + K + bott


# ---------------------------------------------------------------------------
# meter_ste: hard one-hot forward vs soft simplex
# ---------------------------------------------------------------------------

def test_meter_ste_true_is_hard_onehot_in_forward():
    m = _make_model(meter_ste=True)
    acts, beat, down = _inputs()
    out = m(acts, beat_targets=beat, downbeat_targets=down)
    meter = out["samples"]["meter_soft"]  # [B, T, K]
    assert meter.shape == (B, T, K)
    # Rows sum to 1 (simplex / one-hot).
    s = meter.sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    # Hard one-hot: the '1' sits at the argmax position (STE forward = argmax).
    argmax_oh = torch.nn.functional.one_hot(
        meter.argmax(dim=-1), num_classes=K
    ).to(meter.dtype)
    assert torch.equal(meter.round(), argmax_oh)
    # Strict: every value is (numerically) 0 or 1.
    near01 = (meter < 1e-5) | (meter > 1.0 - 1e-5)
    assert near01.all(), "meter_ste=True must give hard 0/1 entries"
    # Exactly one '1' per row.
    n_ones = meter.round().sum(dim=-1)
    assert torch.equal(n_ones, torch.ones_like(n_ones))


def test_meter_ste_false_is_soft_simplex_not_onehot():
    m = _make_model(meter_ste=False)
    acts, beat, down = _inputs()
    out = m(acts, beat_targets=beat, downbeat_targets=down)
    meter = out["samples"]["meter_soft"]
    # Simplex: nonneg + rows sum to 1.
    assert (meter >= 0).all()
    s = meter.sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    # Soft: NOT all entries are 0/1 (with high prob given random logits & tau=1).
    near01 = (meter < 1e-5) | (meter > 1.0 - 1e-5)
    assert not near01.all(), "meter_ste=False should be a soft simplex, not one-hot"


def test_meter_hard_flag_wired_from_meter_ste():
    assert _make_model(meter_ste=True).meter_hard is True
    assert _make_model(meter_ste=False).meter_hard is False


def test_meter_ste_sample_from_prior_hard():
    m = _make_model(meter_ste=True)
    acts, _, _ = _inputs()
    out = m.sample_from_prior(acts, temperature=0.5)
    meter = out["meter_soft"]
    near01 = (meter < 1e-5) | (meter > 1.0 - 1e-5)
    assert near01.all()
    n_ones = meter.round().sum(dim=-1)
    assert torch.equal(n_ones, torch.ones_like(n_ones))


# ---------------------------------------------------------------------------
# bar_phase=True: keys present in posterior / prior / samples / sample_from_prior
# ---------------------------------------------------------------------------

def test_bar_phase_true_forward_keys_and_shapes():
    m = _make_model(bar_phase=True)
    acts, beat, down = _inputs()
    out = m(acts, beat_targets=beat, downbeat_targets=down)

    post, prior, samples = out["posterior"], out["prior"], out["samples"]

    # Posterior carries barphase_mu + barphase_log_kappa (source uses log_kappa here).
    assert "barphase_mu" in post
    assert "barphase_log_kappa" in post
    assert post["barphase_mu"].shape == (B, T)
    assert post["barphase_log_kappa"].shape == (B, T)

    # Prior carries barphase_mu + barphase_kappa.
    assert "barphase_mu" in prior
    assert "barphase_kappa" in prior
    assert prior["barphase_mu"].shape == (B, T)
    assert prior["barphase_kappa"].shape == (B, T)

    # Samples carry bar_phase trajectory.
    assert "bar_phase" in samples
    assert samples["bar_phase"].shape == (B, T)

    # bar_phase sample lives on the circle [0, 2π).
    bp = samples["bar_phase"]
    assert torch.isfinite(bp).all()
    assert (bp >= 0).all() and (bp < TWO_PI + 1e-4).all()

    # prior barphase_kappa is a valid concentration (>= 0 via softplus).
    assert (prior["barphase_kappa"] >= 0).all()


def test_bar_phase_true_sample_from_prior_keys():
    m = _make_model(bar_phase=True)
    acts, _, _ = _inputs()
    out = m.sample_from_prior(acts, temperature=0.3)
    assert "bar_phase" in out
    assert "bar_phase_mu" in out
    assert out["bar_phase"].shape == (B, T)
    assert out["bar_phase_mu"].shape == (B, T)
    for k in ("bar_phase", "bar_phase_mu"):
        v = out[k]
        assert torch.isfinite(v).all()
        assert (v >= 0).all() and (v < TWO_PI + 1e-4).all()


# ---------------------------------------------------------------------------
# bar_phase=False: NO barphase keys anywhere (the "--" case)
# ---------------------------------------------------------------------------

def test_bar_phase_false_has_no_barphase_keys():
    m = _make_model(bar_phase=False)
    acts, beat, down = _inputs()
    out = m(acts, beat_targets=beat, downbeat_targets=down)

    post, prior, samples = out["posterior"], out["prior"], out["samples"]
    for d, name in ((post, "posterior"), (prior, "prior")):
        bad = [k for k in d if "barphase" in k or "bar_phase" in k]
        assert not bad, f"{name} unexpectedly has bar keys: {bad}"
    assert "bar_phase" not in samples

    sp = m.sample_from_prior(acts, temperature=0.3)
    assert "bar_phase" not in sp
    assert "bar_phase_mu" not in sp

    # The model also should not register bar-phase submodules.
    assert not hasattr(m, "bar_post_head")
    assert not hasattr(m, "prior_bar_incr_ffn")


def test_bar_phase_flag_creates_submodules():
    m = _make_model(bar_phase=True)
    for attr in ("bar_init_post_head", "bar_init_prior_head", "bar_post_head",
                 "prior_bar_incr_ffn", "prior_bar_kappa_ffn", "prior_bar_corr_ffn"):
        assert hasattr(m, attr), f"bar_phase=True must build {attr}"


# ---------------------------------------------------------------------------
# Core output contract (shapes / finiteness) holds for all flag combos
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bar_phase", [False, True])
@pytest.mark.parametrize("meter_ste", [False, True])
@pytest.mark.parametrize("decoder_use_h_prior", [True, False])
def test_forward_output_contract(bar_phase, meter_ste, decoder_use_h_prior):
    m = _make_model(bar_phase=bar_phase, meter_ste=meter_ste,
                    decoder_use_h_prior=decoder_use_h_prior)
    acts, beat, down = _inputs()
    out = m(acts, beat_targets=beat, downbeat_targets=down)

    # beat_logits shape [B, T, 2], finite.
    bl = out["beat_logits"]
    assert bl.shape == (B, T, 2)
    assert torch.isfinite(bl).all()

    samples = out["samples"]
    assert samples["phase"].shape == (B, T)
    assert samples["log_tempo"].shape == (B, T)
    assert samples["meter_soft"].shape == (B, T, K)

    # phase on the circle, all latents finite.
    ph = samples["phase"]
    assert (ph >= 0).all() and (ph < TWO_PI + 1e-4).all()
    for v in samples.values():
        assert torch.isfinite(v).all()

    # posterior / prior shared keys present & finite.
    for d in (out["posterior"], out["prior"]):
        for key in ("meter_logits", "phase_mu", "tempo_mu"):
            assert key in d
            assert torch.isfinite(d[key]).all()
        assert d["meter_logits"].shape == (B, T, K)


# ---------------------------------------------------------------------------
# sample_from_prior output contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bar_phase", [False, True])
def test_sample_from_prior_contract(bar_phase):
    m = _make_model(bar_phase=bar_phase)
    acts, _, _ = _inputs()
    out = m.sample_from_prior(acts, temperature=0.2)
    for key in ("phase", "phase_mu", "log_tempo", "meter_soft", "meter_onehot",
                "beat_logits"):
        assert key in out, f"missing {key}"
        assert torch.isfinite(out[key]).all()
    assert out["beat_logits"].shape == (B, T, 2)
    # meter_onehot is a genuine one-hot regardless of meter_ste.
    oh = out["meter_onehot"]
    assert oh.shape == (B, T, K)
    near01 = (oh < 1e-6) | (oh > 1.0 - 1e-6)
    assert near01.all()
    n_ones = oh.round().sum(dim=-1)
    assert torch.equal(n_ones, torch.ones_like(n_ones))
    # phase_mu trajectory is the deterministic mean chain, on the circle.
    pm = out["phase_mu"]
    assert (pm >= 0).all() and (pm < TWO_PI + 1e-4).all()


# ---------------------------------------------------------------------------
# meter_onehot ground-truth: argmax(meter_soft) == argmax(meter_onehot)
# ---------------------------------------------------------------------------

def test_sample_from_prior_meter_onehot_matches_argmax():
    """meter_onehot must one-hot the argmax of meter_soft (torch.one_hot ground truth)."""
    m = _make_model(bar_phase=False)
    acts, _, _ = _inputs()
    out = m.sample_from_prior(acts, temperature=0.5)
    soft = out["meter_soft"]
    oh = out["meter_onehot"]
    gt = torch.nn.functional.one_hot(soft.argmax(dim=-1), num_classes=K).to(oh.dtype)
    assert torch.equal(oh, gt)
