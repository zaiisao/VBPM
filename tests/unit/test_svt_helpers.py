"""Unit tests for pure helpers in models/svt_core.py.

Targets:
  - SVTModel._beat_targets_to_distance  (vectorized "fraction since last beat")
  - SVTModel._parse_phase_head          (atan2 wrap + softplus kappa)
  - PositionalEncoding                  (sinusoidal PE, batch-first)

Ground truth strategy:
  - _beat_targets_to_distance: independent plain-numpy reference encoding the
    DOCUMENTED semantics  distance[t] = (t - last_beat)/(next_beat - last_beat),
    with last_beat = 0 sentinel when no prior beat and next_beat = T sentinel
    when no following beat. Plus structural invariants ([0,1), zero at beats).
  - _parse_phase_head: closed-form math (atan2 of known angles, softplus>=0).
  - PositionalEncoding: closed-form sinusoid formula + shape/bound invariants.

No tautologies: every assertion is against an independent reference or an
analytic invariant, never against another call of the same function.
"""

import sys
sys.path.insert(0, "/home/sogang/jaehoon/CHART")

import math

import numpy as np
import pytest
import torch

from models.svt_core import SVTModel, PositionalEncoding

TWO_PI = 2.0 * math.pi
_dist = SVTModel._beat_targets_to_distance
_parse = SVTModel._parse_phase_head


# ---------------------------------------------------------------------------
# Independent numpy reference for the beat-distance feature
# ---------------------------------------------------------------------------

def _np_beat_distance(bt_np: np.ndarray) -> np.ndarray:
    """Plain-Python/numpy reference of the documented [0,1) semantics.

    distance[t] = (t - last_beat) / (next_beat - last_beat), clamped to [0,1],
    span clamped to >= 1, with:
      last_beat = largest beat index <= t, or 0 if none seen yet
      next_beat = smallest beat index  > t, or T (sentinel) if none after t
    """
    B, T = bt_np.shape
    out = np.zeros((B, T), dtype=np.float64)
    for b in range(B):
        beats = np.where(bt_np[b] > 0.5)[0]
        for t in range(T):
            le = beats[beats <= t]
            last = int(le.max()) if le.size else 0
            gt = beats[beats > t]
            nxt = int(gt.min()) if gt.size else T
            span = max(nxt - last, 1)
            val = (t - last) / span
            out[b, t] = min(max(val, 0.0), 1.0)
    return out


# ---------------------------------------------------------------------------
# _beat_targets_to_distance  — vs independent numpy reference
# ---------------------------------------------------------------------------

BEAT_PATTERNS = {
    "single_beat_mid": [0, 0, 1, 0, 0, 0],
    "single_beat_first": [1, 0, 0, 0, 0],
    "single_beat_last": [0, 0, 0, 0, 1],
    "first_beat_not_at_0": [0, 0, 0, 1, 0, 0],
    "evenly_spaced": [1, 0, 1, 0, 1, 0, 1, 0],
    "evenly_spaced_span3": [1, 0, 0, 1, 0, 0, 1, 0, 0],
    "no_beats": [0, 0, 0, 0, 0],
    "beats_both_ends": [1, 0, 0, 0, 1],
    "all_beats": [1, 1, 1, 1],
    "irregular": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    "adjacent_beats": [0, 1, 1, 0, 1, 0],
}


@pytest.mark.parametrize("name", list(BEAT_PATTERNS.keys()))
def test_beat_distance_matches_numpy_reference(name):
    pat = BEAT_PATTERNS[name]
    bt = torch.tensor([pat], dtype=torch.float32)
    got = _dist(bt).cpu().numpy()
    ref = _np_beat_distance(np.asarray([pat], dtype=np.float64))
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=1e-6, rtol=0,
                               err_msg=f"pattern {name}")


def test_beat_distance_batched_matches_reference():
    """Independent rows in a batch must each match the reference."""
    pats = [
        [0, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],   # no beats
        [1, 0, 0, 0, 0, 0, 0, 1],
    ]
    bt = torch.tensor(pats, dtype=torch.float32)
    got = _dist(bt).cpu().numpy()
    ref = _np_beat_distance(np.asarray(pats, dtype=np.float64))
    np.testing.assert_allclose(got, ref, atol=1e-6, rtol=0)


def test_beat_distance_random_matches_reference():
    """Randomized fuzz vs the numpy reference (Monte-Carlo coverage)."""
    g = torch.Generator().manual_seed(0)
    for _ in range(40):
        B = int(torch.randint(1, 4, (1,), generator=g).item())
        T = int(torch.randint(2, 30, (1,), generator=g).item())
        p = float(torch.rand(1, generator=g).item()) * 0.6  # beat density
        bt = (torch.rand(B, T, generator=g) < p).float()
        got = _dist(bt).cpu().numpy()
        ref = _np_beat_distance(bt.cpu().numpy().astype(np.float64))
        np.testing.assert_allclose(got, ref, atol=1e-6, rtol=0)


def test_beat_distance_invariant_range_in_unit_interval():
    """Documented contract: all values in [0, 1) (strictly < 1)."""
    g = torch.Generator().manual_seed(1)
    for _ in range(30):
        B = int(torch.randint(1, 3, (1,), generator=g).item())
        T = int(torch.randint(2, 25, (1,), generator=g).item())
        bt = (torch.rand(B, T, generator=g) < 0.3).float()
        d = _dist(bt)
        assert torch.all(d >= 0.0)
        # Strictly below 1: span is clamped >=1 and numerator <= span-1,
        # so the max attainable value is (span-1)/span < 1.
        assert torch.all(d < 1.0), d.max().item()


def test_beat_distance_zero_at_beat_frames():
    """distance == 0 exactly at every beat frame."""
    g = torch.Generator().manual_seed(2)
    for _ in range(30):
        T = int(torch.randint(2, 25, (1,), generator=g).item())
        bt = (torch.rand(1, T, generator=g) < 0.4).float()
        d = _dist(bt)[0]
        mask = bt[0] > 0.5
        if mask.any():
            assert torch.allclose(d[mask], torch.zeros(int(mask.sum())),
                                  atol=1e-7)


def test_beat_distance_monotone_ramp_between_beats():
    """Between two consecutive beats the distance ramps strictly upward."""
    bt = torch.tensor([[1, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=torch.float32)
    d = _dist(bt)[0]
    # frames 0..4 are one segment (beat at 0, next beat at 5)
    seg = d[0:5]
    diffs = seg[1:] - seg[:-1]
    assert torch.all(diffs > 0), seg.tolist()
    # constant step (linear ramp) of 1/span = 1/5
    assert torch.allclose(diffs, torch.full_like(diffs, 1.0 / 5.0), atol=1e-6)


def test_beat_distance_all_beats_is_all_zero():
    bt = torch.ones(1, 6)
    d = _dist(bt)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-7)


def test_beat_distance_no_beats_clean_ramp_below_one():
    """No-beat row ramps t/T (stays < 1), per the [0,1) contract."""
    T = 5
    bt = torch.zeros(1, T)
    d = _dist(bt)[0]
    expected = torch.arange(T, dtype=torch.float32) / T  # 0, .2, .4, .6, .8
    assert torch.allclose(d, expected, atol=1e-6)
    assert d.max().item() < 1.0


def test_beat_distance_dtype_and_shape_preserved():
    bt = torch.zeros(3, 7)
    d = _dist(bt)
    assert d.shape == (3, 7)
    assert torch.isfinite(d).all()


# ---------------------------------------------------------------------------
# _parse_phase_head  — atan2 wrap + softplus kappa
# ---------------------------------------------------------------------------

def test_parse_phase_head_mu_matches_atan2_wrapped():
    """mu == atan2(sin, cos) wrapped to [0, 2pi); kappa == softplus(raw)."""
    # cover all 4 quadrants + axes with known target angles
    angles = torch.tensor([0.0, math.pi / 6, math.pi / 2, 2.0 * math.pi / 3,
                           math.pi, -math.pi / 2, -3.0 * math.pi / 4, 1e-4])
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    raw_k = torch.linspace(-4.0, 4.0, angles.numel())
    out3 = torch.stack([cos, sin, raw_k], dim=-1)  # [..., 3] = [cos, sin, kraw]
    mu, kappa = _parse(out3)

    ref_mu = torch.remainder(torch.atan2(sin, cos), TWO_PI)
    assert torch.allclose(mu, ref_mu, atol=1e-6)
    # also equals the original angle reduced mod 2pi
    assert torch.allclose(mu, torch.remainder(angles, TWO_PI), atol=1e-5)

    ref_kappa = torch.nn.functional.softplus(raw_k)
    assert torch.allclose(kappa, ref_kappa, atol=1e-6)


def test_parse_phase_head_mu_in_range_and_unit_circle_recovered():
    """mu in [0, 2pi); cos/sin(mu) reproduce the normalized input direction."""
    g = torch.Generator().manual_seed(3)
    vec = torch.randn(200, 2, generator=g)
    # avoid exact-zero vectors (atan2(0,0)=0 is defined but uninformative)
    vec = vec + 0.01 * torch.sign(vec + (vec == 0).float())
    raw_k = torch.randn(200, 1, generator=g)
    out3 = torch.cat([vec[:, :1], vec[:, 1:2], raw_k], dim=-1)  # [cos_raw, sin_raw, kraw]
    mu, kappa = _parse(out3)

    assert torch.all(mu >= 0.0) and torch.all(mu < TWO_PI)
    # Recovered unit direction must match the normalized (cos_raw, sin_raw).
    norm = vec / vec.norm(dim=-1, keepdim=True)
    rec = torch.stack([torch.cos(mu), torch.sin(mu)], dim=-1)
    assert torch.allclose(rec, norm, atol=1e-5)


def test_parse_phase_head_kappa_nonnegative_and_softplus():
    """kappa = softplus(raw) >= 0 and strictly > 0, monotone in raw."""
    raw = torch.linspace(-50.0, 50.0, 101)
    out3 = torch.stack([torch.ones_like(raw), torch.zeros_like(raw), raw], dim=-1)
    _, kappa = _parse(out3)
    assert torch.all(kappa >= 0.0)
    assert torch.all(kappa > 0.0)  # softplus is strictly positive
    # monotone increasing in raw
    assert torch.all(kappa[1:] - kappa[:-1] >= -1e-7)
    # closed form
    assert torch.allclose(kappa, torch.nn.functional.softplus(raw), atol=1e-6)


def test_parse_phase_head_batched_shapes():
    out3 = torch.randn(4, 11, 3)
    mu, kappa = _parse(out3)
    assert mu.shape == (4, 11)
    assert kappa.shape == (4, 11)
    assert torch.isfinite(mu).all() and torch.isfinite(kappa).all()


# ---------------------------------------------------------------------------
# PositionalEncoding
# ---------------------------------------------------------------------------

def test_positional_encoding_output_shape_equals_input():
    pe = PositionalEncoding(d_model=16, max_len=100)
    x = torch.zeros(2, 7, 16)
    y = pe(x)
    assert y.shape == x.shape


def test_positional_encoding_added_table_in_minus1_1():
    """The added sinusoidal table is bounded in [-1, 1] (sin/cos)."""
    pe = PositionalEncoding(d_model=32, max_len=500)
    x = torch.zeros(1, 200, 32)   # x=0 -> output == pure PE table
    y = pe(x)
    assert torch.all(y <= 1.0 + 1e-6)
    assert torch.all(y >= -1.0 - 1e-6)
    # table is not trivially zero
    assert y.abs().max() > 0.5


def test_positional_encoding_is_additive_residual():
    """forward(x) == x + PE(positions); independent of x content."""
    pe = PositionalEncoding(d_model=24, max_len=64)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(3, 20, 24, generator=g)
    zeros = torch.zeros(3, 20, 24)
    table = pe(zeros)            # the added PE for these positions
    y = pe(x)
    assert torch.allclose(y, x + table, atol=1e-6)


def test_positional_encoding_matches_closed_form_sinusoid():
    """Compare the added table to the textbook PE formula computed in numpy."""
    d_model, T = 16, 25
    pe = PositionalEncoding(d_model=d_model, max_len=200)
    table = pe(torch.zeros(1, T, d_model))[0].cpu().numpy()

    pos = np.arange(T)[:, None]
    i2 = np.arange(0, d_model, 2)
    div = np.exp(i2 * (-math.log(10000.0) / d_model))
    ref = np.zeros((T, d_model))
    ref[:, 0::2] = np.sin(pos * div)
    ref[:, 1::2] = np.cos(pos * div)
    np.testing.assert_allclose(table, ref, atol=1e-5, rtol=0)


def test_positional_encoding_position0_pattern():
    """At position 0: sin(0)=0 on even dims, cos(0)=1 on odd dims."""
    d_model = 12
    pe = PositionalEncoding(d_model=d_model, max_len=10)
    table = pe(torch.zeros(1, 1, d_model))[0, 0].cpu().numpy()
    assert np.allclose(table[0::2], 0.0, atol=1e-6)   # sin(0)
    assert np.allclose(table[1::2], 1.0, atol=1e-6)   # cos(0)


def test_positional_encoding_distinct_positions_differ():
    """Different time positions get different encodings (injective table)."""
    pe = PositionalEncoding(d_model=16, max_len=50)
    table = pe(torch.zeros(1, 30, 16))[0]
    # pairwise rows should not be identical
    for a in range(0, 30, 7):
        for b in range(a + 1, 30, 7):
            assert not torch.allclose(table[a], table[b], atol=1e-4)


def test_positional_encoding_finite_and_no_nan():
    pe = PositionalEncoding(d_model=8, max_len=1000)
    y = pe(torch.randn(2, 300, 8))
    assert torch.isfinite(y).all()
