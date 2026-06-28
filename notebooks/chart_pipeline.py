"""CHART pipeline — the deployable bar-pointer beat tracker, as presented in ELBO_for_DBN.ipynb.

This is the *working* pipeline distilled from the investigation: a FIXED bar-pointer prior + an audio
beat-activation observation + Sequential-Monte-Carlo (particle-filter) inference. It is the classic
DBN's inference machinery wrapped around the document's bar-pointer latent geometry. Self-contained so
the notebook can import it; verified clean by the oracle/component stress-tests.

Key entry points:
  smc_track(activation, m, fps, sig_t, ...)  -> (beat_times, bpm_map, post_bpm, post_w)   # the PF
  free_run_phase(T, m, fps)                  -> phi trajectory with NO observation (the deploy baseline)
  oracle_activation(bpm_fn, dur, fps)        -> (synthetic perfect activation, true beat times)
  load_easy(i) / load_smc(tid)               -> cached Beat-This activations + ground truth
  f1(ref,est), continuity(ref,est)           -> mir_eval metrics (F, CMLt, AMLt)
"""
from __future__ import annotations
import os, glob, math
import numpy as np
import mir_eval

try:
    import torch
    _DEV = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch = None
    _DEV = "cpu"

TWO_PI = 2.0 * math.pi

# ----- data locations (cached Beat-This activations from the investigation) -----
_EASY = "/home/sogang/jaehoon/CHART/cache/acts/bt_val_rich"          # song_*.pt: act2[T,2], beat_targets, fps=86.13
_SMC_ACT = "/home/sogang/jaehoon/Analyze-SMC/beat_this_activations_cache"  # smc_XXX.npz: beat,downbeat @50fps
_SMC_ANN = "/home/sogang/jaehoon/Analyze-SMC/smc_metadata/annotations"     # SMC_XXX_*.txt beat times (sec)
FPS_EASY = 86.1328125
FPS_SMC = 50.0


# ===================== metrics =====================
def f1(ref, est):
    ref = np.asarray(ref, float); est = np.asarray(est, float)
    if len(ref) < 2: return float("nan")
    if len(est) == 0: return 0.0
    return float(mir_eval.beat.f_measure(ref, est))


def continuity(ref, est):
    """returns (CMLt, AMLt) — AMLt is octave/metrical-level tolerant."""
    ref = np.asarray(ref, float); est = np.asarray(est, float)
    if len(ref) < 2 or len(est) < 2: return (0.0, 0.0)
    c = mir_eval.beat.continuity(ref, est)
    return float(c[1]), float(c[3])


# ===================== the particle filter (SMC inference) =====================
def smc_track(activation, m=4, fps=FPS_SMC, sig_t=0.08, kappa_b=16.0, K=1000,
              bpm_lo=40.0, bpm_hi=250.0, seed=0):
    """Bar-pointer particle filter. FIXED prior (broad uniform tempo init + fixed-form log-tempo
    random walk + deterministic phase advance — NO trainable/audio-conditioned dynamics). Observation
    = the audio beat-activation via the geometric template T(phi)=exp(kappa*(cos(m*phi)-1)).

    activation: np[T] beat-probability in [0,1].  Returns (beat_times_sec, bpm_map, post_bpm, post_w).
    NOTE: read tempo from the OUTPUT beats, not bpm_map at high sig_t (bpm_map low-biases at slow tempo).
    """
    assert torch is not None, "torch required"
    dev = _DEV; rng = np.random.RandomState(seed); T = len(activation)
    a = torch.tensor(np.clip(activation, 1e-3, 1 - 1e-3), device=dev, dtype=torch.float32)
    la, l1a = torch.log(a), torch.log(1 - a)
    LT_MIN = math.log(TWO_PI * bpm_lo / 60 / m / fps); LT_MAX = math.log(TWO_PI * bpm_hi / 60 / m / fps)
    lt = torch.tensor(np.log(TWO_PI * rng.uniform(bpm_lo, bpm_hi, K) / 60 / m / fps), device=dev, dtype=torch.float32)
    phi = torch.tensor(rng.uniform(0, TWO_PI, K), device=dev, dtype=torch.float32)
    logw = torch.zeros(K, device=dev)
    phi_steps = np.empty((T, K), np.float32); res = np.empty((T, K), np.int64)
    with torch.no_grad():
        for t in range(T):
            if t > 0:
                lt = (lt + sig_t * torch.randn(K, device=dev)).clamp(LT_MIN, LT_MAX)   # fixed-form tempo RW
                phi = (phi + torch.exp(lt)) % TWO_PI                                    # deterministic advance
            Tb = torch.exp(kappa_b * (torch.cos(m * phi) - 1.0))                        # beat template
            logw = logw + Tb * la[t] + (1 - Tb) * l1a[t]                                # accumulate emission
            w = torch.softmax(logw, 0); phi_steps[t] = phi.cpu().numpy()
            if 1.0 / float((w * w).sum()) < K / 2:                                      # adaptive resample
                idx = torch.multinomial(w, K, replacement=True); res[t] = idx.cpu().numpy()
                phi = phi[idx]; lt = lt[idx]; logw = torch.zeros(K, device=dev)
            else:
                res[t] = np.arange(K)
            if t == T - 1:
                fw = torch.softmax(logw, 0).cpu().numpy(); fbpm = (60 * fps * m * torch.exp(lt) / TWO_PI).cpu().numpy()

    def trace(j):
        anc = np.empty(T, np.int64); anc[T - 1] = j
        for tt in range(T - 2, -1, -1): anc[tt] = res[tt][anc[tt + 1]]
        return np.array([phi_steps[tt, anc[tt]] for tt in range(T)])
    phi_map = trace(int(np.argmax(fw)))
    beats = beats_from_phase(phi_map, m, fps)
    dphi = np.diff(phi_map) % TWO_PI; dphi = dphi[(dphi > 1e-4) & (dphi < math.pi)]
    bpm_map = 60 * fps * m * float(np.median(dphi)) / TWO_PI if len(dphi) else 0.0
    return beats, bpm_map, fbpm, fw


def beats_from_phase(phi, m, fps, min_dist=0.10):
    """Geometric read-out: beats = the m subdivisions of the bar (wraps of m*phi)."""
    psi = (int(m) * np.asarray(phi, float)) % TWO_PI
    w = np.where(np.diff(psi) < -math.pi)[0] + 1
    out, last = [], -1e9
    for fr in w:
        if fr - last >= min_dist * fps: out.append(fr); last = fr
    return np.array(out, float) / fps


def free_run_phase(T, m, fps, bpm=None, seed=0):
    """DEPLOY BASELINE: roll the fixed prior open-loop with NO observation (the document's free-run)."""
    rng = np.random.RandomState(seed)
    if bpm is None: bpm = rng.uniform(60, 180)
    lt = math.log(TWO_PI * bpm / 60 / m / fps); phi = rng.uniform(0, TWO_PI); out = np.empty(T, np.float32)
    for t in range(T):
        lt = lt + 0.02 * rng.randn(); phi = (phi + math.exp(lt)) % TWO_PI; out[t] = phi
    return out


# ===================== oracle / synthetic activation (known answer) =====================
def oracle_activation(bpm_fn, dur, fps, sigma=1.5):
    """Build a perfect activation from a (possibly time-varying) tempo. bpm_fn(t)->bpm. Returns
    (activation[T], true_beat_times). Used for the oracle stress-tests (known-answer verification)."""
    if not callable(bpm_fn):
        b = float(bpm_fn); bpm_fn = lambda t: b
    t, bts = 0.0, []
    while t < dur: bts.append(t); t += 60.0 / bpm_fn(t)
    bts = np.array(bts); T = int(dur * fps); a = np.zeros(T, np.float32)
    a[np.clip(np.round(bts * fps).astype(int), 0, T - 1)] = 1.0
    k = int(4 * sigma); xs = np.arange(-k, k + 1); g = np.exp(-0.5 * (xs / sigma) ** 2); g /= g.sum()
    a = np.convolve(a, g, "same"); a = a / (a.max() + 1e-9) * 0.95 + 0.02
    return np.clip(a, 0.02, 0.97), bts[bts < T / fps]


# ===================== cached Beat-This activations + GT =====================
def load_easy(i):
    """i-th easy val song: returns (beat_activation[T], beat_times_sec, fps)."""
    f = sorted(glob.glob(f"{_EASY}/song_*.pt"))[i]
    d = torch.load(f, map_location="cpu"); a2 = d["act2"][:, 0]
    a = torch.sigmoid(a2).numpy() if a2.abs().max() > 1 else a2.numpy()
    bt = d["beat_targets"].numpy(); ref = np.where(bt > 0.5)[0] / FPS_EASY
    return a, ref, FPS_EASY


def n_easy():
    return len(glob.glob(f"{_EASY}/song_*.pt"))


def smc_ids():
    return [os.path.splitext(os.path.basename(f))[0] for f in sorted(glob.glob(f"{_SMC_ACT}/smc_*.npz"))]


def load_smc(tid):
    """SMC-MIREX track tid (e.g. 'smc_001'): returns (beat_activation[T], beat_times_sec, fps) or None."""
    g = glob.glob(f"{_SMC_ANN}/{tid.upper()}_*.txt")
    if not g: return None
    ref = np.loadtxt(g[0]).astype(float); ref = ref[:, 0] if ref.ndim > 1 else ref; ref = np.sort(ref[np.isfinite(ref)])
    a = np.load(f"{_SMC_ACT}/{tid}.npz")["beat"]
    return a, ref, FPS_SMC
