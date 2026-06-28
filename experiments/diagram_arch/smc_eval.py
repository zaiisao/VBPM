"""SMC-MIREX evaluation of the trained diagram models (hard cross-dataset generalization test).

  A (pretrained-frozen Beat-This + VAE): eval on cached SMC Beat-This features.
     CAVEAT: those features are 50 fps; the VAE was trained at 86 fps -> an fps/domain shift.
  B (from-scratch TCN + VAE): eval on SMC raw audio -> log-mel at 86 fps -> TCN -> VAE.
     CLEAN: same log-mel pipeline as training; only the audio differs (true generalization).

SMC-MIREX is beats-only (no downbeats). Reports beat-F and AMLt (octave/level-tolerant).
ref: Beat-This no-DBN 0.626 | Beat-This+DBN 0.575 | madmom 0.570  (all beat-F on SMC-MIREX).
"""
import sys, glob, os, math, argparse, importlib.util
import numpy as np
import torch, torchaudio
import mir_eval

sys.path.insert(0, "/home/sogang/jaehoon/CHART")
def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
da = _load("da", "/home/sogang/jaehoon/CHART/experiments/diagram_arch/run.py")
e2e = _load("e2e_mod", "/home/sogang/jaehoon/CHART/experiments/diagram_arch/e2e.py")
from faithful.data import LogMel, N_MELS
DEV = da.DEV

ANN = "/home/sogang/jaehoon/Analyze-SMC/smc_metadata/annotations"
SMC_FEAT = "/home/sogang/jaehoon/CHART/cache/acts/smc_rich_heldout"
SMC_AUDIO = "/home/sogang/jaehoon/Analyze-SMC/SMC_MIREX/SMC_MIREX_Audio"
SR = 22050


def gt_beats(num):
    g = glob.glob(f"{ANN}/SMC_{num}_*.txt")
    if not g: return None
    a = np.loadtxt(g[0]); a = a[:, 0] if (a.ndim > 1) else a
    a = np.sort(np.atleast_1d(a)[np.isfinite(np.atleast_1d(a))])
    return a if len(a) >= 2 else None


def peaks_fps(prob, fps, thr=0.5, min_dist=0.10):
    p = np.asarray(prob); md = int(min_dist * fps)
    cand = [t for t in range(1, len(p) - 1) if p[t] >= thr and p[t] >= p[t - 1] and p[t] >= p[t + 1]]
    out, last = [], -10 ** 9
    for t in cand:
        if t - last >= md: out.append(t); last = t
    return np.array(out, float) / fps


def amlt(ref, est):
    if len(ref) < 2 or len(est) < 2: return 0.0
    try: return float(mir_eval.beat.continuity(ref, est)[3])
    except Exception: return 0.0


@torch.no_grad()
def eval_A(ckpt, n):
    d = torch.load(ckpt, map_location=DEV)
    vae = da.BPVAE(h_dim=d["h_dim"], hidden=64).to(DEV); vae.load_state_dict(d["vae"]); vae.eval()
    Fb, Fp, A = [], [], []
    for f in sorted(glob.glob(f"{SMC_FEAT}/*.pt"))[:n]:
        dd = torch.load(f, map_location="cpu"); num = dd["tid"].split("_")[1]
        ref = gt_beats(num)
        if ref is None: continue
        h = dd["feat"].float().unsqueeze(0).to(DEV); T = h.shape[1]
        z0 = torch.zeros(1, T, device=DEV)
        _, pm, logits = da.rollout(vae, h, z0, z0, sample=False, compute_kl=False)
        prob = torch.sigmoid(logits)[0].cpu().numpy(); pmn = pm[0].cpu().numpy()
        est = peaks_fps(prob[:, 0], 50.0)
        phe = da.phase_beats(pmn, 4)              # phase_beats divides by da.FPS(86) -> approx; report as secondary
        Fb.append(da.fmeas(ref, est)); Fp.append(da.fmeas(ref, phe)); A.append(amlt(ref, est))
    m = lambda x: float(np.nanmean(x)) if x else float("nan")
    return m(Fb), m(Fp), m(A), len(Fb)


@torch.no_grad()
def eval_B(ckpt, n, max_frames=3600):
    d = torch.load(ckpt, map_location=DEV)
    tcn = e2e.TCNFrontend(N_MELS, d["ch"]).to(DEV); tcn.load_state_dict(d["tcn"]); tcn.eval()
    vae = da.BPVAE(h_dim=d["ch"], hidden=64).to(DEV); vae.load_state_dict(d["vae"]); vae.eval()
    lm = LogMel().to(DEV); fps = SR / 256.0
    Fb, Fp, A = [], [], []
    for wav in sorted(glob.glob(f"{SMC_AUDIO}/*.wav"))[:n]:
        num = os.path.basename(wav).split("_")[1].split(".")[0]
        ref = gt_beats(num)
        if ref is None: continue
        wave, sr = torchaudio.load(wav); wave = wave.mean(0)
        if sr != SR: wave = torchaudio.functional.resample(wave, sr, SR)
        logmel = lm(wave.unsqueeze(0).to(DEV)); T = min(logmel.shape[1], max_frames)
        h = tcn(logmel[:, :T]); z0 = torch.zeros(1, T, device=DEV)
        _, pm, logits = da.rollout(vae, h, z0, z0, sample=False, compute_kl=False)
        prob = torch.sigmoid(logits)[0].cpu().numpy(); pmn = pm[0].cpu().numpy()
        est = peaks_fps(prob[:, 0], fps)
        phe = peaks_fps(((4 * np.asarray(pmn)) % (2 * math.pi)), fps)  # not exact; decoder read-out is primary
        Fb.append(da.fmeas(ref, est)); Fp.append(da.fmeas(ref, da.phase_beats(pmn, 4))); A.append(amlt(ref, est))
    m = lambda x: float(np.nanmean(x)) if x else float("nan")
    return m(Fb), m(Fp), m(A), len(Fb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_A", default="checkpoints/diagram_A.pt")
    ap.add_argument("--ckpt_B", default="checkpoints/diagram_B.pt")
    ap.add_argument("--n", type=int, default=217)
    a = ap.parse_args()
    print("==== SMC-MIREX EVAL (beats-only; ref: Beat-This noDBN 0.626 / +DBN 0.575 / madmom 0.570) ====", flush=True)
    if os.path.exists(a.ckpt_A):
        fb, fp, am, n = eval_A(a.ckpt_A, a.n)
        print(f"A  pretrained-frozen Beat-This+VAE (50fps cached feats; FPS-SHIFT CAVEAT), n={n}:", flush=True)
        print(f"     beat-F(decoder)={fb:.3f}  AMLt={am:.3f}  | phase-readout(approx)={fp:.3f}", flush=True)
    else:
        print(f"A  ckpt missing: {a.ckpt_A}", flush=True)
    if os.path.exists(a.ckpt_B):
        fb, fp, am, n = eval_B(a.ckpt_B, a.n)
        print(f"B  from-scratch TCN+VAE (SMC audio->log-mel 86fps; CLEAN cross-dataset), n={n}:", flush=True)
        print(f"     beat-F(decoder)={fb:.3f}  AMLt={am:.3f}  | phase-readout={fp:.3f}", flush=True)
    else:
        print(f"B  ckpt missing: {a.ckpt_B}", flush=True)
    print("SMC_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
