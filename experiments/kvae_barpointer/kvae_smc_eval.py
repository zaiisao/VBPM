"""SMC-MIREX OOD eval of the KVAE e2e model (the cleanest OOD test: our own log-mel TCN, no frozen-
feature fps mismatch). SMC audio -> log-mel(86fps) -> TCN -> Kalman FILTER -> head -> peak-pick beats.
SMC-MIREX is beats-only. ref: Beat-This noDBN 0.626 / +DBN 0.575 / madmom 0.570 (beat-F).
This is the line we can defend: a filter's temporal structure should help where peak-pick degrades.
"""
import sys, glob, os, argparse, importlib.util
import numpy as np
import torch, torchaudio, mir_eval

ROOT = "/home/sogang/jaehoon/CHART"
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/third_party/kalman-vae")
def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
kvae_run = _load("kr", f"{ROOT}/experiments/kvae_barpointer/kvae_run.py")
e2e = _load("ee", f"{ROOT}/experiments/diagram_arch/e2e.py")
KVAEBarPointer = kvae_run.KVAEBarPointer; da = kvae_run.da
from kvae.sample_control import SampleControl
from faithful.data import LogMel, N_MELS
DEV = kvae_run.DEV
ANN = "/home/sogang/jaehoon/Analyze-SMC/smc_metadata/annotations"
SMC_AUDIO = "/home/sogang/jaehoon/Analyze-SMC/SMC_MIREX/SMC_MIREX_Audio"
SR = 22050


def gt_beats(num):
    g = glob.glob(f"{ANN}/SMC_{num}_*.txt")
    if not g: return None
    a = np.loadtxt(g[0]); a = a[:, 0] if a.ndim > 1 else a
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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/kvae_barpointer/m_e2e.pt")
    ap.add_argument("--n", type=int, default=217); ap.add_argument("--max_frames", type=int, default=3600)
    a = ap.parse_args()
    d = torch.load(a.ckpt, map_location=DEV); ar = d["args"]
    tcn = e2e.TCNFrontend(N_MELS, ar["ch"]).to(DEV); tcn.load_state_dict(d["tcn"]); tcn.eval()
    model = KVAEBarPointer(h_dim=ar["ch"], a_dim=ar["a_dim"], z_dim=ar["z_dim"], K=ar["K"]).to(DEV)
    model.load_state_dict(d["model"]); model.eval()
    lm = LogMel().to(DEV); fps = SR / 256.0
    sc = SampleControl(encoder="mean", decoder="mean", state_transition="mean", observation="mean")
    Fb, A = [], []
    print("==== SMC-MIREX OOD (KVAE e2e; ref: BeatThis noDBN 0.626 / +DBN 0.575 / madmom 0.570) ====", flush=True)
    for wav in sorted(glob.glob(f"{SMC_AUDIO}/*.wav"))[:a.n]:
        num = os.path.basename(wav).split("_")[1].split(".")[0]
        ref = gt_beats(num)
        if ref is None: continue
        wave, sr = torchaudio.load(wav); wave = wave.mean(0)
        if sr != SR: wave = torchaudio.functional.resample(wave, sr, SR)
        logmel = lm(wave.unsqueeze(0).to(DEV)); T = min(logmel.shape[1], a.max_frames)
        h = tcn(logmel[:, :T])                                   # [1,T,ch]
        av = model.encoder(h.reshape(-1, ar["ch"])).mean.view(T, 1, model.a_dim)
        fm, *_ = model.ssm.kalman_filter(av, sample_control=sc)
        prob = torch.sigmoid(model.head(fm.view(T, model.z_dim)))[:, 0].cpu().numpy()
        est = peaks_fps(prob, fps)
        Fb.append(da.fmeas(ref, est)); A.append(amlt(ref, est))
    m = lambda x: float(np.nanmean(x)) if x else float("nan")
    print(f"KVAE e2e (SMC audio->log-mel->TCN->filter->head), n={len(Fb)}: beat-F={m(Fb):.3f}  AMLt={m(A):.3f}", flush=True)
    print("SMC_OOD_DONE", flush=True)


if __name__ == "__main__":
    main()
