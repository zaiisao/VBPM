"""EXPERIMENT 0.1 — Oracle-state injection: split phase-drift vs tempo-drift.

During the free-run MEAN chain, every K seconds overwrite the pointer state from GROUND TRUTH,
then keep free-running with the (audio-blind) prior dynamics. Arms:
  A = overwrite phase only;  B = overwrite phase AND tempo.
Guards: K=inf must reproduce open-loop fr_lat (~0.018); K=every-frame must hit the one-step
ceiling (~0.48). The decisive read is the coarse K (~4s), arm A vs B:
  arm A recovers  -> failure is PURE phase-mean drift (a phase-only closed loop suffices).
  needs arm B     -> tempo/octave drift matters too (tempo correction mandatory).
  neither         -> read-out/decoder fault.
Eval-only, faithful. Usage: oracle_inject.py <ckpt.pt>
"""
import sys, math
import numpy as np, torch
import torch.nn.functional as F
import mir_eval
sys.path.insert(0, "/home/sogang/jaehoon/CHART")
from faithful.model import BarPointerVAE
from faithful.data import FPS, N_MELS, LogMel, iter_val_songs
from faithful.distributions import TWO_PI, gumbel_softmax
from faithful.evaluate import beats_from_barphase, downbeats_from_barphase, f_measure

dev = "cuda"
ROOT = "/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data"
DS = ["ballroom", "beatles", "hains", "rwc_popular"]


def prf(ref, est):
    if len(ref) == 0: return (np.nan, np.nan, np.nan)
    if len(est) == 0: return (np.nan, 0.0, 0.0)
    m = mir_eval.util.match_events(ref, est, 0.07)
    p, r = len(m)/len(est), len(m)/len(ref)
    return (p, r, 2*p*r/(p+r) if (p+r) else 0.0)


def ideal_global_phase(bf, df, m, T):
    """GLOBAL (monotone, non-mod) ideal bar phase through every beat; + per-frame ideal log-tempo."""
    downs = np.sort(df); beats = np.sort(bf)
    at, ap = [], []
    for b in beats:
        bar = int(np.searchsorted(downs, b, side="right") - 1)
        if bar < 0: continue
        nb = downs[min(bar+1, len(downs)-1)] - downs[bar]
        k = int(round((b - downs[bar]) / max(1e-9, nb) * m)) if len(downs) > bar+1 else 0
        at.append(b); ap.append(TWO_PI*bar + TWO_PI*(k % m)/m)
    for i, d in enumerate(downs):
        at.append(d); ap.append(TWO_PI*i)
    o = np.argsort(at); at = np.asarray(at)[o]; ap = np.asarray(ap)[o]
    kt, kp = [], []
    for t, p in zip(at, ap):
        if kt and abs(t-kt[-1]) < 1e-6: continue
        if kp and p < kp[-1]: p = kp[-1]
        kt.append(t); kp.append(p)
    Phi = np.interp(np.arange(T)/FPS, kt, kp, left=kp[0], right=kp[-1])      # global phase
    dPhi = np.diff(Phi, prepend=Phi[0]); dPhi = np.clip(dPhi, 1e-4, None)
    lt_ideal = np.log(dPhi)                                                  # per-frame ideal log-tempo
    return Phi, lt_ideal


@torch.no_grad()
def free_run_inject(model, h, Phi_ideal, lt_ideal, Kf, arm):
    """Replicate the free_run MEAN chain; every Kf frames overwrite phi (arm A) or phi+lt (arm B)."""
    B, T, _ = h.shape
    pc = model.encode_prior(h)
    p_m, p_phi_mu, p_phi_k, p_tau_mu, p_tau_s = model.unpack(model.prior_init_head(pc.mean(1)))
    phi_mu = float(p_phi_mu[0] % TWO_PI); lt_mu = float(p_tau_mu[0])
    out = [phi_mu]
    for t in range(1, T):
        phi_mu = (phi_mu + math.exp(lt_mu)) % TWO_PI          # constant-tempo mean chain (faithful free_run)
        if Kf > 0 and (t % Kf == 0):                          # ORACLE injection
            phi_mu = float(Phi_ideal[t] % TWO_PI)
            if arm == "B":
                lt_mu = float(lt_ideal[t])
        out.append(phi_mu)
    return np.array(out)


def main():
    ck = torch.load(sys.argv[1], map_location=dev); a = ck.get("args", {})
    model = BarPointerVAE(h_dim=N_MELS, hidden=a.get("hidden", 64), num_meters=a.get("num_meters", 4),
                          latent_only=a.get("latent_only", False)).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    logmel = LogMel().to(dev)
    K4 = int(round(4.0 * FPS))                                # ~4s injection period
    configs = [("Kinf(open-loop)", 10**9, "A"), ("Kevery(ceiling)", 1, "A"),
               ("K4s armA(phase)", K4, "A"), ("K4s armB(phase+tempo)", K4, "B"),
               ("K2s armA", int(round(2*FPS)), "A"), ("K2s armB", int(round(2*FPS)), "B")]
    acc = {c[0]: {"beat": [], "db": []} for c in configs}
    for key, audio, beats, downs, meta in iter_val_songs(ROOT, DS, max_per_dataset=4):
        T = min(len(beats), 1200)
        bf = np.where(beats.numpy()[:T] > 0.5)[0] / FPS
        df = np.where(downs.numpy()[:T] > 0.5)[0] / FPS
        if len(bf) < 8 or len(df) < 3: continue
        bpb = np.median([np.sum((bf >= df[i]) & (bf < df[i+1])) for i in range(len(df)-1)])
        m = max(2, min(int(round(bpb)) if bpb > 0 else 4, 4))
        h = logmel(audio.to(dev).unsqueeze(0))[:, :T]
        Phi, lt_ideal = ideal_global_phase(bf, df, m, T)
        for name, Kf, arm in configs:
            pm = free_run_inject(model, h, Phi, lt_ideal, Kf, arm)
            acc[name]["beat"].append(prf(bf, beats_from_barphase(pm, m, FPS)))
            acc[name]["db"].append(prf(df, downbeats_from_barphase(pm, FPS)))
    print(f"=== ORACLE INJECTION: {sys.argv[1].split('/')[-2]} ===  (beat P/R/F1 | downbeat F1, +/-70ms)")
    for name, _, _ in configs:
        b = np.array(acc[name]["beat"], float); d = np.array(acc[name]["db"], float)
        print(f"  {name:22s} beat P={np.nanmean(b[:,0]):.3f} R={np.nanmean(b[:,1]):.3f} "
              f"F1={np.nanmean(b[:,2]):.3f} | db_F1={np.nanmean(d[:,2]):.3f}")


if __name__ == "__main__":
    main()
