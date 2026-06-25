"""BLIND-PANEL DIAGNOSTIC: does the whole deployment failure reduce to "estimate ONE global tempo"?
Free-run the MEAN chain with a SINGLE frozen tempo = the song's GT global tempo (audio-blind constant),
and the correct initial phase. If F jumps to the 0.67-0.83 band, deployment = "pick one tempo scalar
+ start phase", not continuous tracking. Pair with phase alignment (panel caveat: a constant-tempo chain
with a phase offset scores ~0 even at the right tempo). Also split by tempo stability.
Faithful (audio-blind constant), eval-only. Usage: tempo_const_test.py <ckpt.pt>
"""
import sys, math
import numpy as np, torch
import mir_eval
sys.path.insert(0, "/home/sogang/jaehoon/CHART")
from faithful.model import BarPointerVAE
from faithful.elbo import free_run
from faithful.data import FPS, N_MELS, LogMel, iter_val_songs
from faithful.distributions import TWO_PI
from faithful.evaluate import beats_from_barphase

dev = "cuda"
ROOT = "/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data"
DS = ["ballroom", "beatles", "hains", "rwc_popular"]


def f1(ref, est):
    if len(ref) == 0: return float("nan")
    if len(est) == 0: return 0.0
    return float(mir_eval.beat.f_measure(ref, est))


def const_tempo_chain(T, phi0, lt_const):
    """audio-blind constant-tempo metronome: phi advances by exp(lt_const) each frame."""
    phi = np.empty(T); phi[0] = phi0 % TWO_PI
    step = math.exp(lt_const)
    for t in range(1, T):
        phi[t] = (phi[t-1] + step) % TWO_PI
    return phi


def best_offset_f1(ref, m, T, lt_const):
    """try a few initial phases (the panel's phase-offset alignment) and take the best."""
    best = 0.0
    for k in range(m):                       # start at each beat subdivision
        phi = const_tempo_chain(T, TWO_PI * k / m, lt_const)
        best = max(best, f1(ref, beats_from_barphase(phi, m, FPS)))
    return best


def main():
    ck = torch.load(sys.argv[1], map_location=dev); a = ck.get("args", {})
    model = BarPointerVAE(h_dim=N_MELS, hidden=a.get("hidden", 64), num_meters=a.get("num_meters", 4),
                          latent_only=a.get("latent_only", False)).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    logmel = LogMel().to(dev)
    rows = []
    for key, audio, beats, downs, meta in iter_val_songs(ROOT, DS, max_per_dataset=4):
        T = min(len(beats), 1200)
        ref = np.where(beats.numpy()[:T] > 0.5)[0] / FPS
        df = np.where(downs.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) < 8: continue
        m = 4
        if len(df) >= 2:
            bpb = np.median([np.sum((ref >= df[i]) & (ref < df[i+1])) for i in range(len(df)-1)])
            m = max(2, min(int(round(bpb)) if bpb > 0 else 4, 4))
        ibi = np.diff(ref)                                   # beat periods (s)
        beat_period_frames = float(np.median(ibi)) * FPS
        lt_const = math.log(TWO_PI / (m * beat_period_frames))   # bar-advance rad/frame from GT global tempo
        tempo_cv = float(np.std(ibi) / (np.mean(ibi) + 1e-9))     # tempo instability
        # model open-loop baseline (faithful free-run)
        h = logmel(audio.to(dev).unsqueeze(0))[:, :T]
        frp = free_run(model, h, temperature=0.3)["phase_mu"][0, :T].cpu().numpy()
        f_model = f1(ref, beats_from_barphase(frp, m, FPS))
        f_const = best_offset_f1(ref, m, T, lt_const)            # GT-constant-tempo metronome (best phase)
        rows.append((key, f_model, f_const, tempo_cv))
    fm = np.array([r[1] for r in rows]); fc = np.array([r[2] for r in rows]); cv = np.array([r[3] for r in rows])
    stable = cv < 0.04                                            # near-constant-tempo songs
    print(f"songs={len(rows)}")
    print(f"  model free-run (open-loop)          mean F1 = {np.nanmean(fm):.3f}")
    print(f"  GT-CONSTANT-tempo metronome (best phase) F1 = {np.nanmean(fc):.3f}   <-- the diagnostic")
    print(f"  GT-const on NEAR-CONSTANT-tempo songs F1 = {np.nanmean(fc[stable]):.3f}  (n={int(stable.sum())})")
    print(f"  GT-const on tempo-VARYING songs       F1 = {np.nanmean(fc[~stable]):.3f}  (n={int((~stable).sum())})")
    print("\nINTERPRETATION: if GT-const F1 is high (0.6-0.8) -> deployment ~= 'estimate ONE global tempo'.")
    print("If GT-const is also low -> real tempo VARIES and continuous tracking is genuinely required.")


if __name__ == "__main__":
    main()
