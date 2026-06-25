"""TASK 2 — OVERFIT ONE SONG (expressivity / sufficiency test).

Train the faithful model on a SINGLE song to convergence; measure teacher-forced posterior
beat-F two ways (latent subdivision read-out AND decoder read-out). If the model cannot
reach ~1.0 teacher-forced on ONE song, the architecture cannot express the answer and no
data/regularization/optimization story matters.

Binary choices (run BOTH per the user directive): lr in {1e-3, 1e-2}.
"""
import sys, math, argparse
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "/home/sogang/jaehoon/CHART")
from faithful.model import BarPointerVAE
from faithful.elbo import strict_elbo
from faithful.data import FPS, N_MELS, LogMel, iter_val_songs
from faithful.distributions import TWO_PI
from faithful.evaluate import beats_from_barphase, beats_from_activation, f_measure

ROOT = "/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data"
DS = ["ballroom", "beatles", "hains", "rwc_popular"]
dev = "cuda"


@torch.no_grad()
def teacher_forced(model, h, b, pc, qc):
    """Posterior rollout using posterior means as the previous-latent chain (teacher-forced)."""
    B, T, _ = h.shape
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([qc[:, 0], z0], -1)))
    phi = qpm % TWO_PI; lt = qtm; meter = F.softmax(qm, -1)
    traj = [phi]; zf = [model.z_features(meter, phi, lt)]
    for t in range(1, T):
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(
            torch.cat([qc[:, t], model.z_features(meter, phi, lt)], -1)))
        phi = qpm % TWO_PI; lt = qtm; meter = F.softmax(qm, -1)
        traj.append(phi); zf.append(model.z_features(meter, phi, lt))
    dec = torch.sigmoid(torch.stack([model.decode(zf[t], pc[:, t]) for t in range(T)], 1))
    return torch.stack(traj, 1)[0].cpu().numpy(), dec[0].cpu().numpy()


def run(lr, song, steps=400):
    key, audio, beats, downs, meta = song
    T = min(len(beats), 250)
    logmel = LogMel().to(dev)
    h = logmel(audio.to(dev).unsqueeze(0))[:, :T]
    b = beats[:T].to(dev).unsqueeze(0).float()
    ref = np.where(beats.numpy()[:T] > 0.5)[0] / FPS
    df = np.where(downs.numpy()[:T] > 0.5)[0] / FPS
    m = 4
    if len(df) >= 2:
        bpb = np.median([np.sum((ref >= df[i]) & (ref < df[i + 1])) for i in range(len(df) - 1)])
        m = max(2, min(int(round(bpb)) if bpb > 0 else 4, 4))
    torch.manual_seed(0)
    model = BarPointerVAE(h_dim=N_MELS, hidden=64, num_meters=4).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist = []
    for s in range(1, steps + 1):
        opt.zero_grad()
        temp = max(0.3, 1.0 - 0.7 * s / steps)
        loss, info = strict_elbo(model, h, b, temperature=temp)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if s % 100 == 0 or s == 1:
            model.eval()
            pc = model.encode_prior(h); qc = model.encode_posterior(h, b)
            phi, dec = teacher_forced(model, h, b, pc, qc)
            bf_lat = f_measure(ref, beats_from_barphase(phi, m, FPS))
            bf_dec = f_measure(ref, beats_from_activation(dec, FPS))
            hist.append((s, info["recon"], info["kl_phase"], info["kl_tempo"],
                         bf_lat, bf_dec, float(dec.max()), float(dec.mean())))
            print(f"  [lr={lr}] s{s} recon={info['recon']:.1f} klp={info['kl_phase']:.2f} "
                  f"klt={info['kl_tempo']:.2f} TF-beatF lat={bf_lat:.3f} dec={bf_dec:.3f} "
                  f"decmax={dec.max():.2f} decmean={dec.mean():.3f}", flush=True)
            model.train()
    return {"key": key, "lr": lr, "m": m, "n_ref": int(len(ref)), "T": T, "hist": hist,
            "final": hist[-1]}


if __name__ == "__main__":
    songs = list(iter_val_songs(ROOT, DS, max_per_dataset=1))
    song = None
    for s in songs:
        if int((s[2] > 0.5).sum()) >= 30:    # enough beats
            song = s; break
    print(f"SONG = {song[0]}  beats={int((song[2]>0.5).sum())}")
    for lr in (1e-3, 1e-2):
        print(f"=== OVERFIT lr={lr} ===")
        r = run(lr, song)
        f = r["final"]
        print(f"FINAL lr={lr}: TF-beatF latent={f[4]:.3f} decoder={f[5]:.3f} (n_ref={r['n_ref']}, m={r['m']})")
