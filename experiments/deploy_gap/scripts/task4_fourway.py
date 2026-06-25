"""TASK 4 — FOUR-WAY F MEASUREMENT on a trained checkpoint.

Discriminates a genuine train/deploy gap from a model that never fit at all:
  (i)   TF-posterior DECODER-F  : decode posterior-rolled z, peak-pick      (what ELBO recon optimizes)
  (i')  TF-posterior LATENT-F   : subdivision read-out of posterior phase
  (ii)  free-run DECODER-F      : decode free-run prior z, peak-pick
  (iii) free-run LATENT-F       : subdivision read-out of free-run phase_mu
  (iv)  TF-PRIOR LATENT-F       : prior one-step mean seeded by posterior-mean previous latent

Reading: if (i)/(i') high but (iii) low -> genuine train/deploy gap. If (i) also ~0 -> the
model never fit the data; "posterior collapse" is a misnomer and the decoder/likelihood is the
problem, not the latent. Usage: task4_fourway.py <checkpoint.pt>
"""
import sys, math
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "/home/sogang/jaehoon/CHART")
from faithful.model import BarPointerVAE
from faithful.elbo import free_run
from faithful.data import FPS, N_MELS, LogMel, iter_val_songs
from faithful.distributions import TWO_PI
from faithful.evaluate import beats_from_barphase, beats_from_activation, f_measure

dev = "cuda"
ROOT = "/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data"
DS = ["ballroom", "beatles", "hains", "rwc_popular"]


@torch.no_grad()
def tf_posterior(model, h, b):
    """Teacher-forced posterior chain (means). Returns phi[T], log_tempo[T], decoder_prob[T]."""
    B, T, _ = h.shape
    pc = model.encode_prior(h); qc = model.encode_posterior(h, b)
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([qc[:, 0], z0], -1)))
    phi = qpm % TWO_PI; lt = qtm; meter = F.softmax(qm, -1)
    phis = [phi]; lts = [lt]; zf = [model.z_features(meter, phi, lt)]
    for t in range(1, T):
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(
            torch.cat([qc[:, t], model.z_features(meter, phi, lt)], -1)))
        phi = qpm % TWO_PI; lt = qtm; meter = F.softmax(qm, -1)
        phis.append(phi); lts.append(lt); zf.append(model.z_features(meter, phi, lt))
    dec = torch.sigmoid(torch.stack([model.decode(zf[t], pc[:, t]) for t in range(T)], 1))
    return (torch.stack(phis, 1)[0].cpu().numpy(), torch.stack(lts, 1)[0].cpu().numpy(),
            dec[0].cpu().numpy())


def main():
    ck = torch.load(sys.argv[1], map_location=dev)
    a = ck.get("args", {})
    model = BarPointerVAE(h_dim=N_MELS, hidden=a.get("hidden", 64),
                          num_meters=a.get("num_meters", 4),
                          latent_only=a.get("latent_only", False)).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    logmel = LogMel().to(dev)
    R = {k: [] for k in ("tf_post_dec", "tf_post_lat", "fr_dec", "fr_lat", "tf_prior_lat")}
    for key, audio, beats, downs, meta in iter_val_songs(ROOT, DS, max_per_dataset=4):
        T = min(len(beats), 1500)
        ref = np.where(beats.numpy()[:T] > 0.5)[0] / FPS
        df = np.where(downs.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) < 8:
            continue
        m = 4
        if len(df) >= 2:
            bpb = np.median([np.sum((ref >= df[i]) & (ref < df[i+1])) for i in range(len(df)-1)])
            m = max(2, min(int(round(bpb)) if bpb > 0 else 4, 4))
        h = logmel(audio.to(dev).unsqueeze(0))[:, :T]
        b = beats[:T].to(dev).unsqueeze(0).float()
        phi_p, lt_p, dec_p = tf_posterior(model, h, b)
        out = free_run(model, h, temperature=0.3)
        fr_phi = out["phase_mu"][0, :T].cpu().numpy()
        fr_dec = out["decoder_prob"][0, :T].cpu().numpy()
        # (iv) prior one-step mean seeded by posterior-mean previous latent
        prior_phi = np.empty_like(phi_p)
        prior_phi[0] = phi_p[0]
        prior_phi[1:] = (phi_p[:-1] + np.exp(lt_p[:-1])) % TWO_PI
        R["tf_post_dec"].append(f_measure(ref, beats_from_activation(dec_p, FPS)))
        R["tf_post_lat"].append(f_measure(ref, beats_from_barphase(phi_p, m, FPS)))
        R["fr_dec"].append(f_measure(ref, beats_from_activation(fr_dec, FPS)))
        R["fr_lat"].append(f_measure(ref, beats_from_barphase(fr_phi, m, FPS)))
        R["tf_prior_lat"].append(f_measure(ref, beats_from_barphase(prior_phi, m, FPS)))
    print(f"checkpoint: {sys.argv[1]}  songs={len(R['fr_lat'])}")
    for k in ("tf_post_dec", "tf_post_lat", "tf_prior_lat", "fr_dec", "fr_lat"):
        print(f"  {k:14s} = {np.nanmean(R[k]):.3f}")


if __name__ == "__main__":
    main()
