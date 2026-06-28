"""END-TO-END test of the diagram architecture (the contribution: train the whole stack jointly).

Two frontends, SAME bar-pointer VAE (b-dropout encoder, latent-only decoder, hybrid prior):
  * PRETRAINED-FROZEN: cached Beat-This [T,512] features (experiments/diagram_arch/run.py).
  * FROM-SCRATCH (this file): a learnable TCN over log-mel from raw audio, randomly initialized,
    trained JOINTLY with the VAE end-to-end.

Data: the four WaveBeat datasets (ballroom, beatles, hainsworth, rwc_popular), parsed by WaveBeat's
DownbeatDataset (reused via faithful.data -- correct for all four annotation formats).

Deploy = encoder(h, b=0) -> z -> latent-only decoder -> beats. Leak controls (shuffled / zero audio)
confirm the score genuinely depends on the audio.
"""
import sys, math, random, argparse, importlib.util
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/CHART")
_spec = importlib.util.spec_from_file_location("da", "/home/sogang/jaehoon/CHART/experiments/diagram_arch/run.py")
da = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(da)
BPVAE, rollout, elbo_loss = da.BPVAE, da.rollout, da.elbo_loss
peaks, fmeas, phase_beats, phase_downbeats = da.peaks, da.fmeas, da.phase_beats, da.phase_downbeats
DEV = da.DEV

from faithful.data import build_train_loader, iter_val_songs, LogMel, FPS, N_MELS

KEYS = ["ballroom", "beatles", "hains", "rwc_popular"]


class TCNFrontend(nn.Module):
    """Small dilated-conv beat frontend over log-mel. Random init -> trained end-to-end."""
    def __init__(self, n_mels=128, ch=128, n_layers=6, p=0.1):
        super().__init__()
        self.inp = nn.Conv1d(n_mels, ch, 1)
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            d = 2 ** i
            self.blocks.append(nn.Sequential(
                nn.Conv1d(ch, ch, 3, padding=d, dilation=d), nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(p),
                nn.Conv1d(ch, ch, 3, padding=d, dilation=d), nn.BatchNorm1d(ch), nn.ReLU()))

    def forward(self, logmel):                              # [B, T, n_mels]
        x = self.inp(logmel.transpose(1, 2))
        for b in self.blocks:
            x = x + b(x)
        return x.transpose(1, 2)                            # [B, T, ch]


def _align(h, b, db):
    T = min(h.shape[1], b.shape[1])
    return h[:, :T], b[:, :T], db[:, :T]


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


@torch.no_grad()
def evaluate(tcn, vae, songs, give_beats, h_mode="real", max_frames=1200):
    tcn.eval(); vae.eval(); db_b, db_d, ph_b, ph_d = [], [], [], []
    n = len(songs)
    for i, (lm, b, db) in enumerate(songs):
        lm_use = songs[(i + 1) % n][0] if h_mode == "shuffle" else lm
        T = min(lm_use.shape[0], b.shape[0], max_frames)
        lm_in = (torch.zeros(1, T, N_MELS, device=DEV) if h_mode == "zero" else lm_use[:T].unsqueeze(0).to(DEV))
        h = tcn(lm_in)
        bi = b[:T].unsqueeze(0).to(DEV) if give_beats else torch.zeros(1, T, device=DEV)
        di = db[:T].unsqueeze(0).to(DEV) if give_beats else torch.zeros(1, T, device=DEV)
        _, phase_mu, logits = rollout(vae, h, bi, di, sample=False, compute_kl=False)
        prob = torch.sigmoid(logits)[0].cpu().numpy(); pm = phase_mu[0].cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS; dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2:
            db_b.append(fmeas(ref, peaks(prob[:, 0]))); ph_b.append(fmeas(ref, phase_beats(pm, 4)))
        if len(dref) >= 2:
            db_d.append(fmeas(dref, peaks(prob[:, 1], min_dist=0.30))); ph_d.append(fmeas(dref, phase_downbeats(pm)))
    tcn.train(); vae.train()
    m = lambda x: float(np.nanmean(x)) if x else float("nan")
    return {"dec_beat": m(db_b), "dec_db": m(db_d), "phase_beat": m(ph_b), "phase_db": m(ph_d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data")
    ap.add_argument("--val_per_ds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval_every", type=int, default=750)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--examples_per_epoch", type=int, default=2000)
    ap.add_argument("--pw_b", type=float, default=8.0)
    ap.add_argument("--pw_db", type=float, default=20.0)
    ap.add_argument("--fb", type=float, default=0.1)
    ap.add_argument("--b_drop", type=float, default=0.5)
    ap.add_argument("--save", default="", help="if set, save {tcn, vae, ch} to this path")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

    logmel = LogMel().to(DEV)
    print(f"[e2e] 4 datasets {KEYS} | FROM-SCRATCH TCN(log-mel)+VAE end-to-end | ch={a.ch}", flush=True)
    print(f"[e2e] preloading val ({a.val_per_ds}/dataset) ...", flush=True)
    val = []
    for k, audio, beats, downs, meta in iter_val_songs(a.root, KEYS, max_per_dataset=a.val_per_ds):
        with torch.no_grad():
            lm = logmel(audio.unsqueeze(0).to(DEV))[0].cpu()
        T = min(lm.shape[0], beats.shape[0]); val.append((lm[:T], beats[:T], downs[:T]))
    print(f"[e2e] val songs = {len(val)} | building train loader (all train songs, 4 datasets) ...", flush=True)
    dl = cycle(build_train_loader(a.root, KEYS, a.frames, a.bs, examples_per_epoch=a.examples_per_epoch, num_workers=4))

    tcn = TCNFrontend(N_MELS, a.ch).to(DEV); vae = BPVAE(h_dim=a.ch, hidden=64).to(DEV)
    opt = torch.optim.Adam(list(tcn.parameters()) + list(vae.parameters()), lr=a.lr)
    npar = sum(p.numel() for p in tcn.parameters()) + sum(p.numel() for p in vae.parameters())
    print(f"[e2e] trainable params = {npar:,}", flush=True)

    for step in range(1, a.steps + 1):
        temp = 1.0 + (0.3 - 1.0) * min(step / a.steps, 1.0)
        audio, b, db = next(dl)
        h = logmel(audio.to(DEV)); h, b, db = _align(h, b.to(DEV), db.to(DEV))
        h = tcn(h)
        loss, info = elbo_loss(vae, h, b, db, temp, a.pw_b, a.pw_db, a.fb, a.b_drop)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(tcn.parameters()) + list(vae.parameters()), 5.0); opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            ho = evaluate(tcn, vae, val, give_beats=False)
            tf = evaluate(tcn, vae, val, give_beats=True)
            print(f"\n[e2e] step {step} | recon {info['recon']:.1f} | KL m/phi/tau "
                  f"{info['klm']:.2f}/{info['klp']:.2f}/{info['klt']:.2f}", flush=True)
            print(f"  H-ONLY (deploy): decoder beat {ho['dec_beat']:.3f} db {ho['dec_db']:.3f} | "
                  f"phase beat {ho['phase_beat']:.3f} db {ho['phase_db']:.3f}", flush=True)
            print(f"  TEACHER-FORCED : decoder beat {tf['dec_beat']:.3f} db {tf['dec_db']:.3f}", flush=True)

    real = evaluate(tcn, vae, val, give_beats=False, h_mode="real")
    shuf = evaluate(tcn, vae, val, give_beats=False, h_mode="shuffle")
    zero = evaluate(tcn, vae, val, give_beats=False, h_mode="zero")
    print("\n[e2e] --- LEAK CONTROLS (h-only; decoder beat / db) ---")
    print(f"  real audio     : beat {real['dec_beat']:.3f}  db {real['dec_db']:.3f}")
    print(f"  shuffled audio : beat {shuf['dec_beat']:.3f}  db {shuf['dec_db']:.3f}   (must collapse)")
    print(f"  zero audio     : beat {zero['dec_beat']:.3f}  db {zero['dec_db']:.3f}   (must collapse)")
    if a.save:
        import os as _os
        _os.makedirs(_os.path.dirname(a.save) or ".", exist_ok=True)
        torch.save({"tcn": tcn.state_dict(), "vae": vae.state_dict(), "ch": a.ch}, a.save)
        print(f"\n[e2e][saved] TCN+VAE -> {a.save}")

    print("\n[e2e] === FROM-SCRATCH end-to-end (TCN+VAE, random init, 4 datasets) ===")
    print("ref: pretrained-frozen Beat-This+VAE ~0.53-0.70 beat / ~0.85 db | faithful log-mel free-run ~0")


if __name__ == "__main__":
    main()
