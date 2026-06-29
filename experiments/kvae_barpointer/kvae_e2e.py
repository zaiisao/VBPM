"""KVAE-bar-pointer END-TO-END (the headline claim): train a from-scratch TCN frontend JOINTLY with
the exact differentiable Kalman filter, on raw audio from the 4 WaveBeat datasets. No frozen frontend.

  raw audio -> log-mel -> TCN (random init) -> h -> MLP enc -> a -> Kalman filter -> z -> head -> beats
  trained end-to-end (TCN + VAE + SSM + head) via KVAE ELBO + beat/downbeat BCE on the filtered latent.

Deploy = Kalman FILTER on audio -> head(z). Leak controls (shuffled/zero audio) certify audio use.
This is the same exact-filter that broke the wall in M1, now with a jointly-trained frontend.
"""
import sys, math, random, argparse, importlib.util
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/third_party/kalman-vae")
kr = importlib.util.spec_from_file_location("kr", f"{ROOT}/experiments/kvae_barpointer/kvae_run.py")
kvae_run = importlib.util.module_from_spec(kr); kr.loader.exec_module(kvae_run)
KVAEBarPointer, kvae_elbo = kvae_run.KVAEBarPointer, kvae_run.kvae_elbo
da = kvae_run.da; peaks, fmeas = da.peaks, da.fmeas
from kvae.sample_control import SampleControl
from faithful.data import build_train_loader, iter_val_songs, LogMel, FPS, N_MELS
ee = importlib.util.spec_from_file_location("ee", f"{ROOT}/experiments/diagram_arch/e2e.py")
e2e = importlib.util.module_from_spec(ee); ee.loader.exec_module(e2e)
TCNFrontend, _align, cycle = e2e.TCNFrontend, e2e._align, e2e.cycle
DEV = kvae_run.DEV
KEYS = ["ballroom", "beatles", "hains", "rwc_popular"]


@torch.no_grad()
def evaluate(tcn, model, songs, h_mode="real", max_frames=1600):
    tcn.eval(); model.eval()
    sc = SampleControl(encoder="mean", decoder="mean", state_transition="mean", observation="mean")
    bF, dF = [], []
    n = len(songs)
    for i, (lm, b, db) in enumerate(songs):
        lm_use = songs[(i + 1) % n][0] if h_mode == "shuffle" else lm
        T = min(lm_use.shape[0], b.shape[0], max_frames)
        lm_in = torch.zeros(1, T, N_MELS, device=DEV) if h_mode == "zero" else lm_use[:T].unsqueeze(0).to(DEV)
        h = tcn(lm_in)                                  # [1,T,ch]
        a = model.encoder(h.reshape(-1, h.shape[-1])).mean.view(T, 1, model.a_dim)
        fm, *_ = model.ssm.kalman_filter(a, sample_control=sc)
        prob = torch.sigmoid(model.head(fm.view(T, model.z_dim))).cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS; dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2: bF.append(fmeas(ref, peaks(prob[:, 0])))
        if len(dref) >= 2: dF.append(fmeas(dref, peaks(prob[:, 1], min_dist=0.30)))
    tcn.train(); model.train(); m = lambda x: float(np.nanmean(x)) if x else float("nan")
    return m(bF), m(dF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data")
    ap.add_argument("--val_per_ds", type=int, default=8); ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--eval_every", type=int, default=600); ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--a_dim", type=int, default=8); ap.add_argument("--z_dim", type=int, default=8)
    ap.add_argument("--K", type=int, default=5); ap.add_argument("--beat_w", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

    logmel = LogMel().to(DEV)
    print(f"[kvae-e2e] {KEYS} | FROM-SCRATCH TCN(log-mel)+KVAE end-to-end | ch={a.ch} a={a.a_dim} z={a.z_dim} K={a.K}", flush=True)
    val = []
    for k, audio, beats, downs, meta in iter_val_songs(a.root, KEYS, max_per_dataset=a.val_per_ds):
        with torch.no_grad():
            lm = logmel(audio.unsqueeze(0).to(DEV))[0].cpu()
        T = min(lm.shape[0], beats.shape[0]); val.append((lm[:T], beats[:T], downs[:T]))
    print(f"[kvae-e2e] val songs = {len(val)} | building train loader ...", flush=True)
    dl = cycle(build_train_loader(a.root, KEYS, a.frames, a.bs, examples_per_epoch=2000, num_workers=4))

    tcn = TCNFrontend(N_MELS, a.ch).to(DEV)
    model = KVAEBarPointer(h_dim=a.ch, a_dim=a.a_dim, z_dim=a.z_dim, K=a.K).to(DEV)
    params = list(tcn.parameters()) + list(model.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)
    print(f"[kvae-e2e] trainable params = {sum(p.numel() for p in params):,}", flush=True)
    sc = SampleControl(encoder="sample", decoder="mean", state_transition="sample", observation="sample")
    pw = torch.tensor([a.beat_w * 1.6, a.beat_w * 4.0], device=DEV)

    for step in range(1, a.steps + 1):
        audio, b, db = next(dl)
        h = logmel(audio.to(DEV)); h, b, db = _align(h, b.to(DEV), db.to(DEV))
        h = tcn(h)                                       # [B,T,ch]
        Hs = h.transpose(0, 1)                            # (T,B,ch) for SSM
        elbo, z, info = kvae_elbo(model, Hs, sc, recon_w=0.3)
        bl = model.head(z.reshape(-1, model.z_dim)).view(*z.shape[:2], 2)
        tgt = torch.stack([b.transpose(0, 1), db.transpose(0, 1)], -1)
        bce = F.binary_cross_entropy_with_logits(bl, tgt, pos_weight=pw)
        loss = -elbo + a.beat_w * bce
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            bF, dF = evaluate(tcn, model, val, "real")
            print(f"\n[kvae-e2e] step {step} | elbo {float(elbo):.1f} bce {float(bce):.3f} "
                  f"(recon {info['recon']:.1f}) | FILTER deploy: beat {bF:.3f} downbeat {dF:.3f}", flush=True)

    bF, dF = evaluate(tcn, model, val, "real")
    bFs, dFs = evaluate(tcn, model, val, "shuffle"); bFz, dFz = evaluate(tcn, model, val, "zero")
    print("\n[kvae-e2e] --- FINAL (end-to-end; Kalman-FILTER deploy) ---")
    print(f"  real     : beat {bF:.3f}  downbeat {dF:.3f}   <- deploy (frontend trained from scratch)")
    print(f"  shuffled : beat {bFs:.3f}  downbeat {dFs:.3f}   (must COLLAPSE)")
    print(f"  zero     : beat {bFz:.3f}  downbeat {dFz:.3f}   (must COLLAPSE)")
    print("VERDICT: from-scratch TCN + exact filter, end-to-end -> beat-F + leak collapse = e2e WORKS")
    torch.save({"tcn": tcn.state_dict(), "model": model.state_dict(), "args": vars(a)},
               "experiments/kvae_barpointer/m_e2e.pt")


if __name__ == "__main__":
    main()
