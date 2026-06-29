"""DECISIVE test of the user's hypothesis: would ABSOLUTE positional encodings make phi/dotphi
inferable from h? We append sinusoidal ABSOLUTE-frame-index channels to h and train the standard
BPVAE geometric model. Then the leak test swaps ONLY the audio part of h while keeping the SAME
positional channels (position is position). Predictions if PE is a CHEAT not a fix:
  * geometric beat-F rises in-domain (real) -- the model can fit average beat positions from t,
  * but SHUFFLE does NOT collapse (it's keying on position, not audio),
  * and ZERO-audio does NOT collapse (position alone still drives it).
Contrast: a true audio-locked model collapses hard under shuffle/zero. We run WITH-PE vs NO-PE.
"""
import sys, math, importlib.util, argparse, random
import numpy as np
import torch

ROOT = "/home/sogang/jaehoon/CHART"; sys.path.insert(0, ROOT)
s = importlib.util.spec_from_file_location("da", f"{ROOT}/experiments/diagram_arch/run.py")
da = importlib.util.module_from_spec(s); s.loader.exec_module(da)
BPVAE, elbo_loss, rollout, load_pool = da.BPVAE, da.elbo_loss, da.rollout, da.load_pool
phase_beats, phase_downbeats, fmeas = da.phase_beats, da.phase_downbeats, da.fmeas
DEV = da.DEV; FPS = 86.1328125; M = 4; TWO_PI = 2 * math.pi


def pe_channels(T, d=16):
    """Classic sinusoidal ABSOLUTE positional encoding over frame index 0..T-1 -> [T, d]."""
    pos = np.arange(T)[:, None].astype(np.float32)
    i = np.arange(d // 2)[None, :].astype(np.float32)
    div = np.power(10000.0, 2 * i / d)
    pe = np.zeros((T, d), dtype=np.float32)
    pe[:, 0::2] = np.sin(pos / div); pe[:, 1::2] = np.cos(pos / div)
    return torch.from_numpy(pe)


def augment(songs, n_pe):
    """Append PE channels to each song's audio h. Returns augmented songs + the n_pe split point."""
    out = []
    for h, b, db in songs:
        T = h.shape[0]
        hp = torch.cat([h, pe_channels(T, n_pe)], -1) if n_pe else h
        out.append((hp, b, db))
    return out


def sample_batch_aug(songs, frames, bs):
    """Crop random windows; PE rides along per-frame so absolute-position info is preserved in-crop."""
    idx = np.random.randint(0, len(songs), bs); hs, bs_, ds = [], [], []
    for j in idx:
        h, b, db = songs[j]; T = h.shape[0]
        st = np.random.randint(0, max(1, T - frames))
        hs.append(h[st:st + frames]); bs_.append(b[st:st + frames]); ds.append(db[st:st + frames])
    L = min(x.shape[0] for x in hs)
    h = torch.stack([x[:L] for x in hs]).to(DEV)
    b = torch.stack([x[:L] for x in bs_]).to(DEV); db = torch.stack([x[:L] for x in ds]).to(DEV)
    return h, b, db


@torch.no_grad()
def evaluate_leak(model, songs, n_pe, h_mode, frames=1600):
    """h_mode: real | shuffle (swap AUDIO only, keep PE) | zero (zero AUDIO, keep PE)."""
    model.eval(); gb, gd, dgb = [], [], []; n = len(songs)
    for i, (hp, b, db) in enumerate(songs):
        T = min(hp.shape[0], b.shape[0], frames)
        audio = hp[:, :hp.shape[1] - n_pe] if n_pe else hp
        pe = hp[:, hp.shape[1] - n_pe:] if n_pe else None
        if h_mode == "shuffle":
            audio = songs[(i + 1) % n][0][:, :songs[(i + 1) % n][0].shape[1] - n_pe] if n_pe else songs[(i + 1) % n][0]
        elif h_mode == "zero":
            audio = torch.zeros_like(audio)
        Tm = min(T, audio.shape[0], (pe.shape[0] if n_pe else T))
        a = audio[:Tm]
        h_in = (torch.cat([a, pe[:Tm]], -1) if n_pe else a).unsqueeze(0).to(DEV)
        z = torch.zeros(1, Tm, device=DEV)
        # geometric (free-run prior) read-out + decoder read-out
        _, phis, logits = rollout(model, h_in, z, z, sample=False, compute_kl=False)
        phi = phis[0].cpu().numpy()
        ref = np.where(b.numpy()[:Tm] > 0.5)[0] / FPS; dref = np.where(db.numpy()[:Tm] > 0.5)[0] / FPS
        if len(ref) >= 2:
            gb.append(fmeas(ref, phase_beats(phi, M)))
            dgb.append(fmeas(ref, da.peaks(torch.sigmoid(logits[0, :, 0]).cpu().numpy())))
        if len(dref) >= 2: gd.append(fmeas(dref, phase_downbeats(phi)))
    model.train(); f = lambda x: float(np.nanmean(x)) if x else float("nan")
    return f(gb), f(gd), f(dgb)


def run(tag, n_pe, train, val, steps):
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    model = BPVAE(h_dim=512 + n_pe, hidden=64).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, steps + 1):
        temp = 1.0 + (0.3 - 1.0) * min(step / steps, 1.0)
        h, b, db = sample_batch_aug(train, 256, 16)
        loss, info = elbo_loss(model, h, b, db, temp, 8.0, 20.0, 0.1, 0.5)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % 200 == 0 or step == steps:
            gb, gd, dg = evaluate_leak(model, val, n_pe, "real")
            print(f"  [{tag}] step {step:4d} | GEOM beat {gb:.3f} db {gd:.3f} | DEC beat {dg:.3f}", flush=True)
    gb, gd, dg = evaluate_leak(model, val, n_pe, "real")
    gbs, _, dgs = evaluate_leak(model, val, n_pe, "shuffle")
    gbz, _, dgz = evaluate_leak(model, val, n_pe, "zero")
    print(f"  [{tag}] FINAL geom: real {gb:.3f} shuf {gbs:.3f} zero {gbz:.3f} | dec: real {dg:.3f} shuf {dgs:.3f} zero {dgz:.3f}", flush=True)
    return (gb, gbs, gbz), (dg, dgs, dgz)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--n_train", type=int, default=400); ap.add_argument("--n_val", type=int, default=40)
    ap.add_argument("--n_pe", type=int, default=16)
    a = ap.parse_args()
    train = load_pool("cache/acts/bt_train_rich", a.n_train, seed=1)
    val = load_pool("cache/acts/bt_val_rich", a.n_val, seed=2)
    print(f"PE PROBE | train={len(train)} val={len(val)} | n_pe={a.n_pe}\n", flush=True)
    print("=== NO PE (baseline, audio-only) ===", flush=True)
    nb = run("nope", 0, train, val, a.steps)
    print("\n=== WITH absolute PE ===", flush=True)
    tr_pe = augment(train, a.n_pe); va_pe = augment(val, a.n_pe)
    wp = run("withpe", a.n_pe, tr_pe, va_pe, a.steps)
    print("\n==== VERDICT ====")
    for name, (g, d) in (("NO-PE", nope := nb), ("WITH-PE", wp)):
        gr, gs, gz = g
        verdict = "audio-locked" if (gr > 0.45 and gs < gr - 0.2 and gz < gr - 0.2) else \
                  ("POSITION-LOCKED (cheat)" if (gr > 0.45 and (gs > gr - 0.15 or gz > gr - 0.15)) else "weak")
        print(f"  {name:8s}: geom real {gr:.3f} shuf {gs:.3f} zero {gz:.3f} -> {verdict}")
    print("  If WITH-PE has HIGH real but shuf/zero DON'T collapse -> PE is a position cheat, not an audio fix")
    print("DONE")


if __name__ == "__main__":
    main()
