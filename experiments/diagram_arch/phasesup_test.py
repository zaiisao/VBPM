"""FIX TEST: phase supervision. Pull the ENCODER's phi toward the GT rotating bar-phase ramp during
training (phi is NOT clamped -- the encoder produces it). At deploy (h-only, b=0) the encoder must
produce a rotating phi from audio alone. Measure the GEOMETRIC read-out (beats = phi-wraps) -- the
deployment your design uses. If it jumps from ~0.03-0.13 (free) to high, the fix works.

Reports the GEOMETRIC (bar-pointer) read-out as the headline (your deployment); decoder read-out only
as the discarded-head reference; plus phi revolutions (did it actually start rotating?).
"""
import sys, glob, math, random, importlib.util
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/CHART")
s = importlib.util.spec_from_file_location("da", "experiments/diagram_arch/run.py"); da = importlib.util.module_from_spec(s); s.loader.exec_module(da)
BPVAE, rollout, peaks, fmeas, phase_beats, phase_downbeats = da.BPVAE, da.rollout, da.peaks, da.fmeas, da.phase_beats, da.phase_downbeats
DEV = da.DEV; TWO_PI = 2 * math.pi; FPS = 86.1328125


def gt_barphase(beat_fr, m, T):
    phi = np.zeros(T)
    if len(beat_fr) < 2: return phi
    vals = np.arange(len(beat_fr)) * (TWO_PI / m)
    for k in range(len(beat_fr) - 1):
        a, b = beat_fr[k], beat_fr[k + 1]
        phi[a:b] = np.linspace(vals[k], vals[k + 1], b - a, endpoint=False)
    phi[beat_fr[-1]:] = vals[-1]
    return phi % TWO_PI


def load(cache_dir, n, seed):
    fs = sorted(glob.glob(f"{cache_dir}/*.pt")); random.Random(seed).shuffle(fs)
    out = []
    for f in fs[:n]:
        d = torch.load(f, map_location="cpu"); hh = d["activations"].float()
        if hh.shape[0] < 400 or d["beat_targets"].sum() < 8: continue
        out.append((hh, d["beat_targets"].float(), d["downbeat_targets"].float()))
    return out


def make_batch(songs, frames, bs):
    hs, bs_, ds_, ph = [], [], [], []
    while len(hs) < bs:
        hh, b, db = random.choice(songs)
        if hh.shape[0] <= frames: continue
        s0 = random.randint(0, hh.shape[0] - frames)
        bb = b[s0:s0 + frames]; dd = db[s0:s0 + frames]
        if bb.sum() < 2: continue
        bf = np.where(bb.numpy() > 0.5)[0]
        hs.append(hh[s0:s0 + frames]); bs_.append(bb); ds_.append(dd)
        ph.append(torch.tensor(gt_barphase(bf, 4, frames), dtype=torch.float32))
    return (torch.stack(hs).to(DEV), torch.stack(bs_).to(DEV), torch.stack(ds_).to(DEV), torch.stack(ph).to(DEV))


@torch.no_grad()
def evaluate(model, val, h_mode="real", frames=1600):
    model.eval(); gb, gd, dcb, revs = [], [], [], []
    n = len(val)
    for i, (hh, b, db) in enumerate(val):
        h_use = val[(i + 1) % n][0] if h_mode == "shuffle" else hh
        T = min(h_use.shape[0], b.shape[0], frames)
        h_in = (torch.zeros(1, T, hh.shape[1], device=DEV) if h_mode == "zero" else h_use[:T].unsqueeze(0).to(DEV))
        z0 = torch.zeros(1, T, device=DEV)
        _, pm, logits = rollout(model, h_in, z0, z0, sample=False, compute_kl=False)
        phi = pm[0].cpu().numpy(); prob = torch.sigmoid(logits)[0].cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS; dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2:
            gb.append(fmeas(ref, phase_beats(phi, 4))); dcb.append(fmeas(ref, peaks(prob[:, 0])))
        if len(dref) >= 2:
            gd.append(fmeas(dref, phase_downbeats(phi)))
        dphi = np.diff(phi); revs.append(float(np.sum(np.where(dphi < -math.pi, dphi + TWO_PI, dphi)) / TWO_PI))
    model.train()
    m = lambda x: float(np.nanmean(x)) if x else float("nan")
    return m(gb), m(gd), m(dcb), m(revs)


def main():
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    lam = 5.0; b_drop = 0.5
    print(f"loading bt features (train 200 / val 40) | phase-supervision lambda={lam}, b_drop={b_drop}", flush=True)
    train = load("cache/acts/bt_train_rich", 200, 1); val = load("cache/acts/bt_val_rich", 40, 2)
    print(f"train={len(train)} val={len(val)} | phi from ENCODER (not clamped), supervised to GT ramp", flush=True)
    model = BPVAE(h_dim=512, hidden=64).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, 401):
        temp = 1.0 + (0.3 - 1.0) * min(step / 400, 1.0)
        h, b, db, gp = make_batch(train, 256, 16)
        keep = (torch.rand(h.shape[0], 1, device=DEV) >= b_drop).float()
        (klm, klp, klt), pm, logits = rollout(model, h, b * keep, db * keep, temp, sample=True, compute_kl=True)
        pw = torch.tensor([8.0, 20.0], device=DEV)
        recon = F.binary_cross_entropy_with_logits(logits, torch.stack([b, db], -1), pos_weight=pw, reduction="none").sum((1, 2))
        psup = (1.0 - torch.cos(pm - gp)).sum(1)            # circular phase supervision, per sequence
        loss = (recon + klm + klp + klt + lam * psup).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % 200 == 0 or step == 400:
            gb, gd, dcb, rv = evaluate(model, val, "real")
            print(f"\nstep {step} | recon {float(recon.mean()):.1f} psup {float(psup.mean()):.1f} | "
                  f"H-ONLY geometric: beat {gb:.3f} downbeat {gd:.3f} | phi revolutions {rv:.1f} "
                  f"(GT bars~{int(np.mean([(d>0.5).sum() for _,_,d in val]))}) | (decoder ref {dcb:.3f})", flush=True)
    gb, gd, dcb, rv = evaluate(model, val, "real")
    gbs, gds, _, _ = evaluate(model, val, "shuffle"); gbz, gdz, _, _ = evaluate(model, val, "zero")
    print("\n--- FINAL (h-only deploy, GEOMETRIC bar-pointer read-out = your deployment) ---")
    print(f"  real audio : beat {gb:.3f}  downbeat {gd:.3f}   phi-revolutions {rv:.1f}")
    print(f"  shuffled   : beat {gbs:.3f}  downbeat {gds:.3f}   (must collapse)")
    print(f"  zero       : beat {gbz:.3f}  downbeat {gdz:.3f}   (must collapse)")
    print(f"  decoder ref (discarded head): beat {dcb:.3f}")
    print("VERDICT: if geometric beat/downbeat jump high AND phi rotates (revolutions ~= #bars) -> phase-sup FIXES it")


if __name__ == "__main__":
    main()
