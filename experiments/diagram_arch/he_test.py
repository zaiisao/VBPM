"""FIX TEST 5 (He 2019): aggressive inference. Integrated-phi (bounded tempo) + LEARNED decoder.
Per outer step, do K encoder-only updates (fresh batches) before one decoder update -- pushing the
encoder to maximize I(x;z) so it escapes the collapse basin (He, Berg-Kirkpatrick et al. 2019).
Measure the GEOMETRIC read-out (deployment), the decoder read-out (reference), and leak controls.
"""
import sys, glob, math, random, importlib.util
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/CHART")
s = importlib.util.spec_from_file_location("da", "experiments/diagram_arch/run.py"); da = importlib.util.module_from_spec(s); s.loader.exec_module(da)
BPVAE, peaks, fmeas, phase_beats, phase_downbeats = da.BPVAE, da.peaks, da.fmeas, da.phase_beats, da.phase_downbeats
DEV = da.DEV; TWO_PI = 2 * math.pi; FPS = 86.1328125; M = 4
LT_MIN = math.log(TWO_PI * 40 / 60 / M / FPS); LT_MAX = math.log(TWO_PI * 250 / 60 / M / FPS)
ENC_KEYS = ("post_gru", "post_ctx", "post_head", "z0")


def integ(model, h, b_in, db_in):
    B, T, _ = h.shape
    pc = model.enc_post(h, b_in, db_in)
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, 0], z0], -1)))
    phi = qpm % TWO_PI; lt = qtm.clamp(LT_MIN, LT_MAX); m = F.softmax(qm, -1)
    zf = [model.zfeat(m, phi, lt)]; phis = [phi]; temps = [lt]; phiprev, ltprev, mprev = phi, lt, m
    for t in range(1, T):
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, t], model.zfeat(mprev, phiprev, ltprev)], -1)))
        lt = qtm.clamp(LT_MIN, LT_MAX); phi = (phiprev + torch.exp(ltprev)) % TWO_PI; m = F.softmax(qm, -1)
        zf.append(model.zfeat(m, phi, lt)); phis.append(phi); temps.append(lt); phiprev, ltprev, mprev = phi, lt, m
    logits = torch.stack([model.decode(zf[t]) for t in range(T)], 1)
    return torch.stack(phis, 1), logits


def load(cd, n, seed):
    fs = sorted(glob.glob(f"{cd}/*.pt")); random.Random(seed).shuffle(fs); out = []
    for f in fs[:n]:
        d = torch.load(f, map_location="cpu"); hh = d["activations"].float()
        if hh.shape[0] < 400 or d["beat_targets"].sum() < 8: continue
        out.append((hh, d["beat_targets"].float(), d["downbeat_targets"].float()))
    return out


def batch(songs, frames, bs):
    hs, bs_, ds_ = [], [], []
    while len(hs) < bs:
        hh, b, db = random.choice(songs)
        if hh.shape[0] <= frames: continue
        s0 = random.randint(0, hh.shape[0] - frames); bb = b[s0:s0 + frames]
        if bb.sum() < 2: continue
        hs.append(hh[s0:s0 + frames]); bs_.append(bb); ds_.append(db[s0:s0 + frames])
    return torch.stack(hs).to(DEV), torch.stack(bs_).to(DEV), torch.stack(ds_).to(DEV)


def loss_fn(model, h, b, db):
    keep = (torch.rand(h.shape[0], 1, device=DEV) >= 0.5).float()
    phis, logits = integ(model, h, b * keep, db * keep)
    pw = torch.tensor([8.0, 20.0], device=DEV)
    return F.binary_cross_entropy_with_logits(logits, torch.stack([b, db], -1), pos_weight=pw)


@torch.no_grad()
def evaluate(model, val, h_mode="real", frames=1600):
    model.eval(); gb, gd, dcb, revs = [], [], [], []
    n = len(val)
    for i, (hh, b, db) in enumerate(val):
        h_use = val[(i + 1) % n][0] if h_mode == "shuffle" else hh
        T = min(h_use.shape[0], b.shape[0], frames)
        h_in = torch.zeros(1, T, hh.shape[1], device=DEV) if h_mode == "zero" else h_use[:T].unsqueeze(0).to(DEV)
        z = torch.zeros(1, T, device=DEV)
        phis, logits = integ(model, h_in, z, z); phi = phis[0].cpu().numpy(); prob = torch.sigmoid(logits)[0].cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS; dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2: gb.append(fmeas(ref, phase_beats(phi, 4))); dcb.append(fmeas(ref, peaks(prob[:, 0])))
        if len(dref) >= 2: gd.append(fmeas(dref, phase_downbeats(phi)))
        dphi = np.diff(phi); revs.append(float(np.sum(np.where(dphi < -math.pi, dphi + TWO_PI, dphi)) / TWO_PI))
    model.train(); m = lambda x: float(np.nanmean(x)) if x else float("nan")
    return m(gb), m(gd), m(dcb), m(revs)


def main():
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    K = 5  # He aggressive inner encoder steps
    print(f"He-2019 aggressive inference (K={K} encoder steps/outer) | integrated-phi + LEARNED decoder + bounded tempo", flush=True)
    train = load("cache/acts/bt_train_rich", 200, 1); val = load("cache/acts/bt_val_rich", 40, 2)
    model = BPVAE(h_dim=512, hidden=64).to(DEV)
    enc = [p for nm, p in model.named_parameters() if any(k in nm for k in ENC_KEYS)]
    dec = [p for nm, p in model.named_parameters() if not any(k in nm for k in ENC_KEYS)]
    print(f"encoder params {sum(p.numel() for p in enc):,} | other {sum(p.numel() for p in dec):,}", flush=True)
    enc_opt = torch.optim.Adam(enc, lr=1e-3); dec_opt = torch.optim.Adam(dec, lr=1e-3)
    for step in range(1, 201):
        for _ in range(K):                       # aggressive: burn the encoder on fresh batches
            h, b, db = batch(train, 256, 16)
            enc_opt.zero_grad(); l = loss_fn(model, h, b, db); l.backward()
            torch.nn.utils.clip_grad_norm_(enc, 5.0); enc_opt.step()
        h, b, db = batch(train, 256, 16)         # then a generative (decoder) step
        dec_opt.zero_grad(); l = loss_fn(model, h, b, db); l.backward()
        torch.nn.utils.clip_grad_norm_(dec, 5.0); dec_opt.step()
        if step % 100 == 0 or step == 200:
            gb, gd, dcb, rv = evaluate(model, val, "real")
            print(f"\nstep {step} | loss {float(l):.2f} | H-ONLY geometric: beat {gb:.3f} downbeat {gd:.3f} | "
                  f"phi-revs {rv:.1f} | (decoder ref {dcb:.3f})", flush=True)
    gb, gd, dcb, rv = evaluate(model, val, "real")
    gbs, gds, _, _ = evaluate(model, val, "shuffle"); gbz, gdz, _, _ = evaluate(model, val, "zero")
    print("\n--- FINAL (h-only deploy) ---")
    print(f"  geometric real     : beat {gb:.3f}  downbeat {gd:.3f}  phi-revs {rv:.1f}")
    print(f"  geometric shuffled : beat {gbs:.3f}  downbeat {gds:.3f}   (must COLLAPSE)")
    print(f"  geometric zero     : beat {gbz:.3f}  downbeat {gdz:.3f}   (must COLLAPSE)")
    print(f"  decoder ref        : beat {dcb:.3f}")
    print("VERDICT: He aggressive -> if geometric jumps + leak collapses = ESCAPED collapse; if only decoder works = feature-degeneracy; if nothing = wall")


if __name__ == "__main__":
    main()
