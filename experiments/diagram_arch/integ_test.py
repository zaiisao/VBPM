"""FIX TEST 2: integrate phi from the tempo + supervise the TEMPO (rate), not the absolute phase.
Circular phase-supervision failed to fix the RATE (no consistent gradient once phi under-rotates).
Here phi is the INTEGRAL of the encoder's per-frame tempo (phi_t = phi_{t-1} + exp(tempo)), so rotation
is structural, and we supervise the tempo to the GT advance (a non-circular target -> strong gradient).
This also makes the tempo correct (the user's point). Deploy h-only, read beats GEOMETRICALLY.
"""
import sys, glob, math, random, importlib.util
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/CHART")
s = importlib.util.spec_from_file_location("da", "experiments/diagram_arch/run.py"); da = importlib.util.module_from_spec(s); s.loader.exec_module(da)
BPVAE, peaks, fmeas, phase_beats, phase_downbeats = da.BPVAE, da.peaks, da.fmeas, da.phase_beats, da.phase_downbeats
DEV = da.DEV; TWO_PI = 2 * math.pi; FPS = 86.1328125


def gt_barphase(beat_fr, m, T):
    phi = np.zeros(T)
    if len(beat_fr) < 2: return phi
    vals = np.arange(len(beat_fr)) * (TWO_PI / m)
    for k in range(len(beat_fr) - 1):
        a, b = beat_fr[k], beat_fr[k + 1]; phi[a:b] = np.linspace(vals[k], vals[k + 1], b - a, endpoint=False)
    phi[beat_fr[-1]:] = vals[-1]
    return phi % TWO_PI


def integ_rollout(model, h, b_in, db_in):
    """phi = INTEGRAL of the encoder's tempo. Returns phis[B,T], tempo_logs[B,T], logits[B,T,2]."""
    B, T, _ = h.shape
    pc = model.enc_post(h, b_in, db_in)
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, 0], z0], -1)))
    m = F.softmax(qm, -1); lt = qtm; phi = qpm                  # initial phase from encoder
    zf = [model.zfeat(m, phi, lt)]; phis = [phi]; temps = [qtm]; mprev, phiprev, ltprev = m, phi, lt
    for t in range(1, T):
        zp = model.zfeat(mprev, phiprev, ltprev)
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, t], zp], -1)))
        m = F.softmax(qm, -1); lt = qtm
        phi = (phiprev + torch.exp(ltprev)) % TWO_PI            # INTEGRATE the tempo -> structural rotation
        zf.append(model.zfeat(m, phi, lt)); phis.append(phi); temps.append(qtm); mprev, phiprev, ltprev = m, phi, lt
    logits = torch.stack([model.decode(zf[t]) for t in range(T)], 1)
    return torch.stack(phis, 1), torch.stack(temps, 1), logits


def load(cache_dir, n, seed):
    fs = sorted(glob.glob(f"{cache_dir}/*.pt")); random.Random(seed).shuffle(fs); out = []
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
        s0 = random.randint(0, hh.shape[0] - frames); bb = b[s0:s0 + frames]; dd = db[s0:s0 + frames]
        if bb.sum() < 2: continue
        bf = np.where(bb.numpy() > 0.5)[0]
        hs.append(hh[s0:s0 + frames]); bs_.append(bb); ds_.append(dd); ph.append(torch.tensor(gt_barphase(bf, 4, frames), dtype=torch.float32))
    return torch.stack(hs).to(DEV), torch.stack(bs_).to(DEV), torch.stack(ds_).to(DEV), torch.stack(ph).to(DEV)


@torch.no_grad()
def evaluate(model, val, h_mode="real", frames=1600):
    model.eval(); gb, gd, revs, bpm = [], [], [], []
    n = len(val)
    for i, (hh, b, db) in enumerate(val):
        h_use = val[(i + 1) % n][0] if h_mode == "shuffle" else hh
        T = min(h_use.shape[0], b.shape[0], frames)
        h_in = torch.zeros(1, T, hh.shape[1], device=DEV) if h_mode == "zero" else h_use[:T].unsqueeze(0).to(DEV)
        z = torch.zeros(1, T, device=DEV)
        phis, temps, _ = integ_rollout(model, h_in, z, z)
        phi = phis[0].cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS; dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2: gb.append(fmeas(ref, phase_beats(phi, 4)))
        if len(dref) >= 2: gd.append(fmeas(dref, phase_downbeats(phi)))
        dphi = np.diff(phi); revs.append(float(np.sum(np.where(dphi < -math.pi, dphi + TWO_PI, dphi)) / TWO_PI))
        bpm.append(60 * FPS * 4 * float(np.exp(temps[0].cpu().numpy()).mean()) / TWO_PI)
    model.train(); m = lambda x: float(np.nanmean(x)) if x else float("nan")
    return m(gb), m(gd), m(revs), m(bpm)


def main():
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    lt_w, lp_w, b_drop = 10.0, 2.0, 0.5
    print(f"phi=INTEGRAL of tempo | tempo-sup w={lt_w}, offset phase-sup w={lp_w}, b_drop={b_drop}", flush=True)
    train = load("cache/acts/bt_train_rich", 200, 1); val = load("cache/acts/bt_val_rich", 40, 2)
    print(f"train={len(train)} val={len(val)}", flush=True)
    model = BPVAE(h_dim=512, hidden=64).to(DEV); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, 501):
        h, b, db, gp = make_batch(train, 256, 16)
        keep = (torch.rand(h.shape[0], 1, device=DEV) >= b_drop).float()
        phis, temps, logits = integ_rollout(model, h, b * keep, db * keep)
        pw = torch.tensor([8.0, 20.0], device=DEV)
        recon = F.binary_cross_entropy_with_logits(logits, torch.stack([b, db], -1), pos_weight=pw, reduction="none").sum((1, 2))
        gt_adv = (torch.diff(gp, dim=1) % TWO_PI).clamp(1e-4, math.pi)        # GT per-frame advance
        tempo_sup = F.mse_loss(temps[:, 1:], torch.log(gt_adv), reduction="none").sum(1)   # supervise rate (log-space)
        phase_sup = (1.0 - torch.cos(phis - gp)).sum(1)                       # offset only
        loss = (recon + lt_w * tempo_sup + lp_w * phase_sup).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % 250 == 0 or step == 500:
            gb, gd, rv, bpm = evaluate(model, val, "real")
            nb = int(np.mean([(d > 0.5).sum() for _, _, d in val]))
            print(f"\nstep {step} | recon {float(recon.mean()):.1f} tsup {float(tempo_sup.mean()):.2f} | "
                  f"H-ONLY geometric: beat {gb:.3f} downbeat {gd:.3f} | phi-revs {rv:.1f} (GT bars ~{nb}) | tempo ~{bpm:.0f}BPM", flush=True)
    gb, gd, rv, bpm = evaluate(model, val, "real")
    gbs, gds, _, _ = evaluate(model, val, "shuffle"); gbz, gdz, _, _ = evaluate(model, val, "zero")
    print("\n--- FINAL (h-only deploy, GEOMETRIC bar-pointer read-out) ---")
    print(f"  real     : beat {gb:.3f}  downbeat {gd:.3f}  phi-revs {rv:.1f}  tempo ~{bpm:.0f}BPM")
    print(f"  shuffled : beat {gbs:.3f}  downbeat {gds:.3f}   (must collapse)")
    print(f"  zero     : beat {gbz:.3f}  downbeat {gdz:.3f}   (must collapse)")
    print("VERDICT: integrated-phi + tempo-sup -> if geometric beat/db jump AND revs~=#bars AND tempo~real -> FIXED")


if __name__ == "__main__":
    main()
