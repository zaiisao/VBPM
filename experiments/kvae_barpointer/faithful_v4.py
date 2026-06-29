"""Test the user's idea: dropout the SHORTCUT channels so the decoder must use phi-dot.
phi = integral of phi-dot (closes the free-phase shortcut) + DROPOUT on the meter m in the decoder
input (closes the meter info-channel, the biggest shortcut per ablation). If phi-dot then grounds
(phi rotates ~#bars, leak collapses), redundancy was the blocker; if it still floors, the
ill-conditioned integral gradient (obstacle #2) is the wall.
"""
import sys, math, importlib.util, argparse, random
import numpy as np, torch, torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"; sys.path.insert(0, ROOT)
fv = importlib.util.spec_from_file_location("fv", f"{ROOT}/experiments/kvae_barpointer/faithful_v2.py")
v2 = importlib.util.module_from_spec(fv); fv.loader.exec_module(v2)
da = v2.da; BPVAE, load_pool, sample_batch = da.BPVAE, da.load_pool, da.sample_batch
soft_lt = v2.soft_lt
kl_von_mises, kl_log_normal, kl_categorical = da.kl_von_mises, da.kl_log_normal, da.kl_categorical
fmeas, phase_beats, phase_downbeats = da.fmeas, da.phase_beats, da.phase_downbeats
DEV = da.DEV; TWO_PI = 2*math.pi; FPS = 86.1328125; M = 4


def rollout(model, h, b_in, db_in, temp=0.5, sample=True, compute_kl=True, m_drop=0.0):
    B, T, _ = h.shape
    pc = model.enc_post(h, b_in, db_in); pr = model.enc_prior(h) if compute_kl else None
    klm = klp = klt = (h.new_zeros(B) if compute_kl else None)
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    mkeep = (torch.rand(B, 1, device=DEV) >= m_drop).float() if m_drop > 0 else torch.ones(B, 1, device=DEV)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, 0], z0], -1)))
    m = (da.gumbel_softmax(qm, temp) if sample else F.softmax(qm, -1)); lt = soft_lt(qtm)
    phi = (da.sample_von_mises(qpm % TWO_PI, qpk) % TWO_PI) if sample else (qpm % TWO_PI)
    if compute_kl:
        pm, ppm, ppk, ptm, pts = model.unpack(model.prior_init(pr.mean(1)))
        klm = klm + kl_categorical(torch.log_softmax(qm, -1), torch.log_softmax(pm, -1))
        klp = klp + kl_von_mises(phi, qpk, ppm, ppk); klt = klt + kl_log_normal(qtm, qts, ptm, pts)
    zf = [model.zfeat(m * mkeep, phi, lt)]; phis = [phi]; mprev, phiprev, ltprev = m, phi, lt
    for t in range(1, T):
        zp = model.zfeat(mprev, phiprev, ltprev)
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, t], zp], -1)))
        m = (da.gumbel_softmax(qm, temp) if sample else F.softmax(qm, -1)); lt = soft_lt(qtm)
        phi_mean = (phiprev + torch.exp(lt)) % TWO_PI
        phi = (da.sample_von_mises(phi_mean, qpk) % TWO_PI) if sample else phi_mean
        if compute_kl:
            ppm = (phiprev + torch.exp(ltprev)) % TWO_PI
            ppk = F.softplus(model.prior_pk(pr[:, t]).squeeze(-1)) + 0.01
            ptm = ltprev; pts = F.softplus(model.prior_ts(pr[:, t]).squeeze(-1)) + 1e-3
            klm = klm + kl_categorical(torch.log_softmax(qm, -1), model.meter_logp(mprev, phi, phiprev, pr[:, t]))
            klp = klp + kl_von_mises(phi_mean, qpk, ppm, ppk); klt = klt + kl_log_normal(qtm, qts, ptm, pts)
        zf.append(model.zfeat(m * mkeep, phi, lt)); phis.append(phi); mprev, phiprev, ltprev = m, phi_mean, lt
    logits = torch.stack([model.decode(zf[t]) for t in range(T)], 1)
    return ((klm, klp, klt) if compute_kl else None), torch.stack(phis, 1), logits


@torch.no_grad()
def evaluate(model, val, h_mode="real", frames=1600):
    model.eval(); gb, gd, revs, bpm = [], [], [], []; n = len(val)
    for i, (hh, b, db) in enumerate(val):
        h_use = val[(i + 1) % n][0] if h_mode == "shuffle" else hh
        T = min(h_use.shape[0], b.shape[0], frames)
        h_in = torch.zeros(1, T, hh.shape[1], device=DEV) if h_mode == "zero" else h_use[:T].unsqueeze(0).to(DEV)
        z = torch.zeros(1, T, device=DEV)
        _, phis, _ = rollout(model, h_in, z, z, sample=False, compute_kl=False, m_drop=0.0)
        phi = phis[0].cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS; dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2: gb.append(fmeas(ref, phase_beats(phi, M)))
        if len(dref) >= 2: gd.append(fmeas(dref, phase_downbeats(phi)))
        dphi = np.diff(phi); adv = np.where(dphi < -math.pi, dphi + TWO_PI, dphi)
        revs.append(float(np.sum(adv) / TWO_PI)); a2 = adv[adv > 1e-4]
        bpm.append(M * float(np.median(a2)) / TWO_PI * FPS * 60 if len(a2) else 0.0)
    model.train(); mn = lambda x: float(np.nanmean(x)) if x else float("nan")
    return mn(gb), mn(gd), mn(revs), mn(bpm)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--m_drop", type=float, default=0.5); ap.add_argument("--n_train", type=int, default=400); ap.add_argument("--n_val", type=int, default=40)
    a = ap.parse_args(); torch.manual_seed(0); np.random.seed(0); random.seed(0)
    train = load_pool("cache/acts/bt_train_rich", a.n_train, seed=1); val = load_pool("cache/acts/bt_val_rich", a.n_val, seed=2)
    nb = int(np.mean([(d > 0.5).sum() for _, _, d in val]))
    print(f"FAITHFUL v4 (phi=integral + DROPOUT meter m_drop={a.m_drop}) | train={len(train)} val={len(val)} | GT #bars~{nb}", flush=True)
    model = BPVAE(h_dim=512, hidden=64).to(DEV); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pw = torch.tensor([8.0, 20.0], device=DEV)
    for step in range(1, a.steps + 1):
        temp = 1.0 + (0.3 - 1.0) * min(step / a.steps, 1.0)
        h, b, db = sample_batch(train, 256, 16)
        keep = (torch.rand(h.shape[0], 1, device=DEV) >= 0.5).float()
        (klm, klp, klt), phis, logits = rollout(model, h, b * keep, db * keep, temp, sample=True, compute_kl=True, m_drop=a.m_drop)
        recon = F.binary_cross_entropy_with_logits(logits, torch.stack([b, db], -1), pos_weight=pw, reduction="none").sum((1, 2))
        klm = klm.clamp(min=0.1*256); klp = klp.clamp(min=0.1*256); klt = klt.clamp(min=0.1*256)
        loss = (recon + klm + klp + klt).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % 100 == 0 or step == a.steps:
            gb, gd, rv, bp = evaluate(model, val, "real")
            print(f"  step {step:4d} | recon {float(recon.mean()):.1f} | GEOM beat {gb:.3f} db {gd:.3f} | phi-revs {rv:.1f}/{nb} | tempo {bp:.0f}BPM", flush=True)
    gb, gd, rv, bp = evaluate(model, val, "real"); gbs, gds, _, _ = evaluate(model, val, "shuffle"); gbz, gdz, _, _ = evaluate(model, val, "zero")
    print("\n--- FINAL (phi=integral + meter dropout) ---")
    print(f"  real     : beat {gb:.3f}  db {gd:.3f}  phi-revs {rv:.1f}/{nb}  tempo {bp:.0f}BPM")
    print(f"  shuffled : beat {gbs:.3f}  zero : beat {gbz:.3f}  (must COLLAPSE)")
    print("VERDICT: if phi now rotates ~#bars + leak collapses => dropout grounded phi-dot (redundancy was the blocker)")
    print("DONE")


if __name__ == "__main__":
    main()
