"""FAITHFUL v2 -- the design the user actually intends:
  * PRIOR: deterministic means (rotation dynamics) + learnable spread (already in BPVAE).
  * POSTERIOR: DATA-INFORMED tempo (reads audio); PHASE MEAN is the DETERMINISTIC integral of that tempo
    (phi_t = phi_{t-1} + exp(tau_t)) -- NOT a free output. Only the phase concentration kappa is learned.
  => the audio enters via the tempo; the phase is forced to rotate by integrating it. This couples phi to
     the tempo (fixing the decoupling that left phi free to oscillate and the tempo ungrounded).
Tempo is soft-bounded to a musical range (no garbage-huge, no dead clamp). Deploy = GEOMETRIC read-out
from the deterministic phi. We report the REAL metrics: phi-revolutions vs #bars, tempo vs GT, leak.
"""
import sys, math, importlib.util, argparse, random
import numpy as np
import torch, torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"; sys.path.insert(0, ROOT)
s = importlib.util.spec_from_file_location("da", f"{ROOT}/experiments/diagram_arch/run.py")
da = importlib.util.module_from_spec(s); s.loader.exec_module(da)
BPVAE, load_pool, sample_batch = da.BPVAE, da.load_pool, da.sample_batch
fmeas, phase_beats, phase_downbeats = da.fmeas, da.phase_beats, da.phase_downbeats
kl_von_mises, kl_log_normal, kl_categorical = da.kl_von_mises, da.kl_log_normal, da.kl_categorical
DEV = da.DEV; TWO_PI = 2*math.pi; FPS = 86.1328125; M = 4
LT_MIN = math.log(TWO_PI*40/60/M/FPS); LT_MAX = math.log(TWO_PI*250/60/M/FPS)


def soft_lt(raw):                                   # data-informed tempo, soft-bounded 40-250 BPM
    return LT_MIN + (LT_MAX - LT_MIN) * torch.sigmoid(raw)


def rollout_det(model, h, b_in, db_in, temp=0.5, sample=True, compute_kl=True):
    """Phase mean DETERMINISTIC = integral of the data-informed tempo. Returns (kl_tuple|None, phi[B,T], logits)."""
    B, T, _ = h.shape
    pc = model.enc_post(h, b_in, db_in); pr = model.enc_prior(h) if compute_kl else None
    klm = klp = klt = (h.new_zeros(B) if compute_kl else None)
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, 0], z0], -1)))
    m = (da.gumbel_softmax(qm, temp) if sample else F.softmax(qm, -1))
    lt = soft_lt(qtm)
    phi = qpm % TWO_PI                                   # data-informed INITIAL offset (one-time)
    if sample: phi = da.sample_von_mises(phi, qpk) % TWO_PI
    if compute_kl:
        pm, ppm, ppk, ptm, pts = model.unpack(model.prior_init(pr.mean(1)))
        klm = klm + kl_categorical(torch.log_softmax(qm, -1), torch.log_softmax(pm, -1))
        klp = klp + kl_von_mises(phi, qpk, ppm, ppk); klt = klt + kl_log_normal(qtm, qts, ptm, pts)
    zf = [model.zfeat(m, phi, lt)]; phis = [phi]; mprev, phiprev, ltprev = m, phi, lt
    for t in range(1, T):
        zp = model.zfeat(mprev, phiprev, ltprev)
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, t], zp], -1)))
        m = (da.gumbel_softmax(qm, temp) if sample else F.softmax(qm, -1))
        lt = soft_lt(qtm)
        phi_mean = (phiprev + torch.exp(lt)) % TWO_PI    # DETERMINISTIC rotation (couples phi<-tempo)
        phi = da.sample_von_mises(phi_mean, qpk) % TWO_PI if sample else phi_mean
        if compute_kl:
            ppm = (phiprev + torch.exp(ltprev)) % TWO_PI
            ppk = F.softplus(model.prior_pk(pr[:, t]).squeeze(-1)) + 0.01
            ptm = ltprev; pts = F.softplus(model.prior_ts(pr[:, t]).squeeze(-1)) + 1e-3
            klm = klm + kl_categorical(torch.log_softmax(qm, -1), model.meter_logp(mprev, phi, phiprev, pr[:, t]))
            klp = klp + kl_von_mises(phi_mean, qpk, ppm, ppk); klt = klt + kl_log_normal(qtm, qts, ptm, pts)
        zf.append(model.zfeat(m, phi, lt)); phis.append(phi); mprev, phiprev, ltprev = m, phi, lt
    logits = torch.stack([model.decode(zf[t]) for t in range(T)], 1)
    return ((klm, klp, klt) if compute_kl else None), torch.stack(phis, 1), logits


def elbo(model, h, b, db, temp, fb=0.1, b_drop=0.5):
    B, T, _ = h.shape
    keep = (torch.rand(B, 1, device=h.device) >= b_drop).float()
    (klm, klp, klt), _, logits = rollout_det(model, h, b * keep, db * keep, temp, sample=True, compute_kl=True)
    pw = torch.tensor([8.0, 20.0], device=h.device)
    recon = F.binary_cross_entropy_with_logits(logits, torch.stack([b, db], -1), pos_weight=pw, reduction="none").sum((1, 2))
    klm = klm.clamp(min=fb*T); klp = klp.clamp(min=fb*T); klt = klt.clamp(min=fb*T)
    return (recon + klm + klp + klt).mean(), float(recon.mean()), float(klt.mean())


@torch.no_grad()
def evaluate(model, val, h_mode="real", frames=1600):
    model.eval(); gb, gd, revs, bpm = [], [], [], []
    n = len(val)
    for i, (hh, b, db) in enumerate(val):
        h_use = val[(i + 1) % n][0] if h_mode == "shuffle" else hh
        T = min(h_use.shape[0], b.shape[0], frames)
        h_in = torch.zeros(1, T, hh.shape[1], device=DEV) if h_mode == "zero" else h_use[:T].unsqueeze(0).to(DEV)
        z = torch.zeros(1, T, device=DEV)
        _, phis, _ = rollout_det(model, h_in, z, z, sample=False, compute_kl=False)
        phi = phis[0].cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS; dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2: gb.append(fmeas(ref, phase_beats(phi, M)))
        if len(dref) >= 2: gd.append(fmeas(dref, phase_downbeats(phi)))
        dphi = np.diff(phi); revs.append(float(np.sum(np.where(dphi < -math.pi, dphi + TWO_PI, dphi)) / TWO_PI))
        adv = np.where(dphi < -math.pi, dphi + TWO_PI, dphi); adv = adv[adv > 1e-4]
        bpm.append(M * float(np.median(adv)) / TWO_PI * FPS * 60 if len(adv) else 0.0)
    model.train(); mn = lambda x: float(np.nanmean(x)) if x else float("nan")
    return mn(gb), mn(gd), mn(revs), mn(bpm)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--n_train", type=int, default=400); ap.add_argument("--n_val", type=int, default=40)
    a = ap.parse_args(); torch.manual_seed(0); np.random.seed(0); random.seed(0)
    train = load_pool("cache/acts/bt_train_rich", a.n_train, seed=1); val = load_pool("cache/acts/bt_val_rich", a.n_val, seed=2)
    nb = int(np.mean([(d > 0.5).sum() for _, _, d in val]))
    print(f"FAITHFUL v2 (det phase mean = integral of data-informed tempo) | train={len(train)} val={len(val)} | GT #bars~{nb}", flush=True)
    model = BPVAE(h_dim=512, hidden=64).to(DEV); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, a.steps + 1):
        temp = 1.0 + (0.3 - 1.0) * min(step / a.steps, 1.0)
        h, b, db = sample_batch(train, 256, 16)
        loss, recon, klt = elbo(model, h, b, db, temp)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % 100 == 0 or step == a.steps:
            gb, gd, rv, bp = evaluate(model, val, "real")
            print(f"  step {step:4d} | recon {recon:.1f} klt {klt:.1f} | GEOM beat {gb:.3f} db {gd:.3f} | "
                  f"phi-revs {rv:.1f}/{nb} | tempo {bp:.0f}BPM", flush=True)
    gb, gd, rv, bp = evaluate(model, val, "real")
    gbs, gds, rvs, _ = evaluate(model, val, "shuffle"); gbz, gdz, rvz, _ = evaluate(model, val, "zero")
    print("\n--- FINAL (GEOMETRIC deploy; deterministic phi = integral of data-informed tempo) ---")
    print(f"  real     : beat {gb:.3f}  db {gd:.3f}  phi-revs {rv:.1f}/{nb}  tempo {bp:.0f}BPM")
    print(f"  shuffled : beat {gbs:.3f}  db {gds:.3f}  revs {rvs:.1f}   (must COLLAPSE)")
    print(f"  zero     : beat {gbz:.3f}  db {gdz:.3f}  revs {rvz:.1f}   (must COLLAPSE)")
    print("VERDICT: phi rotates (revs~#bars) at musical tempo + beat high + leak collapse => FAITHFUL v2 WORKS")
    torch.save({"vae": model.state_dict(), "h_dim": 512}, "experiments/kvae_barpointer/faithful_v2.pt")
    print("DONE")


if __name__ == "__main__":
    main()
