"""IDEA #1: SOFT TEMPO SELECTION over a fixed BPM basis (no free continuous tempo).
advance_t = sum_k softmax(tempo_sel(z))_k * adv_k, with adv_k = K fixed log-spaced advances (40-250 BPM).
phi = integral of advance (rotates by construction). Geometric emission kcos(M*phi). This fixes the two
measured failure modes: (a) no saturation (bounded convex combo), (b) gradient is classification-like
over a tempo grid (well-conditioned), not regression-through-an-integral.
"""
import sys, math, importlib.util, argparse, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"; sys.path.insert(0, ROOT)
s = importlib.util.spec_from_file_location("da", f"{ROOT}/experiments/diagram_arch/run.py")
da = importlib.util.module_from_spec(s); s.loader.exec_module(da)
BPVAE, load_pool, sample_batch = da.BPVAE, da.load_pool, da.sample_batch
fmeas, phase_beats, phase_downbeats = da.fmeas, da.phase_beats, da.phase_downbeats
gumbel_softmax, kl_von_mises, kl_categorical = da.gumbel_softmax, da.kl_von_mises, da.kl_categorical
DEV = da.DEV; TWO_PI = 2*math.pi; FPS = 86.1328125; M = 4; KAPPA = 8.0; K_T = 20
BPMS = torch.exp(torch.linspace(math.log(40), math.log(250), K_T)).to(DEV)
ADV = (TWO_PI * BPMS / 60 / M / FPS)                       # [K_T] fixed per-frame bar-phase advances


def geom_logits(phi):
    return torch.stack([KAPPA * torch.cos(M * phi), KAPPA * torch.cos(phi)], -1)


def rollout(model, sel, h, b_in, db_in, temp=0.5, sample=True, compute_kl=True):
    B, T, _ = h.shape
    pc = model.enc_post(h, b_in, db_in); pr = model.enc_prior(h) if compute_kl else None
    klm = klp = klw = (h.new_zeros(B) if compute_kl else None)
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    inp = torch.cat([pc[:, 0], z0], -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(inp))
    m = gumbel_softmax(qm, temp) if sample else F.softmax(qm, -1)
    w = F.softmax(sel(inp), -1); adv = (w * ADV).sum(-1)
    phi = (da.sample_von_mises(qpm % TWO_PI, qpk) % TWO_PI) if sample else (qpm % TWO_PI)
    if compute_kl:
        pm, ppm, ppk, ptm, pts = model.unpack(model.prior_init(pr.mean(1)))
        klm = klm + kl_categorical(torch.log_softmax(qm, -1), torch.log_softmax(pm, -1))
        klp = klp + kl_von_mises(phi, qpk, ppm, ppk)
        klw = klw + (w * (torch.log(w + 1e-9) - math.log(1.0 / K_T))).sum(-1)   # KL(w || uniform)
    phis = [phi]; mprev, phiprev, ltprev = m, phi, torch.log(adv)
    for t in range(1, T):
        zp = model.zfeat(mprev, phiprev, ltprev)
        inp = torch.cat([pc[:, t], zp], -1)
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(inp))
        m = gumbel_softmax(qm, temp) if sample else F.softmax(qm, -1)
        w = F.softmax(sel(inp), -1); adv = (w * ADV).sum(-1)
        phi_mean = (phiprev + adv) % TWO_PI
        phi = (da.sample_von_mises(phi_mean, qpk) % TWO_PI) if sample else phi_mean
        if compute_kl:
            ppm = (phiprev + torch.exp(ltprev)) % TWO_PI
            ppk = F.softplus(model.prior_pk(pr[:, t]).squeeze(-1)) + 0.01
            klm = klm + kl_categorical(torch.log_softmax(qm, -1), model.meter_logp(mprev, phi, phiprev, pr[:, t]))
            klp = klp + kl_von_mises(phi_mean, qpk, ppm, ppk)
            klw = klw + (w * (torch.log(w + 1e-9) - math.log(1.0 / K_T))).sum(-1)
        phis.append(phi); mprev, phiprev, ltprev = m, phi_mean, torch.log(adv)
    return ((klm, klp, klw) if compute_kl else None), torch.stack(phis, 1)


@torch.no_grad()
def evaluate(model, sel, val, h_mode="real", frames=1600):
    model.eval(); sel.eval(); gb, gd, revs, bpm = [], [], [], []; n = len(val)
    for i, (hh, b, db) in enumerate(val):
        h_use = val[(i + 1) % n][0] if h_mode == "shuffle" else hh
        T = min(h_use.shape[0], b.shape[0], frames)
        h_in = torch.zeros(1, T, hh.shape[1], device=DEV) if h_mode == "zero" else h_use[:T].unsqueeze(0).to(DEV)
        z = torch.zeros(1, T, device=DEV)
        _, phis = rollout(model, sel, h_in, z, z, sample=False, compute_kl=False)
        phi = phis[0].cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS; dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2: gb.append(fmeas(ref, phase_beats(phi, M)))
        if len(dref) >= 2: gd.append(fmeas(dref, phase_downbeats(phi)))
        dphi = np.diff(phi); adv = np.where(dphi < -math.pi, dphi + TWO_PI, dphi)
        revs.append(float(np.sum(adv) / TWO_PI)); a2 = adv[adv > 1e-4]
        bpm.append(M * float(np.median(a2)) / TWO_PI * FPS * 60 if len(a2) else 0.0)
    model.train(); sel.train(); mn = lambda x: float(np.nanmean(x)) if x else float("nan")
    return mn(gb), mn(gd), mn(revs), mn(bpm)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--n_train", type=int, default=400); ap.add_argument("--n_val", type=int, default=40)
    a = ap.parse_args(); torch.manual_seed(0); np.random.seed(0); random.seed(0)
    train = load_pool("cache/acts/bt_train_rich", a.n_train, seed=1); val = load_pool("cache/acts/bt_val_rich", a.n_val, seed=2)
    nb = int(np.mean([(d > 0.5).sum() for _, _, d in val]))
    print(f"FAITHFUL v5 (SOFT TEMPO SELECTION, K={K_T}) | train={len(train)} val={len(val)} | GT #bars~{nb}", flush=True)
    model = BPVAE(h_dim=512, hidden=64).to(DEV)
    zf = 2 + 1 + 4; sel = nn.Linear(64 + zf, K_T).to(DEV)
    opt = torch.optim.Adam(list(model.parameters()) + list(sel.parameters()), lr=1e-3)
    pw = torch.tensor([8.0, 20.0], device=DEV)
    for step in range(1, a.steps + 1):
        temp = 1.0 + (0.3 - 1.0) * min(step / a.steps, 1.0)
        h, b, db = sample_batch(train, 256, 16)
        keep = (torch.rand(h.shape[0], 1, device=DEV) >= 0.5).float()
        (klm, klp, klw), phis = rollout(model, sel, h, b * keep, db * keep, temp, sample=True, compute_kl=True)
        recon = F.binary_cross_entropy_with_logits(geom_logits(phis), torch.stack([b, db], -1), pos_weight=pw, reduction="none").sum((1, 2))
        loss = (recon + klm.clamp(min=25.6) + klp.clamp(min=25.6) + 0.1 * klw).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(sel.parameters()), 5.0); opt.step()
        if step % 100 == 0 or step == a.steps:
            gb, gd, rv, bp = evaluate(model, sel, val, "real")
            print(f"  step {step:4d} | recon {float(recon.mean()):.1f} | GEOM beat {gb:.3f} db {gd:.3f} | phi-revs {rv:.1f}/{nb} | tempo {bp:.0f}BPM", flush=True)
    gb, gd, rv, bp = evaluate(model, sel, val, "real"); gbs, _, _, _ = evaluate(model, sel, val, "shuffle"); gbz, _, _, _ = evaluate(model, sel, val, "zero")
    print(f"\n--- FINAL (soft tempo selection) ---")
    print(f"  real {gb:.3f} db {gd:.3f} revs {rv:.1f}/{nb} tempo {bp:.0f}BPM | shuf {gbs:.3f} zero {gbz:.3f}")
    print("DONE")


if __name__ == "__main__":
    main()
