"""Do gradients flow the RIGHT way in faithful_v2 (phi = integral of data-informed tempo)?
Train a short model, then on val measure: (1) recon's gradient NORM on the log-tempo (vs old decoupled
~1.6), (2) DIRECTION correctness -- does -grad_recon push each song's tempo TOWARD its GT tempo?
"""
import sys, math, importlib.util
import numpy as np, torch, torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"; sys.path.insert(0, ROOT)
fv = importlib.util.spec_from_file_location("fv", f"{ROOT}/experiments/kvae_barpointer/faithful_v2.py")
v2 = importlib.util.module_from_spec(fv); fv.loader.exec_module(v2)
da = v2.da; BPVAE, load_pool, sample_batch = da.BPVAE, da.load_pool, da.sample_batch
kl_von_mises, kl_log_normal, kl_categorical = da.kl_von_mises, da.kl_log_normal, da.kl_categorical
soft_lt, LT_MIN, LT_MAX = v2.soft_lt, v2.LT_MIN, v2.LT_MAX
DEV = da.DEV; TWO_PI = 2*math.pi; FPS = 86.1328125; M = 4


def rollout_capture(model, h, b_in, db_in):
    """Deterministic-phase rollout; returns recon-logits, klt, klp, and per-frame qtm list."""
    B, T, _ = h.shape
    pc = model.enc_post(h, b_in, db_in); pr = model.enc_prior(h)
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, 0], z0], -1)))
    m = F.softmax(qm, -1); lt = soft_lt(qtm); phi = qpm % TWO_PI
    pm, ppm, ppk, ptm, pts = model.unpack(model.prior_init(pr.mean(1)))
    klt = kl_log_normal(qtm, qts, ptm, pts); klp = kl_von_mises(phi, qpk, ppm, ppk)
    zf = [model.zfeat(m, phi, lt)]; qtm_list = [qtm]; mprev, phiprev, ltprev = m, phi, lt
    for t in range(1, T):
        zp = model.zfeat(mprev, phiprev, ltprev)
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, t], zp], -1)))
        m = F.softmax(qm, -1); lt = soft_lt(qtm)
        phi_mean = (phiprev + torch.exp(lt)) % TWO_PI
        ppm = (phiprev + torch.exp(ltprev)) % TWO_PI
        ppk = F.softplus(model.prior_pk(pr[:, t]).squeeze(-1)) + 0.01
        ptm = ltprev; pts = F.softplus(model.prior_ts(pr[:, t]).squeeze(-1)) + 1e-3
        klt = klt + kl_log_normal(qtm, qts, ptm, pts); klp = klp + kl_von_mises(phi_mean, qpk, ppm, ppk)
        zf.append(model.zfeat(m, phi_mean, lt)); qtm_list.append(qtm); mprev, phiprev, ltprev = m, phi_mean, lt
    logits = torch.stack([model.decode(zf[t]) for t in range(T)], 1)
    return logits, klt, klp, qtm_list


def gnorm(loss, qtm_list):
    g = torch.autograd.grad(loss, qtm_list, retain_graph=True, allow_unused=True)
    g = [x if x is not None else torch.zeros_like(qtm_list[0]) for x in g]
    return torch.stack(g)                                          # [T,B]


def main():
    torch.manual_seed(0); np.random.seed(0)
    train = load_pool("cache/acts/bt_train_rich", 300, seed=1); val = load_pool("cache/acts/bt_val_rich", 16, seed=2)
    model = BPVAE(h_dim=512, hidden=64).to(DEV); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("training faithful_v2 for 300 steps to get a meaningful decoder ...", flush=True)
    for step in range(1, 301):
        h, b, db = sample_batch(train, 256, 16)
        loss, _, _ = v2.elbo(model, h, b, db, 0.5); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    model.train()
    # measure gradients on a val batch (deploy-like: beats hidden)
    hb = [v for v in val if v[0].shape[0] >= 256][:10]
    h = torch.stack([v[0][:256] for v in hb]).to(DEV); b = torch.stack([v[1][:256] for v in hb]).to(DEV); db = torch.stack([v[2][:256] for v in hb]).to(DEV)
    z = torch.zeros_like(b)
    logits, klt, klp, qtm_list = rollout_capture(model, h, z, z)            # deploy-like
    pw = torch.tensor([8.0, 20.0], device=DEV)
    recon = F.binary_cross_entropy_with_logits(logits, torch.stack([b, db], -1), pos_weight=pw, reduction="none").sum((1, 2)).mean()
    g_r = gnorm(recon, qtm_list); g_t = gnorm(klt.mean(), qtm_list); g_p = gnorm(klp.mean(), qtm_list)
    print(f"\ngrad NORM on log-tempo:  recon {float(g_r.norm()):.2f}  tempo-KL {float(g_t.norm()):.2f}  phase-KL {float(g_p.norm()):.2f}")
    print(f"  (old DECOUPLED model recon->tempo norm was ~1.6; coupling should make recon DOMINANT)")
    # DIRECTION: does -grad_recon push each song's tempo toward GT?
    with torch.no_grad():
        cur_bpm = (M * torch.exp(soft_lt(torch.stack(qtm_list))) / TWO_PI * FPS * 60).mean(0)   # [B]
    upd = -g_r.mean(0)                                                       # [B] update direction on qtm
    gt_bpm = []
    for v in hb:
        bf = np.where(v[1][:256].numpy() > 0.5)[0]
        gt_bpm.append(60 * FPS / np.median(np.diff(bf)) if len(bf) > 2 else np.nan)
    gt_bpm = torch.tensor(gt_bpm, device=DEV)
    err = gt_bpm - cur_bpm                                                   # +ve => need to speed up => want qtm up => upd>0
    valid = ~torch.isnan(err)
    aligned = ((torch.sign(upd[valid]) == torch.sign(err[valid])).float().mean())
    print(f"\nDIRECTION correctness (recon pushes tempo toward GT):  {float(aligned)*100:.0f}% of songs")
    print(f"  current tempo mean {float(cur_bpm[valid].mean()):.0f} BPM  vs GT {float(gt_bpm[valid].mean()):.0f} BPM")
    print(f"  per-song (cur, gt, update-sign): " + ", ".join(f"({float(cur_bpm[i]):.0f},{float(gt_bpm[i]):.0f},{'+' if float(upd[i])>0 else '-'})" for i in range(len(hb)) if valid[i])[:200])
    print("  >50% aligned => recon gradient flows the RIGHT way (toward true tempo)")


if __name__ == "__main__":
    main()
