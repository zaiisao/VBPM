"""Why does the tempo converge to ~0? Replicate the rollout, capture the per-frame log-tempo (qtm), and
measure the gradient each loss term sends to it: reconstruction vs tempo-KL vs phase-KL. If recon's pull
on tempo is ~0 and the KL dominates (and points down), that's the collapse mechanism.
"""
import sys, math, importlib.util
import numpy as np, torch, torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"; sys.path.insert(0, ROOT)
s = importlib.util.spec_from_file_location("da", f"{ROOT}/experiments/diagram_arch/run.py")
da = importlib.util.module_from_spec(s); s.loader.exec_module(da)
BPVAE, load_pool = da.BPVAE, da.load_pool
kl_von_mises, kl_log_normal = da.kl_von_mises, da.kl_log_normal
DEV = da.DEV; TWO_PI = 2*math.pi; FPS = 86.1328125; M = 4


def rollout_capture(model, h, b_in, db_in):
    """Deterministic rollout; returns recon, klt(sum), klp(sum), and the list of per-frame qtm tensors."""
    B, T, _ = h.shape
    pc = model.enc_post(h, b_in, db_in); pr = model.enc_prior(h)
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, 0], z0], -1)))
    m = F.softmax(qm, -1); phi = qpm; lt = qtm
    pm, ppm, ppk, ptm, pts = model.unpack(model.prior_init(pr.mean(1)))
    klt = kl_log_normal(qtm, qts, ptm, pts); klp = kl_von_mises(qpm, qpk, ppm, ppk)
    zf = [model.zfeat(m, phi, lt)]; qtm_list = [qtm]; mprev, phiprev, ltprev = m, phi, lt
    for t in range(1, T):
        zp = model.zfeat(mprev, phiprev, ltprev)
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, t], zp], -1)))
        m = F.softmax(qm, -1); phi = qpm; lt = qtm
        ppm = (phiprev + torch.exp(ltprev)) % TWO_PI
        ppk = F.softplus(model.prior_pk(pr[:, t]).squeeze(-1)) + 0.01
        ptm = ltprev; pts = F.softplus(model.prior_ts(pr[:, t]).squeeze(-1)) + 1e-3
        klt = klt + kl_log_normal(qtm, qts, ptm, pts); klp = klp + kl_von_mises(qpm, qpk, ppm, ppk)
        zf.append(model.zfeat(m, phi, lt)); qtm_list.append(qtm)
        mprev, phiprev, ltprev = m, phi, lt
    logits = torch.stack([model.decode(zf[t]) for t in range(T)], 1)
    return logits, klt, klp, qtm_list


def analyze(tag, model, h, b, db):
    pw = torch.tensor([8.0, 20.0], device=DEV)
    logits, klt, klp, qtm_list = rollout_capture(model, h, b, db)
    recon = F.binary_cross_entropy_with_logits(logits, torch.stack([b, db], -1), pos_weight=pw, reduction="none").sum((1, 2)).mean()
    klt_m = klt.mean(); klp_m = klp.mean()
    def gnorm_dir(loss):
        g = torch.autograd.grad(loss, qtm_list, retain_graph=True, allow_unused=True)
        g = [x if x is not None else torch.zeros_like(qtm_list[0]) for x in g]
        G = torch.stack(g)                                  # [T,B]
        return float(G.norm()), float(G.mean())             # norm, signed mean (down if <0)
    rn, rd = gnorm_dir(recon); tn, td = gnorm_dir(klt_m); pn, pd = gnorm_dir(klp_m)
    with torch.no_grad():
        bpm = M * float(torch.exp(torch.stack(qtm_list)).mean()) / TWO_PI * FPS * 60
    print(f"[{tag}] tempo~{bpm:.0f}BPM | grad on log-tempo (norm, signed-mean[neg=pushes tempo DOWN]):")
    print(f"    recon    : norm {rn:8.3f}  dir {rd:+.4f}")
    print(f"    tempo-KL : norm {tn:8.3f}  dir {td:+.4f}")
    print(f"    phase-KL : norm {pn:8.3f}  dir {pd:+.4f}")
    print(f"    -> KL/recon gradient ratio on tempo = {(tn+pn)/max(rn,1e-9):.1f}x", flush=True)


def main():
    val = load_pool("cache/acts/bt_val_rich", 16, seed=2)
    h = torch.stack([v[0][:256] for v in val if v[0].shape[0] >= 256][:8]).to(DEV)
    b = torch.stack([v[1][:256] for v in val if v[0].shape[0] >= 256][:8]).to(DEV)
    db = torch.stack([v[2][:256] for v in val if v[0].shape[0] >= 256][:8]).to(DEV)
    z = torch.zeros_like(b)
    print(f"batch {tuple(h.shape)}\n")
    # trained checkpoint
    d = torch.load("checkpoints/diagram_rerun.pt", map_location=DEV)
    mt = BPVAE(h_dim=d["h_dim"], hidden=64).to(DEV); mt.load_state_dict(d["vae"]); mt.train()
    print("=== TRAINED checkpoint (deploy-like: beats hidden b=0) ==="); analyze("trained/h-only", mt, h, z, z)
    print("=== TRAINED checkpoint (teacher-forced: beats given) ==="); analyze("trained/TF", mt, h, b, db)
    # fresh init
    torch.manual_seed(0); mf = BPVAE(h_dim=d["h_dim"], hidden=64).to(DEV); mf.train()
    print("=== FRESH init ==="); analyze("fresh", mf, h, b, db)


if __name__ == "__main__":
    main()
