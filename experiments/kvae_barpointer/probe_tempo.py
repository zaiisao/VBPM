"""WHY doesn't decoder beat-prediction translate to a tempo latent? Dissect the trained CVAE:
1) does the decoder OUTPUT carry tempo (inter-beat spacing)?  2) does the tempo LATENT (exp lt)?
3) does phi's RATE (median dphi)?  4) is tempo linearly PROBE-able from the latent z (ridge R^2)?
5) is phi a LOCAL beat-flag (corr with per-frame beats) vs a rotating phase (corr with bar ramp)?
"""
import sys, math, importlib.util
import numpy as np, torch, torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"; sys.path.insert(0, ROOT)
s = importlib.util.spec_from_file_location("da", f"{ROOT}/experiments/diagram_arch/run.py")
da = importlib.util.module_from_spec(s); s.loader.exec_module(da)
BPVAE, load_pool, peaks = da.BPVAE, da.load_pool, da.peaks
DEV = da.DEV; TWO_PI = 2*math.pi; FPS = 86.1328125; M = 4


def capture(model, h):
    """Deterministic deploy rollout; return per-frame phi, lt, zfeat, decoder beat-prob."""
    B, T, _ = h.shape; pc = model.enc_post(h, torch.zeros(B, T, device=DEV), torch.zeros(B, T, device=DEV))
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, 0], z0], -1)))
    m = F.softmax(qm, -1); phi = qpm; lt = qtm
    zf = [model.zfeat(m, phi, lt)]; phis = [phi]; lts = [lt]; mprev, phiprev, ltprev = m, phi, lt
    for t in range(1, T):
        zp = model.zfeat(mprev, phiprev, ltprev)
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, t], zp], -1)))
        m = F.softmax(qm, -1); phi = qpm; lt = qtm
        zf.append(model.zfeat(m, phi, lt)); phis.append(phi); lts.append(lt); mprev, phiprev, ltprev = m, phi, lt
    ZF = torch.stack(zf, 1); logits = torch.stack([model.decode(ZF[:, t]) for t in range(T)], 1)
    return torch.stack(phis, 1)[0], torch.stack(lts, 1)[0], ZF[0], torch.sigmoid(logits)[0]


def main():
    d = torch.load("checkpoints/diagram_rerun.pt", map_location=DEV)
    model = BPVAE(h_dim=d["h_dim"], hidden=64).to(DEV); model.load_state_dict(d["vae"]); model.eval()
    val = load_pool("cache/acts/bt_val_rich", 40, seed=2)
    out_t, lat_t, rate_t, gt_t, zf_feats, phi_beatcorr = [], [], [], [], [], []
    with torch.no_grad():
        for hh, b, db in val:
            T = min(hh.shape[0], b.shape[0], 1600)
            phi, lt, zf, prob = capture(model, hh[:T].unsqueeze(0).to(DEV))
            phi = phi.cpu().numpy(); lt = lt.cpu().numpy(); prob = prob.cpu().numpy(); zf = zf.cpu().numpy()
            bt = b.numpy()[:T]; bf = np.where(bt > 0.5)[0]
            if len(bf) < 3: continue
            gt = 60 * FPS / np.median(np.diff(bf)); gt_t.append(gt)
            # 1) decoder OUTPUT tempo (from predicted beats)
            est = (peaks(prob[:, 0]) * FPS).astype(int)
            out_t.append(60 * FPS / np.median(np.diff(est)) if len(est) > 2 else np.nan)
            # 2) tempo LATENT
            lat_t.append(M * float(np.exp(lt).mean()) / TWO_PI * FPS * 60)
            # 3) phi RATE
            dphi = np.diff(phi); adv = np.where(dphi < -math.pi, dphi + TWO_PI, dphi); adv = adv[adv > 1e-4]
            rate_t.append(M * float(np.median(adv)) / TWO_PI * FPS * 60 if len(adv) else 0.0)
            # 4) probe features: per-song mean latent
            zf_feats.append(zf.mean(0))
            # 5) phi as local beat-flag: corr(decoder-relevant phi feature, beat target)
            #    use cos(phi) and cos(m*phi) as candidate per-frame features
            f = np.cos(M * phi); bd = bt.astype(float)
            phi_beatcorr.append(np.corrcoef(f, bd)[0, 1] if f.std() > 1e-6 else 0.0)
    gt = np.array(gt_t)
    def corr(x):
        x = np.array(x); ok = np.isfinite(x) & np.isfinite(gt)
        return np.corrcoef(x[ok], gt[ok])[0, 1] if ok.sum() > 2 else float("nan")
    # ridge probe latent -> tempo
    from numpy.linalg import lstsq
    X = np.array(zf_feats); Xn = (X - X.mean(0)) / (X.std(0) + 1e-6); Xn = np.hstack([Xn, np.ones((len(Xn), 1))])
    w, *_ = lstsq(Xn, gt, rcond=None); pred = Xn @ w
    r2 = 1 - ((gt - pred) ** 2).sum() / ((gt - gt.mean()) ** 2).sum()
    print(f"n={len(gt)} songs | GT tempo mean {gt.mean():.0f} BPM")
    print(f"1) decoder OUTPUT tempo  vs GT : corr {corr(out_t):+.2f}   (mean {np.nanmean(out_t):.0f} BPM)  <- beats carry tempo?")
    print(f"2) tempo LATENT exp(lt)  vs GT : corr {corr(lat_t):+.2f}   (mean {np.nanmean(lat_t):.0f} BPM)  <- is rate in the latent?")
    print(f"3) phi RATE median-dphi  vs GT : corr {corr(rate_t):+.2f}   (mean {np.nanmean(rate_t):.0f} BPM)")
    print(f"4) PROBE latent->tempo (ridge R^2 over zfeat) : {r2:+.2f}   <- is tempo info anywhere in the latent?")
    print(f"5) phi as local beat-flag: mean corr(cos(m*phi), beat) = {np.nanmean(phi_beatcorr):+.2f}  (high => phi is a per-frame beat feature)")
    print("\nIF (1) high but (2),(3),(4) low => decoder predicts beats per-frame WITHOUT a rate representation:")
    print("   beats<->tempo equivalence holds at the OUTPUT (sequentially), but the per-frame objective never induces a tempo latent.")


if __name__ == "__main__":
    main()
