"""Semi-supervised self-training (structured EM) for the bar-pointer VAE.

QUESTION: can UNLABELED audio + the generative bar-pointer structure reduce the labels
needed for beat / downbeat tracking, vs supervised-only?

SUBSTRATE: the faithful bar-pointer VAE (corrected von Mises sampler) on rich Beat-This
[T,512] features (a strong observation, so the model produces real beats and the test is
fair). DEVIATIONS from the strict paper (flagged, needed so self-training has signal):
  - decoder outputs 2 logits (beat, downbeat) instead of 1
  - pos_weight BCE (beats are ~1.5% of frames) + free-bits KL floor (anti-collapse)
These are method deviations; this is a separate exploratory experiment, not the strict notebook.

DESIGN: split the train pool into labeled (fraction f) + unlabeled (rest); held-out val.
  SUP      : train supervised (BCE+KL) on the labeled fraction only.
  SELF-TRAIN: warm up supervised on labeled, then EM rounds — pseudo-label the unlabeled
              songs with the model's own decoder peaks, retrain on labeled(true)+unlabeled(pseudo).
  Both conditions get the SAME number of gradient steps (fair compute).
Metric: val beat-F / downbeat-F (mir_eval, 70 ms), decoder read-out. Plus pseudo-label quality
(pseudo vs the held-back true labels of the "unlabeled" songs) as the key diagnostic.
"""
import sys, glob, math, random, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import mir_eval

sys.path.insert(0, "/home/sogang/jaehoon/CHART")
from faithful.distributions import (TWO_PI, gumbel_softmax, sample_von_mises,
                                    kl_categorical, kl_von_mises, kl_log_normal)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FPS = 86.1328125


# ----------------------------- model (2-output decoder) ----------------------------- #
class BPVAE(nn.Module):
    def __init__(self, h_dim, hidden=64, num_meters=4):
        super().__init__()
        self.K, self.hidden = num_meters, hidden
        self.zf = 3 + num_meters
        pdim = num_meters + 2 + 1 + 1 + 1
        self.post_gru = nn.GRU(h_dim + 2, hidden, batch_first=True, bidirectional=True)  # reads (h, beat, db)
        self.post_ctx = nn.Linear(2 * hidden, hidden)
        self.prior_gru = nn.GRU(h_dim, hidden, batch_first=True, bidirectional=True)
        self.prior_ctx = nn.Linear(2 * hidden, hidden)
        self.post_head = nn.Sequential(nn.Linear(hidden + self.zf, hidden), nn.Tanh(), nn.Linear(hidden, pdim))
        self.z0 = nn.Parameter(torch.zeros(self.zf))
        self.prior_init = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, pdim))
        self.prior_pk = nn.Linear(hidden, 1)
        self.prior_ts = nn.Linear(hidden, 1)
        self.meter_prior = nn.Sequential(nn.Linear(num_meters + 4 + hidden, hidden), nn.Tanh(),
                                         nn.Linear(hidden, num_meters * num_meters))
        self.decoder = nn.Sequential(nn.Linear(self.zf + hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def enc_post(self, h, b, db):
        out, _ = self.post_gru(torch.cat([h, b.unsqueeze(-1), db.unsqueeze(-1)], -1))
        return torch.tanh(self.post_ctx(out))

    def enc_prior(self, h):
        out, _ = self.prior_gru(h)
        return torch.tanh(self.prior_ctx(out))

    def unpack(self, v):
        K = self.K
        return (v[:, :K], torch.atan2(v[:, K + 1], v[:, K]) % TWO_PI,
                F.softplus(v[:, K + 2]) + 0.01, v[:, K + 3], F.softplus(v[:, K + 4]) + 1e-3)

    def zfeat(self, m, phi, lt):
        return torch.cat([torch.cos(phi).unsqueeze(-1), torch.sin(phi).unsqueeze(-1), lt.unsqueeze(-1), m], -1)

    def meter_logp(self, mprev, phi, phiprev, ctx):
        feats = torch.cat([mprev, torch.cos(phi).unsqueeze(-1), torch.sin(phi).unsqueeze(-1),
                           torch.cos(phiprev).unsqueeze(-1), torch.sin(phiprev).unsqueeze(-1), ctx], -1)
        Pi = F.softmax(self.meter_prior(feats).reshape(-1, self.K, self.K), 2)
        return torch.log(torch.bmm(mprev.unsqueeze(1), Pi).squeeze(1) + 1e-9)

    def decode(self, zf, ctx):
        return self.decoder(torch.cat([zf, ctx], -1))          # [B, 2]


# ----------------------------- ELBO loss (+pos_weight, +free-bits) ----------------------------- #
def elbo_loss(model, h, b, db, temp, pw_b, pw_db, fb):
    B, T, _ = h.shape
    pc = model.enc_post(h, b, db); pr = model.enc_prior(h)
    klm = h.new_zeros(B); klp = h.new_zeros(B); klt = h.new_zeros(B); zf = []
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, 0], z0], -1)))
    pm, ppm, ppk, ptm, pts = model.unpack(model.prior_init(pr.mean(1)))
    m = gumbel_softmax(qm, temp); phi = sample_von_mises(qpm, qpk) % TWO_PI; lt = qtm + qts * torch.randn_like(qtm)
    klm = klm + kl_categorical(torch.log_softmax(qm, -1), torch.log_softmax(pm, -1))
    klp = klp + kl_von_mises(qpm, qpk, ppm, ppk); klt = klt + kl_log_normal(qtm, qts, ptm, pts)
    zf.append(model.zfeat(m, phi, lt)); mprev, phiprev, ltprev = m, phi, lt
    for t in range(1, T):
        zp = model.zfeat(mprev, phiprev, ltprev)
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, t], zp], -1)))
        ppm = (phiprev + torch.exp(ltprev)) % TWO_PI
        ppk = F.softplus(model.prior_pk(pr[:, t]).squeeze(-1)) + 0.01
        ptm = ltprev; pts = F.softplus(model.prior_ts(pr[:, t]).squeeze(-1)) + 1e-3
        m = gumbel_softmax(qm, temp); phi = sample_von_mises(qpm, qpk) % TWO_PI; lt = qtm + qts * torch.randn_like(qtm)
        lpi = model.meter_logp(mprev, phi, phiprev, pr[:, t])
        klm = klm + kl_categorical(torch.log_softmax(qm, -1), lpi)
        klp = klp + kl_von_mises(qpm, qpk, ppm, ppk); klt = klt + kl_log_normal(qtm, qts, ptm, pts)
        zf.append(model.zfeat(m, phi, lt)); mprev, phiprev, ltprev = m, phi, lt
    logits = torch.stack([model.decode(zf[t], pr[:, t]) for t in range(T)], 1)   # [B,T,2]
    pw = torch.tensor([pw_b, pw_db], device=h.device)
    tgt = torch.stack([b, db], -1)
    recon = F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pw, reduction="none").sum((1, 2))
    # free-bits floor per latent (summed over T): max(KL, fb*T)
    klm = torch.clamp(klm, min=fb * T); klp = torch.clamp(klp, min=fb * T); klt = torch.clamp(klt, min=fb * T)
    loss = (recon + klm + klp + klt).mean()
    return loss, {"recon": float(recon.mean()), "klm": float(klm.mean()), "klp": float(klp.mean()), "klt": float(klt.mean())}


@torch.no_grad()
def free_run(model, h, temp=0.3):
    B, T, _ = h.shape
    pr = model.enc_prior(h)
    pm, ppm, ppk, ptm, pts = model.unpack(model.prior_init(pr.mean(1)))
    m = gumbel_softmax(pm, temp); phi = sample_von_mises(ppm, ppk) % TWO_PI; lt = ptm + pts * torch.randn_like(ptm)
    zf = [model.zfeat(m, phi, lt)]; mprev, phiprev, ltprev = m, phi, lt
    for t in range(1, T):
        ppm = (phiprev + torch.exp(ltprev)) % TWO_PI
        ppk = F.softplus(model.prior_pk(pr[:, t]).squeeze(-1)) + 0.01
        ptm = ltprev; pts = F.softplus(model.prior_ts(pr[:, t]).squeeze(-1)) + 1e-3
        phi = sample_von_mises(ppm, ppk) % TWO_PI; lt = ptm + pts * torch.randn_like(ptm)
        m = gumbel_softmax(model.meter_logp(mprev, phi, phiprev, pr[:, t]), temp)
        zf.append(model.zfeat(m, phi, lt)); mprev, phiprev, ltprev = m, phi, lt
    logits = torch.stack([model.decode(zf[t], pr[:, t]) for t in range(T)], 1)
    return torch.sigmoid(logits)                                   # [B,T,2] (beat, downbeat)


# ----------------------------- data ----------------------------- #
def load_pool(cache_dir, n, seed=0):
    files = sorted(glob.glob(f"{cache_dir}/*.pt")); random.Random(seed).shuffle(files)
    out = []
    for f in files[:n]:
        d = torch.load(f, map_location="cpu")
        h = d["activations"].float()
        if h.shape[0] < 400 or d["beat_targets"].sum() < 8:
            continue
        out.append((h, d["beat_targets"].float(), d["downbeat_targets"].float()))
    return out


# ----------------------------- read-out + metrics ----------------------------- #
def peaks(prob, thr=0.5, min_dist=0.10):
    p = np.asarray(prob); md = int(min_dist * FPS)
    cand = [t for t in range(1, len(p) - 1) if p[t] >= thr and p[t] >= p[t - 1] and p[t] >= p[t + 1]]
    out, last = [], -10 ** 9
    for t in cand:
        if t - last >= md:
            out.append(t); last = t
    return np.array(out, float) / FPS

def fmeas(ref, est):
    ref, est = np.asarray(ref, float), np.asarray(est, float)
    if len(ref) == 0: return float("nan")
    if len(est) == 0: return 0.0
    return float(mir_eval.beat.f_measure(ref, est))

@torch.no_grad()
def evaluate(model, songs, max_frames=1600):
    model.eval(); bF, dF = [], []
    for h, b, db in songs:
        T = min(h.shape[0], max_frames)
        prob = free_run(model, h[:T].unsqueeze(0).to(DEV))[0].cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS
        dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2: bF.append(fmeas(ref, peaks(prob[:, 0])))
        if len(dref) >= 2: dF.append(fmeas(dref, peaks(prob[:, 1], min_dist=0.30)))
    model.train()
    return float(np.nanmean(bF)), float(np.nanmean(dF))


# ----------------------------- pseudo-labels (E-step) ----------------------------- #
def _periodic_cv(times):
    """Coefficient of variation of inter-beat intervals — a label-free regularity score.
    Low CV => the model laid down a steady (structure-consistent) grid we can trust."""
    if len(times) < 4: return 9.9
    ibi = np.diff(times)
    return float(np.std(ibi) / (np.mean(ibi) + 1e-9))

@torch.no_grad()
def pseudo_targets(model, songs, max_frames=1600, cv_thresh=0.35):
    """For each unlabeled song, free-run -> peak-pick decoder beat/downbeat -> hard frame targets.
    CONFIDENCE FILTER: keep only songs whose pseudo-beats are periodically consistent (IBI CV <
    cv_thresh) -- the bar-pointer structure's self-check, so we don't train on garbage.
    Returns (kept_targets, pseudo_vs_true_beatF, pseudo_vs_true_dbF, frac_kept)."""
    model.eval(); out, qb, qd, nkept = [], [], [], 0
    for h, b_true, db_true in songs:
        T = min(h.shape[0], max_frames)
        prob = free_run(model, h[:T].unsqueeze(0).to(DEV))[0].cpu().numpy()
        bt = peaks(prob[:, 0]); dt = peaks(prob[:, 1], min_dist=0.30)
        if _periodic_cv(bt) > cv_thresh:                 # drop non-regular (low-confidence) songs
            continue
        nkept += 1
        pb = torch.zeros(T); pb[np.clip((bt * FPS).astype(int), 0, T - 1)] = 1.0
        pd = torch.zeros(T); pd[np.clip((dt * FPS).astype(int), 0, T - 1)] = 1.0
        out.append((h[:T], pb, pd))
        rb = np.where(b_true.numpy()[:T] > 0.5)[0] / FPS; rd = np.where(db_true.numpy()[:T] > 0.5)[0] / FPS
        if len(rb) >= 2: qb.append(fmeas(rb, bt))
        if len(rd) >= 2: qd.append(fmeas(rd, dt))
    model.train()
    qb = float(np.nanmean(qb)) if qb else float("nan")
    qd = float(np.nanmean(qd)) if qd else float("nan")
    return out, qb, qd, nkept / max(len(songs), 1)


# ----------------------------- training ----------------------------- #
def sample_batch(songs, frames, bs):
    hs, bs_, ds_ = [], [], []
    tries = 0
    while len(hs) < bs and tries < bs * 50:
        tries += 1
        h, b, db = random.choice(songs)
        if h.shape[0] <= frames: continue
        s = random.randint(0, h.shape[0] - frames)
        if b[s:s + frames].sum() < 2: continue
        hs.append(h[s:s + frames]); bs_.append(b[s:s + frames]); ds_.append(db[s:s + frames])
    return (torch.stack(hs).to(DEV), torch.stack(bs_).to(DEV), torch.stack(ds_).to(DEV))

def train(model, songs, steps, frames, bs, lr, pw_b, pw_db, fb, opt=None):
    opt = opt or torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        temp = 1.0 + (0.3 - 1.0) * min(step / max(steps, 1), 1.0)
        h, b, db = sample_batch(songs, frames, bs)
        loss, info = elbo_loss(model, h, b, db, temp, pw_b, pw_db, fb)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    return opt, info


def run_condition(name, labeled, unlabeled, val, args, self_train):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    model = BPVAE(h_dim=args.h_dim, hidden=64).to(DEV)
    if not self_train:
        train(model, labeled, args.total_steps, args.frames, args.bs, args.lr, args.pw_b, args.pw_db, args.fb)
        bF, dF = evaluate(model, val)
        return {"cond": name, "beatF": bF, "dbF": dF, "pseudo_bF": None, "pseudo_dF": None}
    # self-train: supervised warmup, then EM rounds on labeled+pseudo
    opt, _ = train(model, labeled, args.warmup, args.frames, args.bs, args.lr, args.pw_b, args.pw_db, args.fb)
    pq_b = pq_d = kept = float("nan")
    pool_for_pseudo = unlabeled[:args.max_unlabeled]
    per_round = (args.total_steps - args.warmup) // args.em_rounds
    for r in range(args.em_rounds):
        pseudo, pq_b, pq_d, kept = pseudo_targets(model, pool_for_pseudo, cv_thresh=args.cv_thresh)
        combined = labeled + pseudo            # true labels + pseudo labels
        print(f"    EM round {r}: kept {kept*100:.0f}% of {len(pool_for_pseudo)} unlabeled "
              f"(pseudo-F vs true: beat {pq_b:.3f}, db {pq_d:.3f})", flush=True)
        train(model, combined, per_round, args.frames, args.bs, args.lr, args.pw_b, args.pw_db, args.fb, opt=opt)
    bF, dF = evaluate(model, val)
    return {"cond": name, "beatF": bF, "dbF": dF, "pseudo_bF": pq_b, "pseudo_dF": pq_d, "kept": kept}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_dir", default="cache/acts/bt_train_rich")
    ap.add_argument("--val_dir", default="cache/acts/bt_val_rich")
    ap.add_argument("--n_pool", type=int, default=300)
    ap.add_argument("--n_val", type=int, default=50)
    ap.add_argument("--h_dim", type=int, default=512)
    ap.add_argument("--fracs", default="0.05,0.15,0.5")
    ap.add_argument("--total_steps", type=int, default=450)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--em_rounds", type=int, default=2)
    ap.add_argument("--frames", type=int, default=320)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pw_b", type=float, default=8.0)
    ap.add_argument("--pw_db", type=float, default=20.0)
    ap.add_argument("--fb", type=float, default=0.1)
    ap.add_argument("--max_unlabeled", type=int, default=120, help="cap #unlabeled songs used for pseudo-labels (speed)")
    ap.add_argument("--cv_thresh", type=float, default=0.35, help="keep pseudo songs with IBI CV below this (confidence)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"loading pool ({args.n_pool}) + val ({args.n_val}) ...", flush=True)
    pool = load_pool(args.pool_dir, args.n_pool, seed=1)
    val = load_pool(args.val_dir, args.n_val, seed=2)
    print(f"  pool={len(pool)} val={len(val)} | h_dim={args.h_dim} | "
          f"total_steps={args.total_steps} (warmup {args.warmup}, {args.em_rounds} EM rounds)", flush=True)

    rows = []
    for f in [float(x) for x in args.fracs.split(",")]:
        nlab = max(1, int(round(f * len(pool))))
        labeled = pool[:nlab]; unlabeled = pool[nlab:]
        print(f"\n=== label_frac={f}  (labeled={len(labeled)}, unlabeled={len(unlabeled)}) ===", flush=True)
        sup = run_condition("SUP", labeled, unlabeled, val, args, self_train=False)
        print(f"  SUP        : beatF={sup['beatF']:.3f}  dbF={sup['dbF']:.3f}", flush=True)
        if len(unlabeled) >= 5:
            st = run_condition("SELF", labeled, unlabeled, val, args, self_train=True)
            print(f"  SELF-TRAIN : beatF={st['beatF']:.3f}  dbF={st['dbF']:.3f}  "
                  f"(pseudo-label quality vs true: beat {st['pseudo_bF']:.3f}, db {st['pseudo_dF']:.3f})", flush=True)
            rows.append((f, len(labeled), sup, st))
        else:
            rows.append((f, len(labeled), sup, None))

    print("\n" + "=" * 78)
    print(f"{'frac':>6} {'nlab':>5} | {'SUP beatF':>10} {'SELF beatF':>11} {'dBeat':>7} | "
          f"{'SUP dbF':>8} {'SELF dbF':>9} {'dDb':>7} | {'pseudoB':>8} {'pseudoD':>8}")
    for f, nlab, sup, st in rows:
        if st is None:
            print(f"{f:>6} {nlab:>5} | {sup['beatF']:>10.3f} {'-':>11} {'-':>7} | {sup['dbF']:>8.3f} {'-':>9} {'-':>7}")
        else:
            print(f"{f:>6} {nlab:>5} | {sup['beatF']:>10.3f} {st['beatF']:>11.3f} {st['beatF']-sup['beatF']:>+7.3f} | "
                  f"{sup['dbF']:>8.3f} {st['dbF']:>9.3f} {st['dbF']-sup['dbF']:>+7.3f} | "
                  f"{st['pseudo_bF']:>8.3f} {st['pseudo_dF']:>8.3f}")
    print("\nSELF > SUP at low frac => unlabeled audio + structure earns its keep. pseudoB/D = pseudo-label F vs true.")


if __name__ == "__main__":
    main()
