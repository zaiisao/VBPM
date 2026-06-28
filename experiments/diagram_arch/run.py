"""The diagram architecture (authoritative spec): amortized encoder inference + latent-only decoder.

  Feature extractor -> h  (Beat-This rich [T,512], cached; strong so h carries the beats)
  Encoder q(z | h, b[train])      -> (m, phi, phidot)   -- reads h ALWAYS, b only in training
                                      (b-dropout: trained to also run on h ALONE)
  Bar-pointer prior p(z | h)      -> HYBRID: deterministic means + trainable kappa/sigma/meter (read h)
  Decoder p(b | z)                -> beats   -- LATENT-ONLY, never reads h
  DEPLOY: encoder(h, b=0) -> z -> beats   (no free-run; the encoder does the inference)

This tests whether the amortized encoder can produce a beat-retrievable z from h alone (the
conditional-VAE goal). Reports the h-only deploy (the diagram) vs teacher-forced (encoder given
the GT beats = the oracle ceiling), via the latent-only decoder and the geometric phase read-out.
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


class BPVAE(nn.Module):
    def __init__(self, h_dim, hidden=64, num_meters=4):
        super().__init__()
        self.K, self.hidden = num_meters, hidden
        self.zf = 3 + num_meters
        pdim = num_meters + 2 + 1 + 1 + 1
        self.post_gru = nn.GRU(h_dim + 2, hidden, batch_first=True, bidirectional=True)  # (h, beat, db)
        self.post_ctx = nn.Linear(2 * hidden, hidden)
        self.prior_gru = nn.GRU(h_dim, hidden, batch_first=True, bidirectional=True)     # h only
        self.prior_ctx = nn.Linear(2 * hidden, hidden)
        self.post_head = nn.Sequential(nn.Linear(hidden + self.zf, hidden), nn.Tanh(), nn.Linear(hidden, pdim))
        self.z0 = nn.Parameter(torch.zeros(self.zf))
        # --- hybrid prior: means deterministic (in rollout); these heads read h ---
        self.prior_init = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, pdim))
        self.prior_pk = nn.Linear(hidden, 1)                                  # kappa_phi = f(h)
        self.prior_ts = nn.Linear(hidden, 1)                                  # sigma_tau = f(h)
        self.meter_prior = nn.Sequential(nn.Linear(num_meters + 4 + hidden, hidden), nn.Tanh(),
                                         nn.Linear(hidden, num_meters * num_meters))
        # --- LATENT-ONLY decoder: input = z features ONLY (no h) ---
        self.decoder = nn.Sequential(nn.Linear(self.zf, hidden), nn.Tanh(), nn.Linear(hidden, 2))

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

    def decode(self, zf):
        return self.decoder(zf)                                              # [B, 2]  (latent-only)


def rollout(model, h, b_in, db_in, temp=0.5, sample=True, compute_kl=True):
    """Roll the ENCODER forward (posterior). sample=False uses posterior MEANS (deterministic, for
    inference). Returns (kl_tuple|None, phase_mu[T], decoder_logits[T,2])."""
    B, T, _ = h.shape
    pc = model.enc_post(h, b_in, db_in)
    pr = model.enc_prior(h) if compute_kl else None
    klm = klp = klt = (h.new_zeros(B) if compute_kl else None)
    zf, phase_mu = [], []

    def step_latents(qm, qpm, qpk, qtm, qts):
        if sample:
            m = gumbel_softmax(qm, temp); phi = sample_von_mises(qpm, qpk) % TWO_PI
            lt = qtm + qts * torch.randn_like(qtm)
        else:
            m = F.softmax(qm, -1); phi = qpm; lt = qtm           # posterior means
        return m, phi, lt

    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, 0], z0], -1)))
    m, phi, lt = step_latents(qm, qpm, qpk, qtm, qts)
    if compute_kl:
        pm, ppm, ppk, ptm, pts = model.unpack(model.prior_init(pr.mean(1)))
        klm = klm + kl_categorical(torch.log_softmax(qm, -1), torch.log_softmax(pm, -1))
        klp = klp + kl_von_mises(qpm, qpk, ppm, ppk); klt = klt + kl_log_normal(qtm, qts, ptm, pts)
    zf.append(model.zfeat(m, phi, lt)); phase_mu.append(qpm)
    mprev, phiprev, ltprev = m, phi, lt

    for t in range(1, T):
        zp = model.zfeat(mprev, phiprev, ltprev)
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, t], zp], -1)))
        m, phi, lt = step_latents(qm, qpm, qpk, qtm, qts)
        if compute_kl:
            ppm = (phiprev + torch.exp(ltprev)) % TWO_PI                      # deterministic mean
            ppk = F.softplus(model.prior_pk(pr[:, t]).squeeze(-1)) + 0.01     # kappa = f(h)
            ptm = ltprev; pts = F.softplus(model.prior_ts(pr[:, t]).squeeze(-1)) + 1e-3  # sigma = f(h)
            lpi = model.meter_logp(mprev, phi, phiprev, pr[:, t])
            klm = klm + kl_categorical(torch.log_softmax(qm, -1), lpi)
            klp = klp + kl_von_mises(qpm, qpk, ppm, ppk); klt = klt + kl_log_normal(qtm, qts, ptm, pts)
        zf.append(model.zfeat(m, phi, lt)); phase_mu.append(qpm)
        mprev, phiprev, ltprev = m, phi, lt

    logits = torch.stack([model.decode(zf[t]) for t in range(T)], 1)          # [B,T,2]
    kl = (klm, klp, klt) if compute_kl else None
    return kl, torch.stack(phase_mu, 1), logits


def elbo_loss(model, h, b, db, temp, pw_b, pw_db, fb, b_drop):
    B, T, _ = h.shape
    # b-dropout: per-sequence, hide the beats from the ENCODER (targets still used for BCE)
    keep = (torch.rand(B, 1, device=h.device) >= b_drop).float()
    (klm, klp, klt), _, logits = rollout(model, h, b * keep, db * keep, temp, sample=True, compute_kl=True)
    pw = torch.tensor([pw_b, pw_db], device=h.device)
    recon = F.binary_cross_entropy_with_logits(logits, torch.stack([b, db], -1), pos_weight=pw, reduction="none").sum((1, 2))
    klm = torch.clamp(klm, min=fb * T); klp = torch.clamp(klp, min=fb * T); klt = torch.clamp(klt, min=fb * T)
    loss = (recon + klm + klp + klt).mean()
    return loss, {"recon": float(recon.mean()), "klm": float(klm.mean()), "klp": float(klp.mean()), "klt": float(klt.mean())}


# ----- read-outs / metrics -----
def peaks(prob, thr=0.5, min_dist=0.10):
    p = np.asarray(prob); md = int(min_dist * FPS)
    cand = [t for t in range(1, len(p) - 1) if p[t] >= thr and p[t] >= p[t - 1] and p[t] >= p[t + 1]]
    out, last = [], -10 ** 9
    for t in cand:
        if t - last >= md: out.append(t); last = t
    return np.array(out, float) / FPS

def phase_beats(phase, m, min_dist=0.10):
    psi = (int(m) * np.asarray(phase, float)) % TWO_PI
    w = np.where(np.diff(psi) < -math.pi)[0] + 1
    out, last = [], -10 ** 9; md = int(min_dist * FPS)
    for t in w:
        if t - last >= md: out.append(t); last = t
    return np.array(out, float) / FPS

def phase_downbeats(phase, min_dist=0.30):
    w = np.where(np.diff(np.asarray(phase)) < -math.pi)[0] + 1
    out, last = [], -10 ** 9; md = int(min_dist * FPS)
    for t in w:
        if t - last >= md: out.append(t); last = t
    return np.array(out, float) / FPS

def fmeas(ref, est):
    ref, est = np.asarray(ref, float), np.asarray(est, float)
    if len(ref) == 0: return float("nan")
    if len(est) == 0: return 0.0
    return float(mir_eval.beat.f_measure(ref, est))


def load_pool(cache_dir, n, seed=0):
    files = sorted(glob.glob(f"{cache_dir}/*.pt")); random.Random(seed).shuffle(files)
    out = []
    for f in files[:n]:
        d = torch.load(f, map_location="cpu")
        h = d["activations"].float()
        if h.shape[0] < 400 or d["beat_targets"].sum() < 8: continue
        out.append((h, d["beat_targets"].float(), d["downbeat_targets"].float()))
    return out


@torch.no_grad()
def evaluate(model, songs, give_beats, max_frames=1600, h_mode="real"):
    """give_beats=False -> h-only deploy (the diagram). True -> teacher-forced (oracle ceiling).
    h_mode: 'real' | 'shuffle' (use ANOTHER song's h) | 'zero' (h=0) -- leak/artifact controls."""
    model.eval(); dec_b, dec_d, ph_b, ph_d = [], [], [], []
    n = len(songs)
    for i, (h, b, db) in enumerate(songs):
        h_use = songs[(i + 1) % n][0] if h_mode == "shuffle" else h    # mismatched audio
        T = min(h_use.shape[0], b.shape[0], max_frames)
        hh = (torch.zeros(1, T, h.shape[1], device=DEV) if h_mode == "zero"
              else h_use[:T].unsqueeze(0).to(DEV))
        bi = b[:T].unsqueeze(0).to(DEV) if give_beats else torch.zeros(1, T, device=DEV)
        di = db[:T].unsqueeze(0).to(DEV) if give_beats else torch.zeros(1, T, device=DEV)
        _, phase_mu, logits = rollout(model, hh, bi, di, sample=False, compute_kl=False)
        prob = torch.sigmoid(logits)[0].cpu().numpy(); pm = phase_mu[0].cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS
        dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2:
            dec_b.append(fmeas(ref, peaks(prob[:, 0]))); ph_b.append(fmeas(ref, phase_beats(pm, 4)))
        if len(dref) >= 2:
            dec_d.append(fmeas(dref, peaks(prob[:, 1], min_dist=0.30))); ph_d.append(fmeas(dref, phase_downbeats(pm)))
    model.train()
    mean = lambda x: float(np.nanmean(x)) if x else float("nan")
    return {"dec_beat": mean(dec_b), "dec_db": mean(dec_d), "phase_beat": mean(ph_b), "phase_db": mean(ph_d)}


def sample_batch(songs, frames, bs):
    hs, bs_, ds_ = [], [], []; tries = 0
    while len(hs) < bs and tries < bs * 50:
        tries += 1
        h, b, db = random.choice(songs)
        if h.shape[0] <= frames: continue
        s = random.randint(0, h.shape[0] - frames)
        if b[s:s + frames].sum() < 2: continue
        hs.append(h[s:s + frames]); bs_.append(b[s:s + frames]); ds_.append(db[s:s + frames])
    return torch.stack(hs).to(DEV), torch.stack(bs_).to(DEV), torch.stack(ds_).to(DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_dir", default="cache/acts/bt_train_rich")
    ap.add_argument("--val_dir", default="cache/acts/bt_val_rich")
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_val", type=int, default=50)
    ap.add_argument("--h_dim", type=int, default=512)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pw_b", type=float, default=8.0)
    ap.add_argument("--pw_db", type=float, default=20.0)
    ap.add_argument("--fb", type=float, default=0.1)
    ap.add_argument("--b_drop", type=float, default=0.5, help="prob of hiding beats from the encoder (per sequence)")
    ap.add_argument("--save", default="", help="if set, save {vae, h_dim} to this path")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

    print(f"loading train({a.n_train}) + val({a.n_val}) ...", flush=True)
    train = load_pool(a.pool_dir, a.n_train, seed=1); val = load_pool(a.val_dir, a.n_val, seed=2)
    print(f"  train={len(train)} val={len(val)} | h_dim={a.h_dim} b_drop={a.b_drop} | latent-only decoder, hybrid prior", flush=True)

    model = BPVAE(h_dim=a.h_dim, hidden=64).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    for step in range(1, a.steps + 1):
        temp = 1.0 + (0.3 - 1.0) * min(step / a.steps, 1.0)
        h, b, db = sample_batch(train, a.frames, a.bs)
        loss, info = elbo_loss(model, h, b, db, temp, a.pw_b, a.pw_db, a.fb, a.b_drop)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            ho = evaluate(model, val, give_beats=False)   # the diagram's deploy
            tf = evaluate(model, val, give_beats=True)     # oracle ceiling
            print(f"\nstep {step} | recon {info['recon']:.1f} | KL m/phi/tau "
                  f"{info['klm']:.2f}/{info['klp']:.2f}/{info['klt']:.2f}", flush=True)
            print(f"  H-ONLY  (diagram deploy): decoder beat {ho['dec_beat']:.3f} db {ho['dec_db']:.3f} | "
                  f"phase beat {ho['phase_beat']:.3f} db {ho['phase_db']:.3f}", flush=True)
            print(f"  TEACHER-FORCED (oracle) : decoder beat {tf['dec_beat']:.3f} db {tf['dec_db']:.3f} | "
                  f"phase beat {tf['phase_beat']:.3f} db {tf['phase_db']:.3f}", flush=True)
    # --- leak / artifact controls (does the h-only score actually depend on h?) ---
    real = evaluate(model, val, give_beats=False, h_mode="real")
    shuf = evaluate(model, val, give_beats=False, h_mode="shuffle")
    zero = evaluate(model, val, give_beats=False, h_mode="zero")
    print("\n--- LEAK CONTROLS (h-only deploy; decoder beat / db) ---")
    print(f"  real h     : beat {real['dec_beat']:.3f}  db {real['dec_db']:.3f}   <- the deploy number")
    print(f"  shuffled h : beat {shuf['dec_beat']:.3f}  db {shuf['dec_db']:.3f}   <- must COLLAPSE if genuine")
    print(f"  zero h     : beat {zero['dec_beat']:.3f}  db {zero['dec_db']:.3f}   <- must COLLAPSE if genuine")
    print("  shuffled/zero staying high => artifact (tempo-clustered generic grid); collapsing => real use of h")

    if a.save:
        import os as _os
        _os.makedirs(_os.path.dirname(a.save) or ".", exist_ok=True)
        torch.save({"vae": model.state_dict(), "h_dim": a.h_dim}, a.save)
        print(f"\n[saved] VAE -> {a.save}")

    print("\n=== diagram architecture: amortized encoder(h) -> z -> latent-only decoder ===")
    print("H-ONLY decoder-beat is the real deploy number; TEACHER-FORCED is the oracle ceiling.")
    print("ref points: amortized wall ~0.4 | DBN pipeline ~0.9 | Beat-This no-DBN ~0.88 (easy)")


if __name__ == "__main__":
    main()
