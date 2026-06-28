"""HYPOTHESIS TEST (not a fix): clamp the latent phase to a GT rotating ramp and see if the rest
"follows." If the decoder then reads beats from the rotating phi AND the tempo posterior converges
to the true beat rate (instead of collapsing to ~0), then the model is FINE given rotation — the only
failure in free training is that phi doesn't rotate. We do NOT train the phase head here (phi is
imposed); this isolates "is rotation the only missing piece?".
"""
import sys, glob, math, random, importlib.util
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/CHART")
s = importlib.util.spec_from_file_location("da", "experiments/diagram_arch/run.py"); da = importlib.util.module_from_spec(s); s.loader.exec_module(da)
from faithful.distributions import gumbel_softmax, kl_categorical, kl_von_mises, kl_log_normal
BPVAE, peaks, fmeas, phase_beats, phase_downbeats = da.BPVAE, da.peaks, da.fmeas, da.phase_beats, da.phase_downbeats
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TWO_PI = 2 * math.pi; FPS = 86.1328125


def gt_barphase(beat_fr, m, T):
    phi = np.zeros(T)
    if len(beat_fr) < 2: return phi
    vals = np.arange(len(beat_fr)) * (TWO_PI / m)
    for k in range(len(beat_fr) - 1):
        a, b = beat_fr[k], beat_fr[k + 1]
        phi[a:b] = np.linspace(vals[k], vals[k + 1], b - a, endpoint=False)
    phi[beat_fr[-1]:] = vals[-1]
    return phi % TWO_PI


def clamp_rollout(model, h, gt_phi, temp=0.5):
    """phi is CLAMPED to gt_phi (per frame). tempo/meter come from the encoder (free). Returns
    (KLs, decoder logits, tempo_mu trajectory)."""
    B, T, _ = h.shape
    z = torch.zeros(B, T, device=h.device)
    pc = model.enc_post(h, z, z)             # h-only (no beats to encoder)
    pr = model.enc_prior(h)
    klm = klp = klt = h.new_zeros(B); zf = []; tempo_mu = []
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, 0], z0], -1)))
    m = gumbel_softmax(qm, temp); phi = gt_phi[:, 0]; lt = qtm + qts * torch.randn_like(qtm)
    pm0, ppm, ppk, ptm, pts = model.unpack(model.prior_init(pr.mean(1)))
    klm = klm + kl_categorical(torch.log_softmax(qm, -1), torch.log_softmax(pm0, -1))
    klp = klp + kl_von_mises(phi, qpk, ppm, ppk); klt = klt + kl_log_normal(qtm, qts, ptm, pts)
    zf.append(model.zfeat(m, phi, lt)); tempo_mu.append(qtm); mprev, phiprev, ltprev = m, phi, lt
    for t in range(1, T):
        zp = model.zfeat(mprev, phiprev, ltprev)
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, t], zp], -1)))
        m = gumbel_softmax(qm, temp); phi = gt_phi[:, t]; lt = qtm + qts * torch.randn_like(qtm)
        ppm = (phiprev + torch.exp(ltprev)) % TWO_PI
        ppk = F.softplus(model.prior_pk(pr[:, t]).squeeze(-1)) + 0.01
        ptm = ltprev; pts = F.softplus(model.prior_ts(pr[:, t]).squeeze(-1)) + 1e-3
        klm = klm + kl_categorical(torch.log_softmax(qm, -1), model.meter_logp(mprev, phi, phiprev, pr[:, t]))
        klp = klp + kl_von_mises(phi, qpk, ppm, ppk); klt = klt + kl_log_normal(qtm, qts, ptm, pts)
        zf.append(model.zfeat(m, phi, lt)); tempo_mu.append(qtm); mprev, phiprev, ltprev = m, phi, lt
    logits = torch.stack([model.decode(zf[t]) for t in range(T)], 1)
    return (klm, klp, klt), logits, torch.stack(tempo_mu, 1)


def load(cache_dir, n, seed):
    out = []
    for f in sorted(glob.glob(f"{cache_dir}/*.pt"))[:n] if seed is None else random.Random(seed).sample(sorted(glob.glob(f"{cache_dir}/*.pt")), n):
        d = torch.load(f, map_location="cpu"); hh = d["activations"].float()
        if hh.shape[0] < 400 or d["beat_targets"].sum() < 8: continue
        out.append((hh, d["beat_targets"].float(), d["downbeat_targets"].float()))
    return out


def make_batch(songs, frames, bs):
    hs, bs_, ds_, ph = [], [], [], []
    while len(hs) < bs:
        hh, b, db = random.choice(songs)
        if hh.shape[0] <= frames: continue
        s0 = random.randint(0, hh.shape[0] - frames)
        bb = b[s0:s0 + frames]; dd = db[s0:s0 + frames]
        if bb.sum() < 2: continue
        bf = np.where(bb.numpy() > 0.5)[0]
        hs.append(hh[s0:s0 + frames]); bs_.append(bb); ds_.append(dd)
        ph.append(torch.tensor(gt_barphase(bf, 4, frames), dtype=torch.float32))
    return (torch.stack(hs).to(DEV), torch.stack(bs_).to(DEV), torch.stack(ds_).to(DEV), torch.stack(ph).to(DEV))


@torch.no_grad()
def evaluate(model, val, frames=1600):
    model.eval(); bF, dF, tempo_err = [], [], []
    for hh, b, db in val:
        T = min(hh.shape[0], frames); bf = np.where(b.numpy()[:T] > 0.5)[0]
        if len(bf) < 4: continue
        gp = torch.tensor(gt_barphase(bf, 4, T), dtype=torch.float32, device=DEV).unsqueeze(0)
        _, logits, tmu = clamp_rollout(model, hh[:T].unsqueeze(0).to(DEV), gp, temp=0.3)
        prob = torch.sigmoid(logits)[0].cpu().numpy()
        ref = bf / FPS; dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        bF.append(fmeas(ref, peaks(prob[:, 0])))
        if len(dref) >= 2: dF.append(fmeas(dref, peaks(prob[:, 1], min_dist=0.30)))
        # tempo slaving: learned advance vs the GT ramp advance
        learned_adv = float(torch.exp(tmu).mean()); gt_adv = float(np.mean(np.diff(gp[0].cpu().numpy()) % TWO_PI))
        learned_bpm = 60 * FPS * 4 * learned_adv / TWO_PI; gt_bpm = 60 * FPS * 4 * gt_adv / TWO_PI
        tempo_err.append((learned_bpm, gt_bpm))
    model.train()
    lb = np.mean([x[0] for x in tempo_err]); gb = np.mean([x[1] for x in tempo_err])
    return float(np.nanmean(bF)), float(np.nanmean(dF)) if dF else float('nan'), lb, gb


def main():
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    print("loading cached Beat-This features (train 200 / val 40) ...", flush=True)
    train = load("cache/acts/bt_train_rich", 200, seed=1); val = load("cache/acts/bt_val_rich", 40, seed=2)
    print(f"train={len(train)} val={len(val)} | CLAMP phi=GT rotating ramp; STRICT ELBO (NO free-bits); tempo/meter/decoder free", flush=True)
    model = BPVAE(h_dim=512, hidden=64).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, 301):
        temp = 1.0 + (0.3 - 1.0) * min(step / 300, 1.0)
        h, b, db, gp = make_batch(train, 256, 16)
        (klm, klp, klt), logits, _ = clamp_rollout(model, h, gp, temp)
        pw = torch.tensor([8.0, 20.0], device=DEV)
        recon = F.binary_cross_entropy_with_logits(logits, torch.stack([b, db], -1), pos_weight=pw, reduction="none").sum((1, 2))
        loss = (recon + klm + klp + klt).mean()   # STRICT ELBO -- NO free-bits (faithful); lets KL_phase pull the tempo
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % 150 == 0 or step == 300:
            bF, dF, lb, gb = evaluate(model, val)
            print(f"step {step}: decoder(beat from clamped rotating phi)={bF:.3f}  downbeat={dF:.3f} | "
                  f"tempo learned~{lb:.0f}BPM vs GT~{gb:.0f}BPM", flush=True)
    print("\nVERDICT: if decoder beat/db are HIGH and learned tempo ~= GT tempo -> the rest FOLLOWS a rotating phi")
    print("  (free training gave: phi static, ~0 revolutions, tempo->0). So the ONLY failure is phi not rotating.")


if __name__ == "__main__":
    main()
