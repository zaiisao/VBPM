"""NEXT-FIX EXPERIMENTS — test the two diagnosed binding problems, factorially.

(A) LIKELIHOOD: widen the beat TARGET to +/-W frames (shift-tolerant proxy for Beat-This loss),
    attacking the majority-class decoder collapse on ~1.5%-positive single-frame Bernoulli.
(B) PRIOR: OU / mean-reverting tempo prior  log tau_t ~ N((1-theta)*log tau_{t-1} + theta*C, sigma),
    bounding Var (RW is theta=0) -> attacks the unbounded-tempo blowup AND its optimization hazard.

Primary metric = tf_post_dec (teacher-forced posterior DECODER beat-F) -- the training-health
number; free-run F is inflated by periodicity. Also reports free-run latent F and TF posterior
latent F. faithful/ is untouched (target widening + OU mean applied here in the rollout copy).

Flags: --widen W (frames, 0=off), --ou THETA (0=pure RW). Usage mirrors ablate.py.
"""
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, "/home/sogang/jaehoon/CHART")
import numpy as np, torch
import torch.nn.functional as F
from faithful.data import FPS, N_MELS, LogMel, build_train_loader, iter_val_songs
from faithful.model import BarPointerVAE
from faithful.distributions import (TWO_PI, gumbel_softmax, sample_von_mises,
                                    kl_categorical, kl_von_mises, kl_log_normal, log_i0)


# --- pointwise log-densities for FIVO importance weights ---
def vm_logp(phi, mu, kappa):      # von Mises log-density at phi
    return kappa * torch.cos(phi - mu) - math.log(TWO_PI) - log_i0(kappa)

def logn_logp(x, mu, sig):        # Gaussian log-density (x = log-tempo, Gaussian in log-space)
    return -0.5 * ((x - mu) / sig) ** 2 - torch.log(sig) - 0.5 * math.log(TWO_PI)

def cat_logp(soft, logits):       # E_soft[log Cat] (soft assignment approx for the relaxed meter)
    return (soft * F.log_softmax(logits, -1)).sum(-1)


def fivo(model, h, b_enc, b_tgt, temp, K):
    """FIVO / SMC-ELBO: K particles per sequence, resample every step by the per-step importance weight
    w = p(b_t|z_t) p(z_t|z_{t-1}) / q(z_t|.). Objective = sum_t [logsumexp_K(log w) - log K] (a tighter
    ELBO than one-step). Uses the LEARNED generative model (prior dynamics + decoder) and the encoder as
    proposal -> a real VAE; the filter does sequential inference that should identify tempo (pure latent).
    K=1 reduces exactly to the standard single-sample ELBO (known-answer test)."""
    B, T, _ = h.shape
    hK = h.repeat_interleave(K, 0); beK = b_enc.repeat_interleave(K, 0); btK = b_tgt.repeat_interleave(K, 0)
    N = B * K
    pc = model.encode_prior(hK); qc = model.encode_posterior(hK, beK)
    logZ = h.new_zeros(B)
    z0 = model.z0.unsqueeze(0).expand(N, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([qc[:, 0], z0], -1)))
    pm, ppm, ppk, ptm, pts = model.unpack(model.prior_init_head(pc.mean(1)))
    meter = gumbel_softmax(qm, temp); phi = sample_von_mises(qpm, qpk) % TWO_PI
    lt = qtm + qts * torch.randn_like(qtm)
    mp, pp, lp = meter, phi, lt
    for t in range(T):
        if t > 0:
            qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(
                torch.cat([qc[:, t], model.z_features(mp, pp, lp)], -1)))
            ppm = (pp + torch.exp(lp)) % TWO_PI
            ppk = F.softplus(model.prior_phase_kappa(pc[:, t]).squeeze(-1)) + 0.01
            ptm = lp; pts = F.softplus(model.prior_tempo_sigma(pc[:, t]).squeeze(-1)) + 1e-3
            meter = gumbel_softmax(qm, temp); phi = sample_von_mises(qpm, qpk) % TWO_PI
            lt = qtm + qts * torch.randn_like(qtm)
            logpm = model.meter_prior_logp(mp, phi, pp, pc[:, t])
        else:
            logpm = F.log_softmax(pm, -1)
        zf = model.z_features(meter, phi, lt)
        dec_logit = model.decode(zf, pc[:, t])
        logp_dec = -F.binary_cross_entropy_with_logits(dec_logit, btK[:, t], reduction="none")
        logp_z = vm_logp(phi, ppm, ppk) + logn_logp(lt, ptm, pts) + cat_logp(meter, logpm)
        logq_z = vm_logp(phi, qpm, qpk) + logn_logp(lt, qtm, qts) + cat_logp(meter, F.log_softmax(qm, -1))
        lw = (logp_dec + logp_z - logq_z).view(B, K)               # per-step importance weight
        logZ = logZ + torch.logsumexp(lw, 1) - math.log(K)
        if K > 1:                                                  # resample (stop-grad indices)
            w = F.softmax(lw, 1)
            idx = torch.multinomial(w, K, replacement=True) + (torch.arange(B, device=h.device) * K).unsqueeze(1)
            idx = idx.view(-1)
            phi, lt, meter = phi[idx], lt[idx], meter[idx]
        mp, pp, lp = meter, phi, lt
    return -logZ.mean()                                            # negative FIVO bound (minimize)
from faithful.evaluate import beats_from_barphase, beats_from_activation, f_measure
from faithful.elbo import free_run

OU_C = -3.3   # log bar-advance-rate center (~120 BPM, 4/4): 2*pi*0.5/fps ~ 0.037 rad/frame, log ~ -3.3


def widen_target(b, W):
    """max-pool the single-frame beat target to +/-W frames (shift tolerance)."""
    if W <= 0:
        return b
    return F.max_pool1d(b.unsqueeze(1), kernel_size=2 * W + 1, stride=1, padding=W).squeeze(1)


METERS_VEC = None   # set per-device in rollout: beats-per-bar for each meter class


def freerun_recon(model, h, b_tgt, temp, gtau=None, tierb_a=0.0):
    """DBN-style: roll the PRIOR forward on its own (no posterior re-anchoring), decode LATENT-ONLY,
    and reconstruct the beats. The latent must explain the beat SPACING with its own dynamics -> this is
    the term that actually forces tempo (teacher-forced recon re-anchors phase every step and hides it).
    Fully differentiable via reparam (von Mises implicit-reparam + Gaussian tempo)."""
    B, T, _ = h.shape
    pc = model.encode_prior(h)
    pm, ppm, ppk, ptm, pts = model.unpack(model.prior_init_head(pc.mean(1)))
    meter = gumbel_softmax(pm, temp); phi = sample_von_mises(ppm, ppk) % TWO_PI
    lt = ptm + pts * torch.randn_like(ptm)
    zf = [model.z_features(meter, phi, lt)]; mp, pp, lp = meter, phi, lt
    for t in range(1, T):
        ppm = (pp + torch.exp(lp)) % TWO_PI
        ppk = F.softplus(model.prior_phase_kappa(pc[:, t]).squeeze(-1)) + 0.01
        if gtau is not None:
            g = gtau(pc[:, t]).squeeze(-1); ptm = lp + tierb_a * (g - lp) if tierb_a > 0 else lp + g
        else:
            ptm = lp
        pts = F.softplus(model.prior_tempo_sigma(pc[:, t]).squeeze(-1)) + 1e-3
        meter = gumbel_softmax(model.meter_prior_logp(mp, phi, pp, pc[:, t]), temp)
        phi = sample_von_mises(ppm, ppk) % TWO_PI
        lt = ptm + pts * torch.randn_like(ptm)
        zf.append(model.z_features(meter, phi, lt)); mp, pp, lp = meter, phi, lt
    # LATENT-ONLY decode (force the latent to carry it; pc passed but ignored when model.latent_only)
    logits = torch.stack([model.decode(zf[t], pc[:, t]) for t in range(T)], 1)
    return F.binary_cross_entropy_with_logits(logits, b_tgt, reduction="none").sum(1).mean()


def rollout(model, h, b_enc, b_tgt, temp, ou, overshoot=1, os_fn=0.0, os_w=1.0, survival_w=0.0,
            gtau=None, tierb_a=0.0, ss_prob=0.0, h_dropout=0.0, distill_w=0.0):
    global METERS_VEC
    B, T, _ = h.shape
    pc = model.encode_prior(h); qc = model.encode_posterior(h, b_enc)
    kl_m = h.new_zeros(B); kl_p = h.new_zeros(B); kl_t = h.new_zeros(B)
    distill = h.new_zeros(B)   # STOP-GRAD distillation: g_tau(audio) -> posterior tempo (accurate)
    zf = []
    s_phi, s_lt = [], []                        # detached posterior samples (overshoot seeds)
    q_pm, q_pk, q_tm, q_ts = [], [], [], []     # posterior params (overshoot targets)
    lts, ms = [], []                            # grad-carrying log_tempo + meter_soft (survival)
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([qc[:, 0], z0], -1)))
    pm, ppm, ppk, ptm, pts = model.unpack(model.prior_init_head(pc.mean(1)))
    meter = gumbel_softmax(qm, temp); phi = sample_von_mises(qpm, qpk) % TWO_PI
    lt = qtm + qts * torch.randn_like(qtm)
    kl_m = kl_m + kl_categorical(torch.log_softmax(qm, -1), torch.log_softmax(pm, -1))
    kl_p = kl_p + kl_von_mises(qpm, qpk, ppm, ppk)
    kl_t = kl_t + kl_log_normal(qtm, qts, ptm, pts)
    zf.append(model.z_features(meter, phi, lt)); mp, pp, lp = meter, phi, lt
    s_phi.append(phi.detach()); s_lt.append(lt.detach()); lts.append(lt); ms.append(meter)
    q_pm.append(qpm); q_pk.append(qpk); q_tm.append(qtm); q_ts.append(qts)
    for t in range(1, T):
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(
            torch.cat([qc[:, t], model.z_features(mp, pp, lp)], -1)))
        ppm = (pp + torch.exp(lp)) % TWO_PI
        ppk = F.softplus(model.prior_phase_kappa(pc[:, t]).squeeze(-1)) + 0.01
        if gtau is not None:
            g = gtau(pc[:, t]).squeeze(-1)               # TIER B: audio-conditioned tempo-mean
            ptm = lp + tierb_a * (g - lp) if tierb_a > 0 else lp + g   # v2 anchor / v1 delta
            if distill_w > 0.0:                          # distill accurate posterior tempo into g (stop-grad on q)
                distill = distill + (g - qtm.detach()) ** 2
        else:
            ptm = lp + ou * (OU_C - lp)                  # <-- OU mean reversion (ou=0 -> RW)
        pts = F.softplus(model.prior_tempo_sigma(pc[:, t]).squeeze(-1)) + 1e-3
        meter = gumbel_softmax(qm, temp); phi = sample_von_mises(qpm, qpk) % TWO_PI
        lt = qtm + qts * torch.randn_like(qtm)
        logpi = model.meter_prior_logp(mp, phi, pp, pc[:, t])
        kl_m = kl_m + kl_categorical(torch.log_softmax(qm, -1), logpi)
        kl_p = kl_p + kl_von_mises(qpm, qpk, ppm, ppk)
        kl_t = kl_t + kl_log_normal(qtm, qts, ptm, pts)
        zf.append(model.z_features(meter, phi, lt))
        s_phi.append(phi.detach()); s_lt.append(lt.detach()); lts.append(lt); ms.append(meter)
        if ss_prob > 0.0:    # SCHEDULED SAMPLING: feed the prior's OWN prediction as next-step prev (mask teacher-forcing)
            use_prior = (torch.rand(B, device=h.device) < ss_prob)
            pri_phi = sample_von_mises(ppm, ppk) % TWO_PI
            pri_lt = ptm + pts * torch.randn_like(ptm)
            pri_meter = gumbel_softmax(logpi, temp)
            pp = torch.where(use_prior, pri_phi, phi)
            lp = torch.where(use_prior, pri_lt, lt)
            mp = torch.where(use_prior.unsqueeze(-1), pri_meter, meter)
        else:
            mp, pp, lp = meter, phi, lt
        q_pm.append(qpm); q_pk.append(qpk); q_tm.append(qtm); q_ts.append(qts)
    pc_dec = pc
    if h_dropout > 0.0 and not model.latent_only:   # h-DROPOUT (word-dropout, Bowman 2016): mask audio frames into decoder -> forces latent use
        pc_dec = pc * (torch.rand(B, T, 1, device=h.device) > h_dropout).float()
    logits = torch.stack([model.decode(zf[t], pc_dec[:, t]) for t in range(T)], 1)
    recon = F.binary_cross_entropy_with_logits(logits, b_tgt, reduction="none").sum(1)
    loss = (recon + kl_m + kl_p + kl_t).mean()
    info = {"recon": float(recon.mean()), "kl_m": float(kl_m.mean()),
            "kl_p": float(kl_p.mean()), "kl_t": float(kl_t.mean()), "os": 0.0, "surv": 0.0, "distill": 0.0}
    if distill_w > 0.0 and gtau is not None:
        loss = loss + distill_w * distill.mean(); info["distill"] = float(distill.mean())
    # 1.3 RENEWAL/SURVIVAL: inhomogeneous-Poisson beat-rate NLL ties tempo(x meter) to event RATE.
    # rate r_t = m_eff * exp(lt_t) / 2pi  (beats/frame); NLL = sum_t r_t - sum_{events} log r_t.
    # Events = ORIGINAL single-frame beats (b_enc), not the widened target. A per-frame decoder
    # cannot represent this global-rate constraint -> makes the tempo latent load-bearing.
    if survival_w > 0.0:
        if METERS_VEC is None or METERS_VEC.device != h.device:
            METERS_VEC = torch.tensor([2., 3., 4., 5.], device=h.device)[:ms[0].shape[-1]]
        LT = torch.stack(lts, 1); M = torch.stack(ms, 1)              # [B,T], [B,T,K]
        m_eff = (M * METERS_VEC).sum(-1)                              # [B,T] soft beats-per-bar
        r = (m_eff * torch.exp(LT) / TWO_PI).clamp(1e-4, 0.5)        # beats/frame
        surv = (r.sum(1) - (b_enc * torch.log(r)).sum(1))            # [B] Poisson process NLL
        loss = loss + survival_w * surv.mean()
        info["surv"] = float(surv.mean())
    # latent overshooting (PlaNet): KL(stop_grad q(z_t) || p^{(d)}(z_t)) for d=2..D, OU-consistent prior mean.
    if overshoot >= 2:
        Sphi = torch.stack(s_phi, 1); Slt = torch.stack(s_lt, 1)
        Qpm = torch.stack(q_pm, 1).detach(); Qpk = torch.stack(q_pk, 1).detach()
        Qtm = torch.stack(q_tm, 1).detach(); Qts = torch.stack(q_ts, 1).detach()
        os_sum = h.new_zeros(())
        for d in range(2, min(overshoot, T - 1) + 1):
            ts = torch.arange(d, T, device=h.device); src = ts - d
            lt_k = Slt[:, src]; phi_adv = torch.zeros_like(lt_k)   # OU-consistent d-step mean rollout
            for _ in range(d):
                phi_adv = phi_adv + torch.exp(lt_k)
                lt_k = lt_k + ou * (OU_C - lt_k)
            ppm_d = (Sphi[:, src] + phi_adv) % TWO_PI
            ptm_d = lt_k                                            # OU mean after d steps
            ppk_d = F.softplus(model.prior_phase_kappa(pc[:, ts]).squeeze(-1)) + 0.01
            pts_d = F.softplus(model.prior_tempo_sigma(pc[:, ts]).squeeze(-1)) + 1e-3
            klp = kl_von_mises(Qpm[:, ts], Qpk[:, ts], ppm_d, ppk_d).mean(0).clamp(min=os_fn)
            klt = kl_log_normal(Qtm[:, ts], Qts[:, ts], ptm_d, pts_d).mean(0).clamp(min=os_fn)
            os_sum = os_sum + (klp + klt).sum()
        os_term = os_sum / (overshoot - 1)
        loss = loss + os_w * os_term
        info["os"] = float(os_term)
    return loss, info


@torch.no_grad()
def tf_posterior(model, h, b):
    B, T, _ = h.shape
    pc = model.encode_prior(h); qc = model.encode_posterior(h, b)
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([qc[:, 0], z0], -1)))
    phi = qpm % TWO_PI; lt = qtm; meter = F.softmax(qm, -1)
    traj = [phi]; lttraj = [lt]; zf = [model.z_features(meter, phi, lt)]
    for t in range(1, T):
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(
            torch.cat([qc[:, t], model.z_features(meter, phi, lt)], -1)))
        phi = qpm % TWO_PI; lt = qtm; meter = F.softmax(qm, -1)
        traj.append(phi); lttraj.append(lt); zf.append(model.z_features(meter, phi, lt))
    dec = torch.sigmoid(torch.stack([model.decode(zf[t], pc[:, t]) for t in range(T)], 1))
    return (torch.stack(traj, 1)[0].cpu().numpy(), torch.stack(lttraj, 1)[0].cpu().numpy(),
            dec[0].cpu().numpy())


@torch.no_grad()
def free_run_tierb(model, gtau, h, T, tierb_a=0.0):
    """Tier-B deployment: free-run MEAN chain with audio-conditioned tempo correction (closed loop).
    v2 (anchor, tierb_a>0): lt_mu += a*(g_target(pc) - lt_mu)  -> RESTORING force toward audio tempo.
    v1 (delta,  tierb_a=0): lt_mu += gtau(pc)                  -> audio-driven drift (accumulates)."""
    pc = model.encode_prior(h)
    p_m, p_phi_mu, p_phi_k, p_tau_mu, p_tau_s = model.unpack(model.prior_init_head(pc.mean(1)))
    phi_mu = float(p_phi_mu[0] % TWO_PI); lt_mu = float(p_tau_mu[0])
    out = [phi_mu]
    for t in range(1, T):
        g = float(gtau(pc[:, t]).squeeze())
        lt_mu = lt_mu + (tierb_a * (g - lt_mu) if tierb_a > 0 else g)
        phi_mu = (phi_mu + math.exp(lt_mu)) % TWO_PI
        out.append(phi_mu)
    return np.array(out)


@torch.no_grad()
def probe(model, dev, root, keys, gtau=None, tierb_a=0.0):
    logmel = LogMel().to(dev)
    tfd, tfl, frl, tacc, pt_pred, pt_true = [], [], [], [], [], []
    for key, audio, beats, downs, meta in iter_val_songs(root, keys, max_per_dataset=4):
        T = min(len(beats), 1200)
        ref = np.where(beats.numpy()[:T] > 0.5)[0] / FPS
        df = np.where(downs.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) < 8:
            continue
        m = 4
        if len(df) >= 2:
            bpb = np.median([np.sum((ref >= df[i]) & (ref < df[i+1])) for i in range(len(df)-1)])
            m = max(2, min(int(round(bpb)) if bpb > 0 else 4, 4))
        h = logmel(audio.to(dev).unsqueeze(0))[:, :T]
        b = beats[:T].to(dev).unsqueeze(0).float()
        phi, ltp, dec = tf_posterior(model, h, b)
        tfd.append(f_measure(ref, beats_from_activation(dec, FPS)))
        tfl.append(f_measure(ref, beats_from_barphase(phi, m, FPS)))
        # deployment free-run: Tier-B closed loop if gtau, else faithful open-loop
        if gtau is not None:
            frp = free_run_tierb(model, gtau, h, T, tierb_a=tierb_a)
        else:
            frp = free_run(model, h, temperature=0.3)["phase_mu"][0, :T].cpu().numpy()
        frl.append(f_measure(ref, beats_from_barphase(frp, m, FPS)))
        gt = 60.0 / np.median(np.diff(ref))
        # POSTERIOR tempo (teacher-forced): beat-BPM = m*exp(median posterior log_tempo)
        pbpm = 60.0 * FPS * m * math.exp(float(np.median(ltp))) / TWO_PI
        tacc.append(float(abs(pbpm - gt) / gt < 0.04))
        pt_pred.append(math.log(max(pbpm, 1e-3))); pt_true.append(math.log(gt))   # slope/corr control
    corr = float(np.corrcoef(pt_pred, pt_true)[0, 1]) if len(pt_pred) > 2 else float("nan")
    return {"tf_post_dec": float(np.nanmean(tfd)), "tf_post_lat": float(np.nanmean(tfl)),
            "fr_lat": float(np.nanmean(frl)), "tf_tempo_Acc1": float(np.nanmean(tacc)) if tacc else 0.0,
            "tf_tempo_corr": corr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--data_root", default="/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data")
    ap.add_argument("--datasets", default="ballroom,beatles,hains,rwc_popular")
    ap.add_argument("--frames", type=int, default=128); ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--widen", type=int, default=0); ap.add_argument("--ou", type=float, default=0.0)
    ap.add_argument("--latent_only", action="store_true")
    ap.add_argument("--overshoot", type=int, default=1); ap.add_argument("--os_free_nats", type=float, default=0.0)
    ap.add_argument("--os_weight", type=float, default=1.0)
    ap.add_argument("--survival_weight", type=float, default=0.0, help="1.3 renewal-IOI Poisson NLL weight (ramped over first half)")
    ap.add_argument("--tierb", action="store_true", help="Tier B: audio-conditioned prior tempo-mean correction head g_tau(pc)")
    ap.add_argument("--tierb_anchor", type=float, default=0.0, help="Tier B v2 anchor strength a (>0: lp+a*(g-lp) restoring; 0: lp+g delta)")
    ap.add_argument("--freerun_weight", type=float, default=0.0, help="DBN-style free-run reconstruction weight (latent must explain beats via its own prior rollout); ramped over first half")
    ap.add_argument("--ss_prob", type=float, default=0.0, help="scheduled sampling: max prob of feeding the prior's own prediction as next-step prev (ramped)")
    ap.add_argument("--h_dropout", type=float, default=0.0, help="decoder audio-frame dropout (word-dropout) to force latent use")
    ap.add_argument("--init_from", default="", help="warm-start: load model (+gtau) state from this checkpoint before training")
    ap.add_argument("--distill_weight", type=float, default=0.0, help="stop-grad distillation: train g_tau to reproduce the (accurate) posterior tempo")
    ap.add_argument("--fivo", action="store_true", help="train with the FIVO/SMC-ELBO objective (particle filter; K=1 == standard ELBO)")
    ap.add_argument("--n_particles", type=int, default=8, help="FIVO particle count")
    ap.add_argument("--selftest", action="store_true", help="run K=1 FIVO == ELBO known-answer test and exit")
    args = ap.parse_args()
    torch.manual_seed(42); dev = "cuda"
    keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[{args.cell}] widen={args.widen} ou={args.ou} steps={args.steps}", flush=True)
    logmel = LogMel().to(dev)
    model = BarPointerVAE(h_dim=N_MELS, hidden=64, num_meters=4, latent_only=args.latent_only).to(dev)
    import torch.nn as nn
    gtau = None
    params = list(model.parameters())
    if args.tierb:                          # Tier B tempo head on the prior context (hidden=64)
        gtau = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1)).to(dev)
        nn.init.zeros_(gtau[-1].weight)
        # v2 anchor: head outputs an ABSOLUTE log-tempo target, init at OU_C (~120bpm bar rate).
        # v1 delta: head outputs a correction, init 0 (= RW).
        nn.init.constant_(gtau[-1].bias, OU_C if args.tierb_anchor > 0 else 0.0)
        params = params + list(gtau.parameters())
    if args.init_from:                      # WARM-START from a healthy checkpoint
        ck = torch.load(args.init_from, map_location=dev)
        model.load_state_dict(ck["model"])
        if gtau is not None and "gtau" in ck:
            gtau.load_state_dict(ck["gtau"])
        print(f"[{args.cell}] warm-started from {args.init_from}", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr)
    loader = build_train_loader(args.data_root, keys, args.frames, args.batch_size,
                                examples_per_epoch=1000, num_workers=4, seed=42)
    di = iter(loader)
    if args.selftest:   # KNOWN-ANSWER: K=1 FIVO is a valid single-sample ELBO; K>1 is a TIGHTER bound (lower loss)
        audio, beats, _ = next(di)
        h = logmel(audio.to(dev))[:, :args.frames]; bt = beats[:, :args.frames].to(dev)
        Tm = min(h.shape[1], bt.shape[1]); h, bt = h[:, :Tm], bt[:, :Tm]; b_tgt = widen_target(bt, args.widen)
        torch.manual_seed(1); l_cf, _ = rollout(model, h, bt, b_tgt, 1.0, 0.0)        # closed-form-KL ELBO
        torch.manual_seed(1); f1v = float(fivo(model, h, bt, b_tgt, 1.0, 1))           # FIVO K=1
        torch.manual_seed(1); f8v = float(fivo(model, h, bt, b_tgt, 1.0, 8))           # FIVO K=8 (tighter)
        print(f"[selftest] closed-form ELBO={float(l_cf):.1f}  FIVO(K=1)={f1v:.1f}  FIVO(K=8)={f8v:.1f}")
        print(f"[selftest] K=1~ELBO ballpark: {abs(f1v-float(l_cf))/abs(float(l_cf))<0.3} | K=8 tighter (<K=1): {f8v<f1v}")
        return
    step = 0; t0 = time.time()
    while step < args.steps:
        try:
            audio, beats, _ = next(di)
        except StopIteration:
            di = iter(loader); audio, beats, _ = next(di)
        step += 1
        temp = 1.0 + (0.3 - 1.0) * min(step / args.steps, 1.0)
        h = logmel(audio.to(dev))[:, :args.frames]; bt = beats[:, :args.frames].to(dev)
        Tm = min(h.shape[1], bt.shape[1]); h, bt = h[:, :Tm], bt[:, :Tm]
        b_tgt = widen_target(bt, args.widen)
        sw = args.survival_weight * min(1.0, step / (0.5 * args.steps))   # ramp over first half
        opt.zero_grad()
        ssp = args.ss_prob * min(1.0, step / (0.5 * args.steps))   # ramp scheduled-sampling prob
        fr = 0.0
        if args.fivo:                          # FIVO/SMC-ELBO objective (real VAE, filtering inference)
            loss = fivo(model, h, bt, b_tgt, temp, args.n_particles)
            info = {"recon": float(loss), "kl_m": 0.0, "kl_p": 0.0, "kl_t": 0.0, "os": 0.0, "surv": 0.0, "distill": 0.0}
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
            if step % 50 == 0 or step == 1:
                print(f"[{args.cell}] s{step} FIVO={info['recon']:.1f} K={args.n_particles} {step/(time.time()-t0):.2f}it/s", flush=True)
            continue
        loss, info = rollout(model, h, bt, b_tgt, temp, args.ou,
                             overshoot=args.overshoot, os_fn=args.os_free_nats, os_w=args.os_weight,
                             survival_w=sw, gtau=gtau, tierb_a=args.tierb_anchor,
                             ss_prob=ssp, h_dropout=args.h_dropout, distill_w=args.distill_weight)
        if args.freerun_weight > 0:
            frw = args.freerun_weight * min(1.0, step / (0.5 * args.steps))   # ramp over first half
            fr_loss = freerun_recon(model, h, b_tgt, temp, gtau=gtau, tierb_a=args.tierb_anchor)
            loss = loss + frw * fr_loss; fr = float(fr_loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        if step % 50 == 0 or step == 1:
            print(f"[{args.cell}] s{step} recon={info['recon']:.1f} "
                  f"kl(m/p/t)={info['kl_m']:.3f}/{info['kl_p']:.3f}/{info['kl_t']:.3f} "
                  f"os={info['os']:.2f} surv={info['surv']:.2f} frrec={fr:.1f} distill={info['distill']:.3f} {step/(time.time()-t0):.2f}it/s", flush=True)
    save = {"model": model.state_dict(), "cell": args.cell, "args": vars(args)}
    if gtau is not None: save["gtau"] = gtau.state_dict()
    torch.save(save, out / "final.pt")
    model.eval()
    if gtau is not None: gtau.eval()
    res = {"cell": args.cell, "widen": args.widen, "ou": args.ou, "latent_only": args.latent_only,
           "overshoot": args.overshoot, "survival_weight": args.survival_weight, "tierb": args.tierb,
           "tierb_anchor": args.tierb_anchor, "freerun_weight": args.freerun_weight,
           "ss_prob": args.ss_prob, "h_dropout": args.h_dropout, "distill_weight": args.distill_weight,
           "final_train": info, **probe(model, dev, args.data_root, keys, gtau=gtau, tierb_a=args.tierb_anchor)}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(f"[{args.cell}] DONE tf_post_dec={res['tf_post_dec']:.3f} tf_post_lat={res['tf_post_lat']:.3f} "
          f"fr_lat={res['fr_lat']:.3f} tf_tempo_Acc1={res['tf_tempo_Acc1']:.3f} "
          f"tf_tempo_corr={res['tf_tempo_corr']:.3f} -> {out/'result.json'}", flush=True)


if __name__ == "__main__":
    main()
