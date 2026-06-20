"""Fast-proxy harness: train CHART on cached activations, read the verdict in minutes.

The frozen frontend is already cached to disk (tests/cache_activations.py), so this
trains ONLY the SVT on tiny tensors — seconds/epoch instead of minutes. Every K steps
it runs the DEPLOYED path (sample_from_prior, free-running) on held-out songs and scores
free-run F / CMLt (beats + downbeats) with mir_eval, alongside per-latent KL. Because we
empirically established the gap barely moves ep4->ep14, the trajectory in the first few
hundred steps is a faithful verdict for a given idea.

It mirrors training/train.py's exact forward + ELBO assembly so the baseline reproduces
the real runs; idea toggles (VRNN prior, Z-Forcing, delta-VAE, aggressive encoder, meter
fix) are wired as flags so each can be A/B'd against the same baseline in one table.

Run (baseline = dir1 replica):
    python tests/fast_proxy.py \
        --train_cache cache/acts/wb_train --heldout_cache cache/acts/wb_val \
        --steps 600 --eval_every 150 --tag baseline
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import torch
from torch import optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.loss import compute_elbo_loss
from models.svt_core import SVTModel, TWO_PI
from models.distributions import von_mises_sample, _A, _log_i0
from evaluation.phase_converter import (
    extract_beat_timestamps,
    extract_beats_from_phase_trajectory,
)
from evaluation.score import (
    evaluate_beats,
    evaluate_downbeats,
    frames_to_beat_times,
)

_BEAT_KEYS = ("F-measure", "CMLt", "AMLt")
_DB_KEYS = ("db_F-measure", "db_CMLt", "db_AMLt")


# ----------------------------- cache loading ------------------------------ #

def _load_cache(cache_dir: str, device: torch.device) -> list[dict]:
    recs = []
    files = glob.glob(str(Path(cache_dir) / "*.pt"))
    if not files:  # support per-dataset subdirs (bt_train/<dataset>/*.pt)
        files = glob.glob(str(Path(cache_dir) / "**" / "*.pt"), recursive=True)
    for f in sorted(files):
        r = torch.load(f, map_location="cpu")
        rec = {
            "activations": r["activations"].to(device),       # [T, 2]
            "beat_targets": r["beat_targets"].to(device),     # [T]
            "downbeat_targets": r["downbeat_targets"].to(device),
            "fps": float(r["fps"]),
        }
        for k in ("phase_prev", "log_tempo_prev", "meter_onehot_prev"):
            if k in r:
                rec[k] = r[k].to(device)
        recs.append(rec)
    return recs


def _make_train_batches(recs: list[dict], frames: int, batch_size: int):
    """Center-crop each cached song to `frames` and stack into fixed batches."""
    def crop(x, T):
        s = max(0, (x.shape[0] - T) // 2)
        return x[s:s + T]

    cropped = []
    for r in recs:
        T = min(r["activations"].shape[0], frames)
        item = {
            "activations": crop(r["activations"], T),
            "beat_targets": crop(r["beat_targets"], T),
            "downbeat_targets": crop(r["downbeat_targets"], T),
        }
        for k in ("phase_prev", "log_tempo_prev", "meter_onehot_prev"):
            if k in r:
                item[k] = crop(r[k], T)
        # Only keep full-length crops so they stack cleanly.
        if item["activations"].shape[0] == frames:
            cropped.append(item)

    batches = []
    for i in range(0, len(cropped), batch_size):
        chunk = cropped[i:i + batch_size]
        if not chunk:
            continue
        batch = {
            "activations": torch.stack([c["activations"] for c in chunk]),
            "beat_targets": torch.stack([c["beat_targets"] for c in chunk]),
            "downbeat_targets": torch.stack([c["downbeat_targets"] for c in chunk]),
        }
        for k in ("phase_prev", "log_tempo_prev", "meter_onehot_prev"):
            if all(k in c for c in chunk):
                batch[k] = torch.stack([c[k] for c in chunk])
        batches.append(batch)
    return batches


# ----------------------------- schedules ---------------------------------- #

def _temp_at(step: int, steps: int, t0: float, t1: float) -> float:
    frac = step / max(steps, 1)
    return t0 + (t1 - t0) * frac


def _beta_at(step: int, steps: int, anneal_frac: float) -> float:
    if anneal_frac <= 0:
        return 1.0
    n = anneal_frac * steps
    return min(1.0, step / max(n, 1))


def _ss_at(step: int, steps: int, anneal_frac: float, max_eps: float) -> float:
    if max_eps <= 0:
        return 0.0
    start = anneal_frac * steps
    if step < start:
        return 0.0
    frac = (step - start) / max(steps - start, 1)
    return min(max_eps, max_eps * frac)


# ----------------------------- He-2019 aggressive inference --------------- #

@torch.no_grad()
def _calc_mi_continuous(model, heldout, max_frames, n_cap=512):
    """Aggregate-posterior mutual information I(x;z) for CHART's CONTINUOUS latents
    (von Mises phase + log-tempo Gaussian), used as He-2019's "STOP BURNING"
    criterion. Transcribed from the vendored encoder.calc_mi (the log-sum-exp
    aggregate-posterior estimator), with the Gaussian phase factor replaced by its
    von Mises analog. MI = E_x E_q[log q(z|x)] - E_x E_q[log q(z)], summed over the
    two continuous factors; (clip, timestep) posteriors are pooled and subsampled.
    """
    was_training = model.training
    model.eval()
    mu_ph, kap_ph, mu_t, lv_t = [], [], [], []
    for r in heldout:
        act = r["activations"][:max_frames].unsqueeze(0)
        post = model(act)["posterior"]
        mu_ph.append(post["phase_mu"].reshape(-1))
        kap_ph.append(post["phase_log_kappa"].exp().reshape(-1))
        mu_t.append(post["tempo_mu"].reshape(-1))
        lv_t.append((2.0 * post["tempo_log_sigma"]).reshape(-1))   # logvar = 2 log sigma
    mu_ph = torch.cat(mu_ph); kap_ph = torch.cat(kap_ph)
    mu_t = torch.cat(mu_t); lv_t = torch.cat(lv_t)
    N = mu_ph.shape[0]
    if N > n_cap:
        idx = torch.randperm(N, device=mu_ph.device)[:n_cap]
        mu_ph, kap_ph, mu_t, lv_t = mu_ph[idx], kap_ph[idx], mu_t[idx], lv_t[idx]
        N = n_cap
    log2pi = math.log(2 * math.pi)

    # --- phase: von Mises analog of encoder.calc_mi ---
    # E_q[log q(phi|x)] = kappa*A(kappa) - log(2pi) - log I0(kappa)   (= -H[vM])
    neg_ent_ph = (_A(kap_ph) * kap_ph - log2pi - _log_i0(kap_ph)).mean()
    phi = von_mises_sample(mu_ph, kap_ph)                              # [N]
    # log q(phi_i | x_j) = kappa_j cos(phi_i - mu_j) - log(2pi) - log I0(kappa_j)
    d_ph = (kap_ph[None, :] * torch.cos(phi[:, None] - mu_ph[None, :])
            - log2pi - _log_i0(kap_ph)[None, :])                       # [N, N]
    log_qz_ph = torch.logsumexp(d_ph, dim=1) - math.log(N)
    mi_ph = neg_ent_ph - log_qz_ph.mean()

    # --- log-tempo: He-2019's exact diagonal-Gaussian formula (nz=1) ---
    var_t = lv_t.exp()
    neg_ent_t = (-0.5 * log2pi - 0.5 * (1 + lv_t)).mean()
    z_t = mu_t + torch.randn_like(mu_t) * var_t.sqrt()                 # [N]
    dev = z_t[:, None] - mu_t[None, :]                                 # [N, N]
    d_t = -0.5 * (dev ** 2) / var_t[None, :] - 0.5 * (log2pi + lv_t[None, :])
    log_qz_t = torch.logsumexp(d_t, dim=1) - math.log(N)
    mi_t = neg_ent_t - log_qz_t.mean()

    if was_training:
        model.train()
    return float(mi_ph + mi_t)


# ----------------------------- evaluation --------------------------------- #

@torch.no_grad()
def _heldout_freerun(model, heldout, fps, max_frames, temperature=0.1, diag=False):
    """Run the deployed free-running prior on every held-out song; average mir_eval."""
    model.eval()
    sums, counts = {}, {}

    def acc(prefix, scores, keys):
        for k in keys:
            key = prefix + k
            sums[key] = sums.get(key, 0.0) + scores[k]
            counts[key] = counts.get(key, 0) + 1

    for r in heldout:
        act = r["activations"][:max_frames].unsqueeze(0)
        bt = r["beat_targets"][:max_frames]
        db = r["downbeat_targets"][:max_frames]
        ref = frames_to_beat_times(bt.cpu().numpy(), fps)
        if len(ref) < 2:
            continue
        ref_db = frames_to_beat_times(db.cpu().numpy(), fps)

        out = model.sample_from_prior(act, temperature=temperature)
        phase = out.get("phase_mu", out["phase"])[0].cpu().numpy()
        bprobs = torch.sigmoid(out["beat_logits"][0, :, 0]).cpu().numpy()
        dbprobs = torch.sigmoid(out["beat_logits"][0, :, 1]).cpu().numpy()

        acc("phase_", evaluate_beats(ref, extract_beats_from_phase_trajectory(phase, fps=fps)), _BEAT_KEYS)
        acc("dec_", evaluate_beats(ref, extract_beat_timestamps(bprobs, fps=fps)), _BEAT_KEYS)
        if len(ref_db) >= 2:
            acc("dec_", evaluate_downbeats(ref_db, extract_beat_timestamps(dbprobs, fps=fps)), _DB_KEYS)
            # Bar-phase read-out (Mode-4): downbeats from φ^bar wraps, mirroring how
            # beats are read off φ wraps.
            if out.get("bar_phase_mu") is not None:
                barphase = out["bar_phase_mu"][0].cpu().numpy()
                acc("barwrap_", evaluate_downbeats(
                    ref_db, extract_beats_from_phase_trajectory(barphase, fps=fps)), _DB_KEYS)
            if diag:
                # Meter-ablation: re-decode the SAME free-run trajectory with meter_soft
                # zeroed. If dec downbeats survive -> they don't come from the meter latent
                # (must be phase/tempo structure); if they vanish -> the floor-KL meter is
                # secretly carrying bar position. h is unused on a latent-only decoder.
                h_dummy = torch.zeros(act.shape[0], out["phase"].shape[1], model.hidden_dim, device=act.device)
                db_logits_nm = model._decode(
                    out["phase"], out["log_tempo"], torch.zeros_like(out["meter_soft"]),
                    h_dummy, out.get("bar_phase"))
                dbprobs_nm = torch.sigmoid(db_logits_nm[0, :, 1]).cpu().numpy()
                acc("decNoMeter_", evaluate_downbeats(
                    ref_db, extract_beat_timestamps(dbprobs_nm, fps=fps)), _DB_KEYS)

    model.train()
    return {k: sums[k] / max(counts[k], 1) for k in sums}


# ----------------------------- main --------------------------------------- #

def _parse_ideas(s: str) -> set:
    """'bar_phase+z_forcing' or 'bar_phase,z_forcing' -> {'bar_phase','z_forcing'}."""
    return {x for x in s.replace("+", ",").split(",") if x and x != "none"}


def build_model(cli, device) -> SVTModel:
    # dir1-replica defaults; --baseline core/faithful strip crutches. Ideas compose:
    # multiple features (bar_phase, z_forcing, meter_ste, aggressive_encoder) can be on.
    ideas = _parse_ideas(cli.idea)
    m = SVTModel(
        hidden_dim=128, nhead=4, num_layers=2, num_meter_classes=cli.num_meter_classes,
        phase_corr_scale=cli.phase_corr_scale,
        tempo_corr_scale=cli.tempo_corr_scale,
        decoder_use_h_prior=not cli.decoder_latent_only,
        tempo_anchor_mode=cli.tempo_anchor_mode,
        tempo_reversion_alpha=cli.tempo_reversion_alpha,
        audio_emission=cli.audio_emission,
        bar_phase=("bar_phase" in ideas),
        meter_ste=("meter_ste" in ideas),
        delta_vae=("delta_vae" in ideas),
        delta_vae_rate=cli.free_bits_tempo,   # reuse the tempo rate target as delta
    ).to(device)
    return m


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train_cache", required=True)
    p.add_argument("--heldout_cache", required=True)
    p.add_argument("--tag", default="run")
    p.add_argument("--device", default="cuda")
    # proxy budget
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--eval_every", type=int, default=150)
    p.add_argument("--frames", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_frames", type=int, default=2048)
    p.add_argument("--max_train", type=int, default=0, help="cap #train songs (0=all)")
    p.add_argument("--max_heldout", type=int, default=0, help="cap #held-out songs (0=all)")
    p.add_argument("--seed", type=int, default=0)
    # schedules
    p.add_argument("--kl_anneal_frac", type=float, default=0.15)
    p.add_argument("--scheduled_sampling_max", type=float, default=0.5)
    p.add_argument("--gumbel_temp_start", type=float, default=1.0)
    p.add_argument("--gumbel_temp_end", type=float, default=0.1)
    # model / loss config (dir1 replica defaults)
    p.add_argument("--baseline", choices=["dir1", "core", "faithful"], default="dir1")
    p.add_argument("--num_meter_classes", type=int, default=8)
    p.add_argument("--phase_corr_scale", type=float, default=0.1)
    p.add_argument("--tempo_corr_scale", type=float, default=0.15)
    p.add_argument("--decoder_latent_only", action="store_true", default=True)
    p.add_argument("--tempo_anchor_mode", default="latent")
    p.add_argument("--tempo_reversion_alpha", type=float, default=0.4)
    p.add_argument("--audio_emission", action="store_true", default=True)
    p.add_argument("--audio_recon_weight", type=float, default=1.0)
    p.add_argument("--bce_pos_weight", type=float, default=5.0)
    p.add_argument("--bce_pos_weight_db", type=float, default=15.0)
    p.add_argument("--free_bits_phase", type=float, default=0.2)
    p.add_argument("--free_bits_tempo", type=float, default=0.1)
    p.add_argument("--free_bits_meter", type=float, default=0.2)
    p.add_argument("--taubar_sup_weight", type=float, default=1.0)
    p.add_argument("--meter_sup_weight", type=float, default=1.0)
    p.add_argument("--phase_sup_weight", type=float, default=1.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--barphase_sup_weight", type=float, default=1.0)
    p.add_argument("--free_bits_barphase", type=float, default=0.2)
    # --- idea toggles (implemented incrementally in svt_core/loss) ---
    p.add_argument("--idea", default="none",
                   help="one of: none|vrnn_prior|z_forcing|delta_vae|aggressive_encoder|meter_fix")
    p.add_argument("--diag", action="store_true",
                   help="downbeat mechanism diagnostics: meter-ablation + db CMLt/AMLt")
    p.add_argument("--save_ckpt", default="",
                   help="if set, save {svt_model, args} to this path (loadable by pf_eval_smc)")
    cli = p.parse_args()

    torch.manual_seed(cli.seed)
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")

    if cli.baseline == "core":
        # Strip the unfaithful direct-latent-supervision crutches.
        cli.taubar_sup_weight = cli.meter_sup_weight = cli.phase_sup_weight = 0.0

    if cli.baseline == "faithful":
        # ou2-orthodox (FAITHFULNESS.md): latent-only decoder + free-bits + OU(ema) anchor
        # + small audio-corr NUDGE (0.1/0.15). Strip everything NOT in the faithful spec:
        # no direct latent supervision, no audio_emission (Dir-1 PF head), no scheduled
        # sampling, no supervised taubar latent.
        cli.taubar_sup_weight = cli.meter_sup_weight = cli.phase_sup_weight = 0.0
        cli.audio_emission = False
        cli.audio_recon_weight = 0.0
        cli.scheduled_sampling_max = 0.0
        cli.tempo_anchor_mode = "ema"          # OU regularizer w/o a supervised tau_bar head
        cli.phase_corr_scale = 0.1             # the disclosed-but-accepted nudge
        cli.tempo_corr_scale = 0.15
        cli.decoder_latent_only = True

    # ---- idea-specific config (features COMPOSE: any subset can be on) ----
    ideas = _parse_ideas(cli.idea)
    aggr_steps = 0       # He-2019 aggressive inference-net updates
    zf_weight = 0.0      # Z-Forcing auxiliary future-prediction cost
    if "vrnn_prior" in ideas:
        cli.phase_corr_scale = 0.5
        cli.tempo_corr_scale = 0.5
    if "z_forcing" in ideas:
        # Auxiliary cost forcing z_t to PREDICT activations Δ frames ahead (cheap stand-in
        # for Z-Forcing). The emission head IS the mechanism, so enable it even on faithful.
        zf_weight = 1.0
        cli.audio_emission = True
    if "aggressive_encoder" in ideas:
        aggr_steps = 3
    # delta_vae: handled in build_model (delta_vae_rate = cli.free_bits_tempo). The
    # free-bits tempo floor is left ON but becomes INERT -- delta-VAE structurally makes
    # KL_tempo >= the same delta, so max(kl, free_bits) never clamps. No double-counting.
    # bar_phase and meter_ste are wired in build_model (they change the architecture).

    train_recs = _load_cache(cli.train_cache, device)
    heldout = _load_cache(cli.heldout_cache, device)
    if cli.max_train > 0:
        train_recs = train_recs[:cli.max_train]
    if cli.max_heldout > 0:
        heldout = heldout[:cli.max_heldout]
    fps = train_recs[0]["fps"] if train_recs else 86.1328125
    batches = _make_train_batches(train_recs, cli.frames, cli.batch_size)
    print(f"[proxy:{cli.tag}] train={len(train_recs)} songs -> {len(batches)} batches | "
          f"heldout={len(heldout)} songs | baseline={cli.baseline} idea={cli.idea} | "
          f"steps={cli.steps} frames={cli.frames} bs={cli.batch_size}")

    model = build_model(cli, device)
    # He-2019 (third_party/vae-lagging-encoder): SEPARATE inference-net (encoder) and
    # generative (prior+decoder) optimizers. In aggressive mode we burn the encoder to
    # convergence, then take a decoder-only step; in vanilla mode we step both each
    # iteration -- numerically identical to one AdamW over all params (Adam updates are
    # per-parameter over disjoint sets).
    _POST_KEYS = ("post_encoder", "post_proj", "post_pos_enc", "trans_post_head",
                  "init_post_head", "tempo_bar_post_head", "bar_post_head")
    enc_params, dec_params = [], []
    for _n, _p in model.named_parameters():
        (enc_params if any(k in _n for k in _POST_KEYS) else dec_params).append(_p)
    enc_opt = optim.AdamW(enc_params, lr=cli.lr)
    dec_opt = optim.AdamW(dec_params, lr=cli.lr)
    aggressive_flag = aggr_steps > 0   # the aggressive_encoder idea is the on-switch
    pre_mi = float("-inf")

    def forward_loss(act, bt, db, meter_tgt, phase_tgt, temp, beta):
        barphase_tgt = (model._beat_targets_to_distance(db) * TWO_PI) if model.bar_phase else None
        bp_w = cli.barphase_sup_weight if model.bar_phase else 0.0
        out = model(act, temperature=temp, beat_targets=bt, downbeat_targets=db)
        total, comps = compute_elbo_loss(
            beat_logits=out["beat_logits"], beat_targets=bt,
            posterior=out["posterior"], prior=out["prior"], beta=beta,
            pos_weight=cli.bce_pos_weight, pos_weight_db=cli.bce_pos_weight_db,
            free_bits_phase=cli.free_bits_phase, free_bits_tempo=cli.free_bits_tempo,
            free_bits_meter=cli.free_bits_meter, free_bits_barphase=cli.free_bits_barphase,
            barphase_targets=barphase_tgt, barphase_sup_weight=bp_w,
            tempo_bar=out.get("tempo_bar"), taubar_sup_weight=cli.taubar_sup_weight,
            meter_targets=meter_tgt, meter_sup_weight=cli.meter_sup_weight if meter_tgt is not None else 0.0,
            phase_targets=phase_tgt, phase_sup_weight=cli.phase_sup_weight if phase_tgt is not None else 0.0,
            downbeat_targets=db,
        )
        if cli.audio_recon_weight > 0 and out.get("audio_recon") is not None:
            ar = ((out["audio_recon"] - act) ** 2).mean()
            total = total + cli.audio_recon_weight * ar
            comps["audio_recon"] = ar.detach()
        if zf_weight > 0 and out.get("audio_recon") is not None:
            kf = 8  # predict activations ~0.1s ahead from the latent
            zf = ((out["audio_recon"][:, :-kf] - act[:, kf:]) ** 2).mean()
            total = total + zf_weight * zf
            comps["z_forcing"] = zf.detach()
        return total, comps, out

    run_comps: dict[str, float] = {}
    history = []

    def _dbF(metrics):
        return max(metrics.get("dec_db_F-measure", 0), metrics.get("barwrap_db_F-measure", 0))

    def fmt_eval(step, comps, metrics):
        kl = (f"kl_ph={comps.get('kl_phase',0):.3f} kl_te={comps.get('kl_tempo',0):.3f} "
              f"kl_me={comps.get('kl_meter',0):.3f} kl_bar={comps.get('kl_barphase',0):.3f}")
        bF = max(metrics.get("phase_F-measure", 0), metrics.get("dec_F-measure", 0))
        bC = max(metrics.get("phase_CMLt", 0), metrics.get("dec_CMLt", 0))
        return (f"[proxy:{cli.tag}] step {step:04d} | {kl} | bce={comps.get('bce',0):.3f} | "
                f"FREE-RUN beatF={bF:.3f} CMLt={bC:.3f} | downbeatF={_dbF(metrics):.3f} "
                f"(dec={metrics.get('dec_db_F-measure',0):.3f} barwrap={metrics.get('barwrap_db_F-measure',0):.3f})")

    bi = 0
    for step in range(1, cli.steps + 1):
        batch = batches[bi % len(batches)]
        bi += 1
        temp = _temp_at(step, cli.steps, cli.gumbel_temp_start, cli.gumbel_temp_end)
        beta = _beta_at(step, cli.steps, cli.kl_anneal_frac)
        model.scheduled_sampling_eps = _ss_at(step, cli.steps, cli.kl_anneal_frac, cli.scheduled_sampling_max)

        act = batch["activations"]
        bt = batch["beat_targets"]
        db = batch["downbeat_targets"]
        meter_tgt = batch.get("meter_onehot_prev")
        phase_tgt = batch.get("phase_prev")

        # --- He-2019 aggressive burn-in (transcribed from vendored text.py) ---
        # Optimize the ENCODER to convergence on FRESH random batches before each
        # generative-model update: inner while-loop with a sliding-window burn-loss
        # plateau break, then a decoder-ONLY outer step. Replaces our earlier
        # fixed-3-step approximation (which was NOT faithful: see DEEP_RESEARCH.md).
        # BUDGET ADAPTATION (disclosed): He uses window/max = 15/100 (text), 10/100
        # (image); each CHART "encoder step" is a full T-step sequential rollout
        # (~seconds), so a 100-iter literal loop is infeasible. We keep the algorithm
        # and shrink the budget to window/max = BURN_WINDOW/BURN_MAX.
        BURN_WINDOW, BURN_MAX = 5, 15
        if aggressive_flag:
            burn_pre, burn_cur, burn_n, sub_iter = 1e4, 0.0, 0, 0
            enc_batch = batch
            while sub_iter < BURN_MAX:
                enc_opt.zero_grad(set_to_none=True)
                inner_total, _, _ = forward_loss(
                    enc_batch["activations"], enc_batch["beat_targets"],
                    enc_batch["downbeat_targets"], enc_batch.get("meter_onehot_prev"),
                    enc_batch.get("phase_prev"), temp, beta)
                if not torch.isfinite(inner_total):
                    break
                burn_cur += float(inner_total); burn_n += 1
                inner_total.backward()
                torch.nn.utils.clip_grad_norm_(enc_params, cli.max_grad_norm)
                enc_opt.step()
                enc_batch = batches[int(torch.randint(len(batches), ()).item())]  # fresh random batch
                sub_iter += 1
                if sub_iter % BURN_WINDOW == 0:
                    burn_cur /= max(burn_n, 1)
                    if burn_pre - burn_cur < 0:          # burn-loss stopped decreasing
                        break
                    burn_pre, burn_cur, burn_n = burn_cur, 0.0, 0

        # Outer update: decoder always; encoder only when NOT aggressive (He-2019).
        enc_opt.zero_grad(set_to_none=True)
        dec_opt.zero_grad(set_to_none=True)
        total, comps, out = forward_loss(act, bt, db, meter_tgt, phase_tgt, temp, beta)
        if not torch.isfinite(total):
            print(f"[proxy:{cli.tag}] non-finite loss at step {step}, skipping")
            continue
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cli.max_grad_norm)
        if not aggressive_flag:
            enc_opt.step()
        dec_opt.step()

        # MI shutoff ("STOP BURNING", He-2019): once aggregate I(x;z) over the
        # continuous latents stops increasing across a proxy-epoch, revert to vanilla.
        if aggressive_flag and step % len(batches) == 0:
            cur_mi = _calc_mi_continuous(model, heldout, cli.eval_frames)
            if cur_mi - pre_mi < 0:
                aggressive_flag = False
                print(f"[proxy:{cli.tag}] STOP BURNING at step {step} "
                      f"(MI {pre_mi:.3f}->{cur_mi:.3f})", flush=True)
            pre_mi = cur_mi

        # EMA of components for a stable KL read-out.
        for k, v in comps.items():
            run_comps[k] = 0.9 * run_comps.get(k, float(v)) + 0.1 * float(v)

        if step % cli.eval_every == 0 or step == cli.steps:
            metrics = _heldout_freerun(model, heldout, fps, cli.eval_frames, diag=cli.diag)
            print(fmt_eval(step, run_comps, metrics), flush=True)
            if cli.diag:
                print(f"[proxy:{cli.tag}] DIAG step {step:04d} | "
                      f"db_dec F={metrics.get('dec_db_F-measure',0):.3f} "
                      f"CMLt={metrics.get('dec_db_CMLt',0):.3f} AMLt={metrics.get('dec_db_AMLt',0):.3f} | "
                      f"db_dec_NO_METER F={metrics.get('decNoMeter_db_F-measure',0):.3f} "
                      f"CMLt={metrics.get('decNoMeter_db_CMLt',0):.3f}", flush=True)
            history.append({"step": step,
                            "kl_phase": round(run_comps.get("kl_phase", 0), 4),
                            "kl_tempo": round(run_comps.get("kl_tempo", 0), 4),
                            "kl_meter": round(run_comps.get("kl_meter", 0), 4),
                            "kl_barphase": round(run_comps.get("kl_barphase", 0), 4),
                            "bce": round(run_comps.get("bce", 0), 4),
                            "beatF": round(max(metrics.get("phase_F-measure", 0), metrics.get("dec_F-measure", 0)), 4),
                            "beatCMLt": round(max(metrics.get("phase_CMLt", 0), metrics.get("dec_CMLt", 0)), 4),
                            "downbeatF": round(_dbF(metrics), 4),
                            "downbeatF_dec": round(metrics.get("dec_db_F-measure", 0), 4),
                            "downbeatF_barwrap": round(metrics.get("barwrap_db_F-measure", 0), 4)})

    final = history[-1] if history else {}
    print(f"\n[proxy:{cli.tag}] RESULT " + json.dumps({"tag": cli.tag, "idea": cli.idea,
          "baseline": cli.baseline, **final}))

    if cli.save_ckpt:
        import os as _os
        ck_ideas = _parse_ideas(cli.idea)
        ck_args = {
            "num_meter_classes": cli.num_meter_classes,
            "phase_corr_scale": cli.phase_corr_scale,
            "tempo_corr_scale": cli.tempo_corr_scale,
            "decoder_latent_only": cli.decoder_latent_only,
            "tempo_anchor_mode": cli.tempo_anchor_mode,
            "tempo_reversion_alpha": cli.tempo_reversion_alpha,
            "audio_emission": cli.audio_emission,
            "bar_phase": "bar_phase" in ck_ideas,
            "meter_ste": "meter_ste" in ck_ideas,
            "delta_vae": "delta_vae" in ck_ideas,
            "delta_vae_rate": cli.free_bits_tempo,
        }
        _os.makedirs(_os.path.dirname(cli.save_ckpt) or ".", exist_ok=True)
        torch.save({"svt_model": model.state_dict(), "args": ck_args}, cli.save_ckpt)
        print(f"[proxy:{cli.tag}] saved checkpoint -> {cli.save_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
