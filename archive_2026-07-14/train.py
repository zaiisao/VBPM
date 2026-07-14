"""Train the VBPM from a YAML config: ``python train.py --config configs/default.yaml``.

One harness for every experiment; all knobs live in the config (the pinned 2026-07-10 recipe is
``configs/default.yaml``). Prints per-log-step ELBO terms plus the PRIOR (Sohn test-time pipeline)
and recognition read-outs; the step-1 gradient-reach check fails LOUDLY if any prior-side network
is starved (lesson: check gradients regularly, not only when metrics already look broken).
"""
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

from config import load_config
from data.dataset import (load_cached_songs, load_meter_only_clips, sample_meter_only_crops,
                          sample_training_crops)
from losses import auxiliary_emission_terms, negative_elbo_terms, physical_prior_anchor
from model.bar_pointer_vae import VariationalBarPointerModel
from model.readout import evaluate_geometric_readout, evaluate_prior_readout


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg):
    return VariationalBarPointerModel(
        feature_dim=cfg.frontend.feature_dim,
        hidden_size=cfg.model.hidden_size,
        num_meters=cfg.model.num_meters,
        transition_correction_scale=cfg.model.transition_correction_scale,
        decoder_input_mode=cfg.model.emission,
        fixed_prior_scales=(tuple(cfg.model.fixed_prior_scales)
                            if cfg.model.fixed_prior_scales else None))


def train(cfg):
    set_all_seeds(cfg.seed)
    # for_training=True enforces the Beat This 8-fold protocol on every record -- training on
    # frontend-memorized (final0-extracted) evidence raises FoldContaminationError, no bypass.
    train_songs = load_cached_songs(cfg.training.train_feature_dir, cfg.training.train_songs,
                                    selection_seed=1, for_training=True)
    for extra_dir in cfg.training.extra_train_dirs:   # tempo-aug pool: same songs, new tempos
        train_songs += load_cached_songs(extra_dir, cfg.training.aug_songs_per_dir,
                                         selection_seed=1, for_training=True)
    validation_songs = load_cached_songs(cfg.training.val_feature_dir, cfg.training.val_songs,
                                         selection_seed=2)
    print(f"loaded {len(train_songs)} train / {len(validation_songs)} val songs", flush=True)

    meter_only_clips = (load_meter_only_clips(cfg.training.meter_only_dir)
                        if cfg.training.meter_only_dir else [])
    if meter_only_clips:
        usable = sum(1 for _, c in meter_only_clips if c < cfg.model.num_meters)
        print(f"meter-only co-training: {usable}/{len(meter_only_clips)} clips within "
              f"{cfg.model.num_meters} classes, every {cfg.training.meter_only_every} steps", flush=True)

    model = build_model(cfg).to(cfg.device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    if cfg.training.compile_model:
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)
        print("torch.compile enabled (reduce-overhead): first steps recompile, then fused", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    obj = cfg.objective
    fps = cfg.frontend.cache_fps
    history = {"step": [], "reconstruction": [], "kl_meter": [], "kl_phase": [], "kl_tempo": [],
               "sawtooth": [], "gsnn": [], "val_step": [], "val_beat_f": [], "val_downbeat_f": [],
               "val_beat_f_prior": [], "val_downbeat_f_prior": [], "val_rotation_prior": []}

    for step in range(1, cfg.training.steps + 1):
        if obj.fivo_weight > 0.0:
            features, beat_targets, downbeat_targets, obs_crop = sample_training_crops(
                train_songs, cfg.training.crop_frames, cfg.training.batch_size,
                device=cfg.device, return_obs=True)
        else:
            features, beat_targets, downbeat_targets = sample_training_crops(
                train_songs, cfg.training.crop_frames, cfg.training.batch_size, device=cfg.device)
        gumbel_temperature = 1.0 + (0.3 - 1.0) * step / cfg.training.steps

        if cfg.model.x_only_posterior:
            # Tutorial section-7 fork: q(z | x) -- the encoder never sees the events it must
            # explain; the targets still drive every emission term below. Deployment = this same
            # encoder + geometric read-out (the recog column), gap-free by construction.
            silent = torch.zeros_like(beat_targets)
            rollout = model.rollout(features, silent, silent,
                                    gumbel_temperature=gumbel_temperature, sample=True, compute_kl=True)
        else:
            rollout = model.rollout(features, beat_targets, downbeat_targets,
                                    gumbel_temperature=gumbel_temperature, sample=True, compute_kl=True)
        negative_elbo, term_means = negative_elbo_terms(
            rollout, beat_targets, downbeat_targets, obj.free_bits_nats_per_frame,
            prior_preserving=obj.prior_preserving_free_bits, meter_ce_weight=obj.meter_ce_weight)
        aux_loss, aux_means = auxiliary_emission_terms(
            rollout, beat_targets, downbeat_targets, sawtooth_weight=obj.sawtooth_weight,
            tempo_slope_weight=obj.tempo_slope_weight, sawtooth_family=obj.sawtooth_family,
            sawtooth_wc_rho=obj.sawtooth_wc_rho, target_beats_per_bar=obj.target_beats_per_bar)
        term_means.update(aux_means)
        if obj.kl_warmup_steps > 0 and step < obj.kl_warmup_steps:
            # beta-anneal (tutorial 9.7): during warm-up, subtract the (1-beta) share of the raw KLs
            beta = step / obj.kl_warmup_steps
            raw_kl = rollout.kl_meter + rollout.kl_phase + rollout.kl_tempo
            negative_elbo = negative_elbo - (1.0 - beta) * raw_kl
        loss = obj.hybrid_alpha * negative_elbo.mean() + aux_loss
        if obj.fivo_weight > 0.0:
            # FIVO: train the deployment filter's own marginal-likelihood bound. Anneal the ELBO's
            # share down (from 1 to fivo_elbo_floor) so the untrained filter warms up first.
            from model.particle_filter import fivo_bound
            if obj.fivo_elbo_anneal_steps > 0:
                frac = min(1.0, step / obj.fivo_elbo_anneal_steps)
                elbo_scale = 1.0 - (1.0 - obj.fivo_elbo_floor) * frac
                loss = elbo_scale * loss
            if getattr(obj, 'use_grid_forward', False):
                from model.grid_decode import grid_forward_loglik
                fivo_ll = grid_forward_loglik(model, features, obs_crop)
            else:
                fivo_ll = fivo_bound(model, features, obs_crop,
                                     num_particles=obj.fivo_num_particles,
                                     gumbel_temperature=gumbel_temperature)
            loss = loss - obj.fivo_weight * fivo_ll.mean()
            term_means["fivo_ll"] = float(fivo_ll.mean())
        if obj.prior_anchor_weight > 0.0:
            anchor = physical_prior_anchor(rollout, obj.prior_anchor_sigma, obj.prior_anchor_concentration)
            if anchor is not None:
                loss = loss + obj.prior_anchor_weight * beat_targets.shape[1] * anchor
                term_means["prior_anchor"] = float(anchor)
        if obj.hybrid_alpha < 1.0:
            # Prediction pipeline (Sohn eq. 8, GSNN): z from the PRIOR network, y never an input --
            # trains the exact pipeline used at test time.
            prior_rollout = model.rollout_prior(features, gumbel_temperature=gumbel_temperature, sample=True)
            gsnn_nats = F.binary_cross_entropy_with_logits(
                prior_rollout.event_logits, torch.stack([beat_targets, downbeat_targets], dim=-1),
                reduction="none").sum(dim=(1, 2))
            loss = loss + (1.0 - obj.hybrid_alpha) * gsnn_nats.mean()
            term_means["gsnn"] = float(gsnn_nats.mean())

        if meter_only_clips and step % cfg.training.meter_only_every == 0:
            # Meter-only batch (Kingma-M2 missing data): beats/downbeats unobserved -> their
            # emission terms are dropped; the observed meter label M drives its categorical
            # emission p(M | m_t); KLs keep their free-bits floor. Event channels are zeroed
            # (the posterior conditions on features alone for these clips).
            mo_features, mo_classes = sample_meter_only_crops(
                meter_only_clips, cfg.training.crop_frames, cfg.training.batch_size,
                cfg.model.num_meters, device=cfg.device)
            silent = torch.zeros(mo_features.shape[:2], device=cfg.device)
            mo_rollout = model.rollout(mo_features, silent, silent,
                                       gumbel_temperature=gumbel_temperature,
                                       sample=True, compute_kl=True)
            kl_floor = obj.free_bits_nats_per_frame * mo_features.shape[1]
            mo_elbo = (mo_rollout.kl_meter.clamp(min=kl_floor)
                       + mo_rollout.kl_phase.clamp(min=kl_floor)
                       + mo_rollout.kl_tempo.clamp(min=kl_floor))
            if obj.prior_preserving_free_bits:
                for pg in (mo_rollout.kl_meter_pg, mo_rollout.kl_phase_pg, mo_rollout.kl_tempo_pg):
                    mo_elbo = mo_elbo + (pg - pg.detach())
            num_frames = mo_rollout.meter_logits.shape[1]
            mo_ce = F.cross_entropy(
                mo_rollout.meter_logits.permute(0, 2, 1),
                mo_classes.unsqueeze(1).expand(-1, num_frames), reduction="none").mean()
            loss = loss + mo_elbo.mean() + cfg.training.meter_only_weight * num_frames * mo_ce
            term_means["meter_only_ce"] = float(mo_ce)

        optimizer.zero_grad()
        loss.backward()
        if step == 1:   # gradient-reach check: a starved prior-side network fails LOUDLY at step 1
            for head_name in ("initial_prior_head", "prior_phase_concentration_head",
                              "prior_tempo_std_head", "meter_transition_network",
                              "transition_correction_head"):
                module = getattr(model, head_name, None)
                if module is None:
                    continue
                gradient_norm = sum(float(p.grad.norm()) for p in module.parameters()
                                    if p.grad is not None)
                print(f"    [grad-reach] {head_name}: {gradient_norm:.4f}"
                      + ("   !! ZERO -- prior-side starvation" if gradient_norm < 1e-8 else ""),
                      flush=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
        optimizer.step()
        if cfg.training.snapshot_every_steps > 0 and step % cfg.training.snapshot_every_steps == 0 and cfg.training.save_path:
            snap = cfg.training.save_path.replace('.pt', f'.step{step}.pt')
            torch.save(model.state_dict(), snap)
            print(f'    snapshot -> {snap}', flush=True)

        history["step"].append(step)
        for term_name in ("reconstruction", "kl_meter", "kl_phase", "kl_tempo"):
            history[term_name].append(term_means[term_name])
        history["sawtooth"].append(term_means.get("sawtooth", float("nan")))
        history["gsnn"].append(term_means.get("gsnn", float("nan")))

        if step % cfg.training.log_every_steps == 0 or step == cfg.training.steps:
            prior_validation = evaluate_prior_readout(
                model, validation_songs[:12], fps, cfg.training.eval_max_frames, device=cfg.device)
            recognition_validation = evaluate_geometric_readout(
                model, validation_songs[:12], fps, "real", cfg.training.eval_max_frames, device=cfg.device)
            history["val_step"].append(step)
            history["val_beat_f"].append(recognition_validation["beat_f"])
            history["val_downbeat_f"].append(recognition_validation["downbeat_f"])
            history["val_beat_f_prior"].append(prior_validation["beat_f"])
            history["val_downbeat_f_prior"].append(prior_validation["downbeat_f"])
            history["val_rotation_prior"].append(prior_validation["rotation_ratio"])
            print(f"  step {step:4d} | recon {term_means['reconstruction']:7.2f} "
                  f"| KL m {term_means['kl_meter']:6.2f} p {term_means['kl_phase']:6.2f} "
                  f"t {term_means['kl_tempo']:6.2f}"
                  + (f" | mCE {term_means['meter_ce']:5.3f}" if "meter_ce" in term_means else "")
                  + (f" | moCE {term_means['meter_only_ce']:5.3f}" if "meter_only_ce" in term_means else "")
                  + (f" | saw {term_means['sawtooth']:5.3f}" if "sawtooth" in term_means else "")
                  + (f" | tSl {term_means['tempo_slope']:5.3f}" if "tempo_slope" in term_means else "")
                  + (f" | anch {term_means['prior_anchor']:5.3f}" if "prior_anchor" in term_means else "")
                  + (f" | fivo {term_means['fivo_ll']:8.1f}" if "fivo_ll" in term_means else "")
                  + f" | PRIOR beat {prior_validation['beat_f']:.3f} db {prior_validation['downbeat_f']:.3f} "
                  f"rot {prior_validation['rotation_ratio']:.2f}"
                  f" | recog beat {recognition_validation['beat_f']:.3f} "
                  f"db {recognition_validation['downbeat_f']:.3f}",
                  flush=True)

    if cfg.training.save_path:
        torch.save(model.state_dict(), cfg.training.save_path)
        print(f"saved -> {cfg.training.save_path}", flush=True)
    return model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.save_path is not None:
        cfg.training.save_path = args.save_path
    train(cfg)


if __name__ == "__main__":
    main()
