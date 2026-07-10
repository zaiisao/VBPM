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
from data.dataset import load_cached_songs, sample_training_crops
from losses import auxiliary_emission_terms, negative_elbo_terms
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

    model = build_model(cfg).to(cfg.device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    obj = cfg.objective
    fps = cfg.frontend.cache_fps
    history = {"step": [], "reconstruction": [], "kl_meter": [], "kl_phase": [], "kl_tempo": [],
               "sawtooth": [], "gsnn": [], "val_step": [], "val_beat_f": [], "val_downbeat_f": [],
               "val_beat_f_prior": [], "val_downbeat_f_prior": [], "val_rotation_prior": []}

    for step in range(1, cfg.training.steps + 1):
        features, beat_targets, downbeat_targets = sample_training_crops(
            train_songs, cfg.training.crop_frames, cfg.training.batch_size, device=cfg.device)
        gumbel_temperature = 1.0 + (0.3 - 1.0) * step / cfg.training.steps

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
        loss = obj.hybrid_alpha * negative_elbo.mean() + aux_loss
        if obj.hybrid_alpha < 1.0:
            # Prediction pipeline (Sohn eq. 8, GSNN): z from the PRIOR network, y never an input --
            # trains the exact pipeline used at test time.
            prior_rollout = model.rollout_prior(features, gumbel_temperature=gumbel_temperature, sample=True)
            gsnn_nats = F.binary_cross_entropy_with_logits(
                prior_rollout.event_logits, torch.stack([beat_targets, downbeat_targets], dim=-1),
                reduction="none").sum(dim=(1, 2))
            loss = loss + (1.0 - obj.hybrid_alpha) * gsnn_nats.mean()
            term_means["gsnn"] = float(gsnn_nats.mean())

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
                  + (f" | saw {term_means['sawtooth']:5.3f}" if "sawtooth" in term_means else "")
                  + (f" | tSl {term_means['tempo_slope']:5.3f}" if "tempo_slope" in term_means else "")
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
