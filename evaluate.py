"""Evaluate a trained VBPM checkpoint from a YAML config:
``python evaluate.py --config configs/default.yaml --checkpoint path.pt [--untrained-control]``.

Reports the three deployment read-outs on the validation split:
  * PRIOR    -- Sohn test-time pipeline (prior rollout at the means, geometric wrap read-out)
  * FILTER   -- particle filter, MAP and Bayesian read-outs (the headline deployment)
  * the UNTRAINED architecture-only control (--untrained-control), so every learned number can be
    quoted against the machinery baseline (e.g. clean GTZAN: trained 0.868/0.754 vs 0.615/0.548).
"""
import argparse
import dataclasses

import torch

from config import load_config
from data.dataset import load_cached_songs
from model.particle_filter import untrained_control_model
from model.readout import evaluate_filter_readout, evaluate_prior_readout
from train import build_model, set_all_seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--untrained-control", action="store_true")
    parser.add_argument("--num-songs", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_all_seeds(cfg.seed)

    songs = load_cached_songs(cfg.training.val_feature_dir,
                              args.num_songs or cfg.training.val_songs, selection_seed=2)
    print(f"loaded {len(songs)} val songs from {cfg.training.val_feature_dir}", flush=True)
    fps = cfg.frontend.cache_fps

    if args.untrained_control:
        model = untrained_control_model(
            feature_dim=cfg.frontend.feature_dim, hidden_size=cfg.model.hidden_size,
            num_meters=cfg.model.num_meters, seed=cfg.seed,
            transition_correction_scale=cfg.model.transition_correction_scale,
            decoder_input_mode=cfg.model.emission).to(cfg.device)
        tag = "UNTRAINED"
    else:
        model = build_model(cfg).to(cfg.device)
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        model.eval()
        tag = args.checkpoint or "fresh (pass --checkpoint)"

    prior = evaluate_prior_readout(model, songs, fps, cfg.training.eval_max_frames, device=cfg.device)
    print(f"RESULT {tag} | PRIOR beat_F {prior['beat_f']:.3f} downbeat_F {prior['downbeat_f']:.3f} "
          f"rot {prior['rotation_ratio']:.2f}", flush=True)
    if cfg.frontend.provides_activations and songs and songs[0].frontend_activations is not None:
        filt = evaluate_filter_readout(model, songs, fps, cfg.training.eval_max_frames,
                                       device=cfg.device, **dataclasses.asdict(cfg.filter))
        print(f"RESULT {tag} | FILTER beat_F {filt['beat_f']:.3f} downbeat_F {filt['downbeat_f']:.3f} "
              f"bayes {filt['beat_f_bayes']:.3f}/{filt['downbeat_f_bayes']:.3f} "
              f"rot {filt['rotation_ratio']:.2f}", flush=True)
    else:
        print("filter read-out skipped: no frontend activations in this cache "
              "(frontend needs a trained activation head)", flush=True)


if __name__ == "__main__":
    main()
