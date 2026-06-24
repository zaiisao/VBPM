"""Per-dataset, beat & downbeat held-out eval: CHART particle filter vs raw frontend.

Breaks the held-out evaluation down BY DATASET and BY beat/downbeat, so in-distribution
(ballroom/beatles/hains/rwc) vs cross-dataset (gtzan) generalization is visible, and the
bar-structure (downbeat) read-out is scored separately from beats. Works for either
frontend (--frontend wavebeat|beat_this). SMC is handled separately (beats-only) by
pf_eval_smc.py.

Run:
    python tests/pf_eval_byds.py --checkpoint checkpoints/ou5_dir1/chart_ep004_f0.3871.pt \
        --frontend wavebeat --extractor_ckpt wavebeat_epoch=98-step=24749.ckpt \
        --dataset_root /home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data \
        --dataset_include ballroom,beatles,hains,rwc_popular,gtzan --max_songs 20
"""

from __future__ import annotations

import argparse
import math as _m
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.svt_core import SVTModel
from evaluation.phase_converter import (
    extract_beat_timestamps,
    extract_beats_from_phase_trajectory,
)
from evaluation.score import evaluate_beats, evaluate_downbeats, frames_to_beat_times
from training.extractors import get_extractor_backend

_BK = ("F-measure", "CMLt", "AMLt")
_DK = ("db_F-measure", "db_CMLt", "db_AMLt")


def _build_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    K = saved.get("num_meter_classes", 8)
    model = SVTModel(
        hidden_dim=128, nhead=4, num_layers=2, num_meter_classes=K,
        phase_corr_scale=saved.get("phase_corr_scale", _m.pi),
        tempo_corr_scale=saved.get("tempo_corr_scale", 1.0),
        decoder_use_h_prior=not saved.get("decoder_latent_only", False),
        posterior_phase_recursive=saved.get("posterior_phase_recursive", False),
        tempo_anchor_mode=saved.get("tempo_anchor_mode", "none"),
        tempo_reversion_alpha=saved.get("tempo_reversion_alpha", 0.0),
        tempo_anchor_ema_beta=saved.get("tempo_anchor_ema_beta", 0.02),
        audio_emission=saved.get("audio_emission", False),
    ).to(device)
    model.load_state_dict(ckpt["svt_model"] if "svt_model" in ckpt else ckpt, strict=True)
    model.eval()
    return model


def _loader_args(cli, dataset):
    a = argparse.Namespace()
    a.wavebeat_root = cli.wavebeat_root
    a.beat_this_root = None
    a.dataset_root = cli.dataset_root
    a.dataset_include = dataset
    a.phases_dir = None
    a.audio_dir = a.annot_dir = None
    a.wavebeat_dataset = dataset
    a.audio_sample_rate = 22050
    a.target_factor = 256
    a.train_length = 2097152
    a.num_workers = cli.num_workers
    a.examples_per_epoch = 1000
    a.preload = False
    a.augment = False
    a.dry_run = False
    a.batch_size = 1
    a.extractor_ckpt = cli.extractor_ckpt
    a.beat_this_checkpoint = cli.beat_this_checkpoint
    a.extractor_fps_mode = cli.extractor_fps_mode
    a.beat_this_loss_tolerance = 3
    a.dist_rank = 0
    a.dist_world_size = 1
    return a


class Acc:
    def __init__(self):
        self.s: dict[str, float] = {}
        self.c: dict[str, int] = {}

    def add(self, prefix, scores, keys):
        for k in keys:
            self.s[prefix + k] = self.s.get(prefix + k, 0.0) + scores[k]
            self.c[prefix + k] = self.c.get(prefix + k, 0) + 1

    def g(self, key):
        return self.s.get(key, 0.0) / max(self.c.get(key, 1), 1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--frontend", default="wavebeat", choices=["wavebeat", "beat_this"])
    p.add_argument("--extractor_ckpt", default=None)
    p.add_argument("--beat_this_checkpoint", default="final0")
    p.add_argument("--extractor_fps_mode", default="resample", choices=["resample", "native"])
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--dataset_include", default="ballroom,beatles,hains,rwc_popular,gtzan")
    p.add_argument("--max_songs", type=int, default=20, help="per dataset")
    p.add_argument("--max_frames", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--n_particles", type=int, default=200)
    p.add_argument("--obs_sigma", type=float, default=0.15)
    p.add_argument("--ess_frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wavebeat_root", default="extractors/wavebeat")
    cli = p.parse_args()

    torch.manual_seed(cli.seed)
    import random
    random.seed(cli.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fps = 22050 / 256

    backend = get_extractor_backend(cli.frontend)
    if cli.frontend == "beat_this":
        ext_args = argparse.Namespace(
            beat_this_root=None, wavebeat_root=cli.wavebeat_root,
            extractor_fps_mode=cli.extractor_fps_mode, target_factor=None,
            audio_sample_rate=22050, beat_this_loss_tolerance=3,
            extractor_ckpt=None, beat_this_checkpoint=cli.beat_this_checkpoint,
        )
    else:
        if cli.extractor_ckpt is None:
            raise SystemExit("--extractor_ckpt required for wavebeat frontend")
        ext_args = argparse.Namespace(wavebeat_root=cli.wavebeat_root, extractor_ckpt=cli.extractor_ckpt)
    extractor = backend.build_model(ext_args, device)
    backend.load_checkpoint(extractor, ext_args, device)
    extractor.eval()
    model = _build_model(cli.checkpoint, device)
    print(f"[byds] ckpt={cli.checkpoint} frontend={cli.frontend} "
          f"audio_emission={model.audio_emission}")

    datasets = [d.strip() for d in cli.dataset_include.split(",") if d.strip()]
    acc = Acc()
    counts: dict[str, int] = {}

    def crop(x, n_):
        s = (x.shape[1] - n_) // 2
        return x[:, s:s + n_]

    with torch.no_grad():
        for ds in datasets:
            loader = backend.build_val_dataloader(_loader_args(cli, ds))
            if loader is None:
                print(f"  [skip] {ds}: no val loader")
                continue
            n = 0
            for batch in loader:
                if n >= cli.max_songs:
                    break
                audio = batch["audio"].to(device)
                ext_target = batch["extractor_target"].to(device)
                beat_targets = batch["beat_targets"].to(device)
                _, activations = backend.compute_loss_and_activations(
                    model=extractor, audio=audio, target=ext_target, frozen=True,
                )
                T_ext = activations.shape[1]
                bt = crop(beat_targets, T_ext) if beat_targets.shape[1] > T_ext else beat_targets
                db = ext_target[:, 1, :]
                db = crop(db, T_ext) if db.shape[1] > T_ext else db
                Tc = min(T_ext, cli.max_frames)
                activations = activations[:, :Tc, :]
                bt = bt[:, :Tc]; db = db[:, :Tc]

                ref_b = frames_to_beat_times(bt[0].cpu().numpy(), fps)
                ref_d = frames_to_beat_times(db[0].cpu().numpy(), fps)
                if len(ref_b) < 2:
                    continue

                # raw frontend: peak-pick its own beat/downbeat channels
                acc.add(f"{ds}|raw|beat|", evaluate_beats(
                    ref_b, extract_beat_timestamps(activations[0, :, 0].cpu().numpy(), fps=fps)), _BK)
                if len(ref_d) >= 2:
                    acc.add(f"{ds}|raw|db|", evaluate_downbeats(
                        ref_d, extract_beat_timestamps(activations[0, :, 1].cpu().numpy(), fps=fps)), _DK)

                # CHART particle filter
                out = model.sample_from_prior_pf(
                    activations, n_particles=cli.n_particles, obs_sigma=cli.obs_sigma,
                    temperature=cli.temperature, ess_frac=cli.ess_frac,
                )
                phase = out.get("phase_mu", out["phase"])[0].cpu().numpy()
                bprob = torch.sigmoid(out["beat_logits"][0, :, 0]).cpu().numpy()
                dbprob = torch.sigmoid(out["beat_logits"][0, :, 1]).cpu().numpy()
                acc.add(f"{ds}|pf_phase|beat|", evaluate_beats(
                    ref_b, extract_beats_from_phase_trajectory(phase, fps=fps)), _BK)
                acc.add(f"{ds}|pf_dec|beat|", evaluate_beats(
                    ref_b, extract_beat_timestamps(bprob, fps=fps)), _BK)
                if len(ref_d) >= 2:
                    acc.add(f"{ds}|pf_dec|db|", evaluate_downbeats(
                        ref_d, extract_beat_timestamps(dbprob, fps=fps)), _DK)
                n += 1
            counts[ds] = n
            print(f"  scored {ds}: {n} songs")

    # ---- report ----
    print(f"\n[byds] {cli.frontend} frontend — per-dataset (beat | downbeat)\n")
    hdr = "  {:<12s} {:<10s} {:>6s} {:>6s} {:>6s}   {:>6s} {:>6s} {:>6s}".format(
        "dataset", "method", "bF", "bCMLt", "bAMLt", "dF", "dCMLt", "dAMLt")
    print(hdr); print("  " + "-" * len(hdr))
    for ds in datasets:
        if counts.get(ds, 0) == 0:
            continue
        for method, label in [("raw", "rawFrontend"), ("pf_phase", "CHART-PF phase"),
                              ("pf_dec", "CHART-PF dec")]:
            bF = acc.g(f"{ds}|{method}|beat|F-measure")
            bC = acc.g(f"{ds}|{method}|beat|CMLt")
            bA = acc.g(f"{ds}|{method}|beat|AMLt")
            dF = acc.g(f"{ds}|{method}|db|db_F-measure")
            dC = acc.g(f"{ds}|{method}|db|db_CMLt")
            dA = acc.g(f"{ds}|{method}|db|db_AMLt")
            db_str = (f"{dF:>6.3f} {dC:>6.3f} {dA:>6.3f}" if method != "pf_phase" else
                      f"{'-':>6s} {'-':>6s} {'-':>6s}")
            print(f"  {ds:<12s} {label:<10s} {bF:>6.3f} {bC:>6.3f} {bA:>6.3f}   {db_str}")
        print()
    print(f"  songs/dataset: " + ", ".join(f"{d}={counts.get(d,0)}" for d in datasets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
