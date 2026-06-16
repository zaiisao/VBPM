"""PF-eval — particle-filter inference vs the open-loop prior rollout on HELD-OUT audio.

Loads a trained (audio_emission) checkpoint and, per held-out song, scores:
  * sample_from_prior      — the OPEN-LOOP rollout (the exposure-biased baseline)
  * sample_from_prior_pf   — the CLOSED-LOOP particle filter (weighted by p(h|z)),
                             swept over a few obs_sigma values.
Reports F / CMLt / AMLt for the phase-wrap and decoder read-outs of each, against
the constant-120 and tempo-oracle baselines. The question: does closing the loop
with the audio-emission likelihood beat the open-loop CMLt (diagnostic 0.28)?

Run:
    python tests/pf_eval.py \
        --checkpoint checkpoints/ou5_dir1/chart.pt \
        --extractor_ckpt wavebeat_epoch=98-step=24749.ckpt \
        --dataset_root /home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data \
        --dataset_include ballroom --max_songs 30 --obs_sigma 0.2,0.4 --n_particles 300
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
from evaluation.score import evaluate_beats, frames_to_beat_times
from training.extractors import get_extractor_backend

_BEAT_KEYS = ("F-measure", "CMLt", "AMLt")


def _const_baseline(T, fps, bpm=120.0):
    period = 60.0 / bpm
    n = int((T / fps) / period)
    return np.arange(n, dtype=np.float64) * period


def _tempo_oracle_baseline(ref_beats, T, fps):
    if len(ref_beats) < 2:
        return np.zeros(0, dtype=np.float64)
    period = float(np.median(np.diff(ref_beats)))
    if period <= 0:
        return np.zeros(0, dtype=np.float64)
    start = float(ref_beats[0])
    n = int(((T / fps) - start) / period)
    return start + np.arange(max(n, 0), dtype=np.float64) * period


def _build_args(cli):
    a = argparse.Namespace()
    a.wavebeat_root = "extractors/wavebeat"
    a.dataset_root = cli.dataset_root
    a.dataset_include = cli.dataset_include
    a.phases_dir = None
    a.audio_dir = a.annot_dir = None
    a.wavebeat_dataset = "ballroom"
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
    a.dist_rank = 0
    a.dist_world_size = 1
    return a


def _build_model(cli, device):
    ckpt = torch.load(cli.checkpoint, map_location=device, weights_only=False)
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
    print(f"[PF] ckpt={cli.checkpoint} anchor={model.tempo_anchor_mode} "
          f"alpha={model.tempo_reversion_alpha} latent_only={not model.decoder_use_h_prior} "
          f"audio_emission={model.audio_emission}")
    if not model.audio_emission:
        raise SystemExit("[PF] checkpoint has no audio_emission head — PF needs Dir 1A.")
    return model


class Acc:
    def __init__(self):
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, prefix, scores, keys=_BEAT_KEYS):
        for k in keys:
            key = prefix + k
            self.sums[key] = self.sums.get(key, 0.0) + scores[k]
            self.counts[key] = self.counts.get(key, 0) + 1

    def get(self, key):
        return self.sums.get(key, 0.0) / max(self.counts.get(key, 1), 1)


def _readout(out, ref_beats, fps, acc, prefix):
    phase = out.get("phase_mu", out["phase"])[0].cpu().numpy()
    bprobs = torch.sigmoid(out["beat_logits"][0, :, 0]).cpu().numpy()
    acc.add(prefix + "phase_", evaluate_beats(
        ref_beats, extract_beats_from_phase_trajectory(phase, fps=fps)))
    acc.add(prefix + "dec_", evaluate_beats(
        ref_beats, extract_beat_timestamps(bprobs, fps=fps)))
    # Bayesian wrap read-out (PF only): weighted per-frame beat probability.
    if "beat_activation" in out:
        ba = out["beat_activation"][0].cpu().numpy()
        acc.add(prefix + "wrap_", evaluate_beats(
            ref_beats, extract_beat_timestamps(ba, fps=fps)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--extractor_ckpt", required=True)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--dataset_include", default="ballroom")
    p.add_argument("--max_songs", type=int, default=30)
    p.add_argument("--max_frames", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--n_particles", type=int, default=300)
    p.add_argument("--obs_sigma", default="0.2,0.4",
                   help="comma-separated obs_sigma values to sweep")
    p.add_argument("--ess_frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    cli = p.parse_args()

    torch.manual_seed(cli.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fps = 22050 / 256
    sigmas = [float(s) for s in cli.obs_sigma.split(",") if s.strip()]
    args = _build_args(cli)

    backend = get_extractor_backend("wavebeat")
    val_loader = backend.build_val_dataloader(args)
    if val_loader is None:
        print("[PF] no validation split available")
        return 1
    extractor = backend.build_model(args, device)
    backend.load_checkpoint(extractor, args, device)
    extractor.eval()
    model = _build_model(cli, device)

    acc = Acc()
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            if n >= cli.max_songs:
                break
            audio = batch["audio"].to(device)
            ext_target = batch["extractor_target"].to(device)
            beat_targets = batch["beat_targets"].to(device)

            _, activations = backend.compute_loss_and_activations(
                model=extractor, audio=audio, target=ext_target, frozen=True,
            )
            T_ext = activations.shape[1]

            def crop(x, n_):
                s = (x.shape[1] - n_) // 2
                return x[:, s:s + n_]
            bt = crop(beat_targets, T_ext) if beat_targets.shape[1] > T_ext else beat_targets

            Tc = min(T_ext, cli.max_frames)
            activations = activations[:, :Tc, :]
            bt = bt[:, :Tc]

            ref_beats = frames_to_beat_times(bt[0].cpu().numpy(), fps)
            if len(ref_beats) < 2:
                continue

            # Open-loop baseline
            _readout(model.sample_from_prior(activations, temperature=cli.temperature),
                     ref_beats, fps, acc, "openloop.")
            # Closed-loop PF, swept over obs_sigma
            for s in sigmas:
                out = model.sample_from_prior_pf(
                    activations, n_particles=cli.n_particles, obs_sigma=s,
                    temperature=cli.temperature, ess_frac=cli.ess_frac,
                )
                _readout(out, ref_beats, fps, acc, f"pf{s}.")

            # Static baselines
            acc.add("base_", evaluate_beats(ref_beats, _const_baseline(Tc, fps)))
            acc.add("oracle_", evaluate_beats(ref_beats, _tempo_oracle_baseline(ref_beats, Tc, fps)))
            n += 1
            if n % 5 == 0:
                print(f"  ...scored {n} songs")

    n = max(n, 1)
    print(f"\n[PF] scored {n} held-out songs ({cli.dataset_include}), "
          f"N={cli.n_particles} ess_frac={cli.ess_frac}\n")
    hdr = "  {:<16s} {:>9s} {:>7s} {:>7s}".format("METHOD/readout", "F", "CMLt", "AMLt")
    print(hdr)

    def row(label, prefix):
        print("  {:<16s} {:>9.3f} {:>7.3f} {:>7.3f}".format(
            label, acc.get(prefix + "F-measure"), acc.get(prefix + "CMLt"),
            acc.get(prefix + "AMLt")))

    row("openloop phase", "openloop.phase_")
    row("openloop dec", "openloop.dec_")
    for s in sigmas:
        row(f"PF s={s} phase", f"pf{s}.phase_")
        row(f"PF s={s} dec", f"pf{s}.dec_")
        row(f"PF s={s} wrap", f"pf{s}.wrap_")
    row("baseline120", "base_")
    row("tempoOracle", "oracle_")

    # Headline: best PF CMLt vs open-loop CMLt
    ol = max(acc.get("openloop.phase_CMLt"), acc.get("openloop.dec_CMLt"))
    pf = max(
        max(acc.get(f"pf{s}.phase_CMLt"), acc.get(f"pf{s}.dec_CMLt"), acc.get(f"pf{s}.wrap_CMLt"))
        for s in sigmas
    )
    print(f"\n[PF] best CMLt: open-loop={ol:.3f}  particle-filter={pf:.3f}  "
          f"(Δ={pf - ol:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
