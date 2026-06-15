"""Gate 4 — REAL beat tracking via the inference path on HELD-OUT audio.

Loads a trained checkpoint, runs the prior-only rollout (the deployed path,
``SVTModel.sample_from_prior``) on the held-out split, and scores it with
mir_eval. Reports F-measure / CMLt / AMLt for beats AND downbeats, comparing
the phase-wrap read-out against the decoder read-out, and against a constant
120-BPM grid baseline. A model that merely drifts at ~120 BPM cannot beat the
baseline and FAILS.

Run:
    python -m tests.gate4_heldout \
        --checkpoint checkpoints/<ckpt>.pt \
        --extractor_ckpt wavebeat_epoch=98-step=24749.ckpt \
        --dataset_root /home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data \
        --dataset_include ballroom --max_songs 60
"""

from __future__ import annotations

import argparse
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

_BEAT_KEYS = ("F-measure", "CMLt", "AMLt")
_DB_KEYS = ("db_F-measure", "db_CMLt", "db_AMLt")


def _const_baseline(T, fps, bpm=120.0):
    period = 60.0 / bpm
    n = int((T / fps) / period)
    return np.arange(n, dtype=np.float64) * period


def _tempo_oracle_baseline(ref_beats, T, fps):
    """Stronger baseline: a constant grid at the song's GT median tempo, phase-
    aligned to the first GT beat. This is what a metronome that KNOWS the tempo
    and the phase would score; beating the constant-120 baseline by a wide CMLt
    margin (not just F) shows the model isn't merely hitting a fixed grid."""
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--extractor_ckpt", required=True)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--dataset_include", default="ballroom")
    p.add_argument("--max_songs", type=int, default=60)
    p.add_argument("--max_frames", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=2)
    cli = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fps = 22050 / 256
    args = _build_args(cli)

    backend = get_extractor_backend("wavebeat")
    val_loader = backend.build_val_dataloader(args)
    if val_loader is None:
        print("[Gate4] no validation split available")
        return 1
    extractor = backend.build_model(args, device)
    backend.load_checkpoint(extractor, args, device)
    extractor.eval()

    ckpt = torch.load(cli.checkpoint, map_location=device, weights_only=False)
    saved = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    K = saved.get("num_meter_classes", 8)
    # Reconstruct with the SAME correction scales used in training (defaults pi/1.0
    # if absent) — essential so the rollout dynamics match the trained weights.
    import math as _m
    model = SVTModel(
        hidden_dim=128, nhead=4, num_layers=2, num_meter_classes=K,
        phase_corr_scale=saved.get("phase_corr_scale", _m.pi),
        tempo_corr_scale=saved.get("tempo_corr_scale", 1.0),
        # These change the architecture (decoder input dim / posterior phase), so
        # they MUST match the checkpoint or load_state_dict fails.
        decoder_use_h_prior=not saved.get("decoder_latent_only", False),
        posterior_phase_recursive=saved.get("posterior_phase_recursive", False),
        # Mean-reverting tempo prior (must match ckpt: 'global' adds a head).
        tempo_anchor_mode=saved.get("tempo_anchor_mode", "none"),
        tempo_reversion_alpha=saved.get("tempo_reversion_alpha", 0.0),
        tempo_anchor_ema_beta=saved.get("tempo_anchor_ema_beta", 0.02),
    ).to(device)
    model.load_state_dict(ckpt["svt_model"] if "svt_model" in ckpt else ckpt, strict=True)
    model.eval()
    print(f"[Gate4] ckpt={cli.checkpoint} phase_corr_scale={model.phase_corr_scale:.3f} "
          f"tempo_corr_scale={model.tempo_corr_scale} "
          f"latent_only={not model.decoder_use_h_prior} "
          f"recursive_phase={model.posterior_phase_recursive}")

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    n = 0

    def acc(prefix, scores, keys):
        # Per-key counts so downbeat metrics (only scored when a song HAS >=2
        # downbeats) are averaged over their own denominator, not all songs.
        for k in keys:
            key = prefix + k
            sums[key] = sums.get(key, 0.0) + scores[k]
            counts[key] = counts.get(key, 0) + 1

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
            db = ext_target[:, 1, :]
            db = crop(db, T_ext) if db.shape[1] > T_ext else db

            Tc = min(T_ext, cli.max_frames)
            activations = activations[:, :Tc, :]
            bt = bt[:, :Tc]
            db = db[:, :Tc]

            ref_beats = frames_to_beat_times(bt[0].cpu().numpy(), fps)
            if len(ref_beats) < 2:
                continue
            ref_db = frames_to_beat_times(db[0].cpu().numpy(), fps)

            out = model.sample_from_prior(activations, temperature=cli.temperature)
            # Phase-wrap read-out uses the DETERMINISTIC mean trajectory (clean
            # sawtooth -> regular IBIs -> CMLt); falls back to the stochastic
            # sample for checkpoints saved before phase_mu existed.
            phase = out.get("phase_mu", out["phase"])[0].cpu().numpy()
            bprobs = torch.sigmoid(out["beat_logits"][0, :, 0]).cpu().numpy()
            dbprobs = torch.sigmoid(out["beat_logits"][0, :, 1]).cpu().numpy()

            acc("phase_", evaluate_beats(ref_beats, extract_beats_from_phase_trajectory(phase, fps=fps)), _BEAT_KEYS)
            acc("dec_", evaluate_beats(ref_beats, extract_beat_timestamps(bprobs, fps=fps)), _BEAT_KEYS)
            acc("base_", evaluate_beats(ref_beats, _const_baseline(Tc, fps)), _BEAT_KEYS)
            acc("tempo_oracle_", evaluate_beats(ref_beats, _tempo_oracle_baseline(ref_beats, Tc, fps)), _BEAT_KEYS)
            if len(ref_db) >= 2:
                acc("dec_", evaluate_downbeats(ref_db, extract_beat_timestamps(dbprobs, fps=fps)), _DB_KEYS)
            n += 1
            if n % 10 == 0:
                print(f"  ...scored {n} songs")

    n = max(n, 1)
    res = {k: sums[k] / max(counts[k], 1) for k in sums}
    print(f"\n[Gate4] scored {n} held-out songs ({cli.dataset_include})\n")
    print("  BEATS      F-measure   CMLt    AMLt")
    print(f"  phase-wrap  {res.get('phase_F-measure',0):.3f}     {res.get('phase_CMLt',0):.3f}   {res.get('phase_AMLt',0):.3f}")
    print(f"  decoder     {res.get('dec_F-measure',0):.3f}     {res.get('dec_CMLt',0):.3f}   {res.get('dec_AMLt',0):.3f}")
    print(f"  baseline120 {res.get('base_F-measure',0):.3f}     {res.get('base_CMLt',0):.3f}   {res.get('base_AMLt',0):.3f}")
    print(f"  tempoOracle {res.get('tempo_oracle_F-measure',0):.3f}     {res.get('tempo_oracle_CMLt',0):.3f}   {res.get('tempo_oracle_AMLt',0):.3f}")
    print("\n  DOWNBEATS  F-measure   CMLt    AMLt")
    print(f"  decoder     {res.get('dec_db_F-measure',0):.3f}     {res.get('dec_db_CMLt',0):.3f}   {res.get('dec_db_AMLt',0):.3f}")

    best_beat = max(res.get("phase_F-measure", 0), res.get("dec_F-measure", 0))
    base = res.get("base_F-measure", 0)
    ok = best_beat > base + 0.10 and best_beat > 0.25
    print(f"\n[Gate4] {'PASS' if ok else 'FAIL'}: best beat F={best_beat:.3f} vs baseline F={base:.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
