"""Gate 2 — overfit a single REAL batch.

Sanity that the (fixed) architecture can fit real data: take one batch of real
audio, freeze the WaveBeat extractor, cache its activations once, then train the
SVT on that single batch for many steps. The reconstruction BCE must collapse
toward 0 and the decoder beat F-measure must approach 1.0. We also report the
prior-rollout F-measure on the same batch as a forward-looking signal for Gate 4.

Run:
    python -m tests.gate2_overfit_real \
        --dataset_root /home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data \
        --extractor_ckpt wavebeat_epoch=98-step=24749.ckpt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.loss import compute_elbo_loss
from models.svt_core import SVTModel
from evaluation.phase_converter import (
    extract_beat_timestamps,
    extract_beats_from_phase_trajectory,
)
from evaluation.score import evaluate_beats, frames_to_beat_times
from training.extractors import get_extractor_backend


def _build_args(cli: argparse.Namespace) -> argparse.Namespace:
    a = argparse.Namespace()
    a.wavebeat_root = "extractors/wavebeat"
    a.dataset_root = cli.dataset_root
    a.dataset_include = cli.dataset_include
    a.phases_dir = None
    a.audio_dir = None
    a.annot_dir = None
    a.wavebeat_dataset = "ballroom"
    a.audio_sample_rate = 22050
    a.target_factor = 256
    a.train_length = 2097152
    a.num_workers = 0
    a.examples_per_epoch = cli.batch_size * 4
    a.preload = False
    a.augment = False
    a.dry_run = False
    a.batch_size = cli.batch_size
    a.extractor_ckpt = cli.extractor_ckpt
    a.dist_rank = 0
    a.dist_world_size = 1
    return a


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--dataset_include", default="ballroom")
    p.add_argument("--extractor_ckpt", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--frames", type=int, default=384)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--pos_weight", type=float, default=5.0,
                   help="BCE positive weight for the beat channel (beats are sparse; 1.0 collapses to all-zeros).")
    p.add_argument("--pos_weight_db", type=float, default=15.0)
    cli = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fps = cli_fps = 22050 / 256
    args = _build_args(cli)

    backend = get_extractor_backend("wavebeat")
    loader = backend.build_dataloader(args)
    extractor = backend.build_model(args, device)
    backend.load_checkpoint(extractor, args, device)
    extractor.eval()
    for prm in extractor.parameters():
        prm.requires_grad = False

    batch = next(iter(loader))
    audio = batch["audio"].to(device)
    ext_target = batch["extractor_target"].to(device)
    beat_targets = batch["beat_targets"].to(device)

    # Cache activations ONCE (frozen extractor).
    with torch.no_grad():
        _, activations = backend.compute_loss_and_activations(
            model=extractor, audio=audio, target=ext_target, frozen=True,
        )
    T_ext = activations.shape[1]

    # Align + crop targets to T_ext, then cap to `frames`.
    def crop(x, n):
        s = (x.shape[1] - n) // 2
        return x[:, s:s + n]
    beat_aligned = crop(beat_targets, T_ext) if beat_targets.shape[1] > T_ext else beat_targets
    db_aligned = crop(ext_target[:, 1, :], T_ext) if ext_target.shape[2] > T_ext else ext_target[:, 1, :]

    Tc = min(T_ext, cli.frames)
    s = (T_ext - Tc) // 2
    activations = activations[:, s:s + Tc, :].contiguous()
    bt = beat_aligned[:, s:s + Tc].contiguous()
    db = db_aligned[:, s:s + Tc].contiguous()

    print(f"[Gate2] batch={activations.shape} beats/seq={bt.sum(1).tolist()} fps={fps:.2f}")

    model = SVTModel(hidden_dim=128, nhead=4, num_layers=2, num_meter_classes=8).to(device)
    opt = optim.Adam(model.parameters(), lr=cli.lr)

    def report(tag, temp):
        model.eval()
        with torch.no_grad():
            out = model(activations, temperature=temp, beat_targets=bt, downbeat_targets=db)
            _, comps = compute_elbo_loss(
                out["beat_logits"], bt, out["posterior"], out["prior"], downbeat_targets=db,
                pos_weight=cli.pos_weight, pos_weight_db=cli.pos_weight_db,
            )
            probs = torch.sigmoid(out["beat_logits"][:, :, 0]).cpu().numpy()
            phase = out["samples"]["phase"].cpu().numpy()
            prior = model.sample_from_prior(activations, temperature=temp)
            prior_phase = prior["phase"].cpu().numpy()
            prior_probs = torch.sigmoid(prior["beat_logits"][:, :, 0]).cpu().numpy()
        bt_np = bt.cpu().numpy()
        f_dec = f_pphase = f_pdec = 0.0
        n = 0
        for b in range(activations.shape[0]):
            ref = frames_to_beat_times(bt_np[b], fps)
            if len(ref) < 2:
                continue
            f_dec += evaluate_beats(ref, extract_beat_timestamps(probs[b], fps=fps))["F-measure"]
            f_pphase += evaluate_beats(ref, extract_beats_from_phase_trajectory(prior_phase[b], fps=fps))["F-measure"]
            f_pdec += evaluate_beats(ref, extract_beat_timestamps(prior_probs[b], fps=fps))["F-measure"]
            n += 1
        n = max(n, 1)
        print(f"[Gate2:{tag}] bce={comps['bce']:.4f} kl_phase={comps['kl_phase']:.3f} "
              f"kl_tempo={comps['kl_tempo']:.3f} kl_meter={comps['kl_meter']:.3f} | "
              f"F_decoder(post)={f_dec/n:.3f} F_prior_phase={f_pphase/n:.3f} F_prior_dec={f_pdec/n:.3f}")
        model.train()
        return comps["bce"].item(), f_dec / n, f_pdec / n, f_pphase / n

    report("init", 1.0)
    for step in range(cli.steps):
        temp = max(0.1, 1.0 - step / cli.steps)
        opt.zero_grad()
        out = model(activations, temperature=temp, beat_targets=bt, downbeat_targets=db)
        total, _ = compute_elbo_loss(
            out["beat_logits"], bt, out["posterior"], out["prior"], downbeat_targets=db,
            pos_weight=cli.pos_weight, pos_weight_db=cli.pos_weight_db,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0:
            report(f"step{step+1}", temp)

    bce_final, f_dec_final, f_pdec_final, f_pphase_final = report("final", 0.1)
    # Gate 2 passes if the architecture FITS the batch (decoder reproduces beats)
    # AND the INFERENCE path (prior-only rollout) also fits — the latter is the
    # direct check that the P0 audio-driven prior learned on this data.
    ok = f_dec_final > 0.8 and f_pdec_final > 0.5
    print(f"\n[Gate2] {'PASS' if ok else 'FAIL'}: "
          f"bce_final={bce_final:.4f} | F_decoder(post)={f_dec_final:.3f} (>0.8) | "
          f"F_prior_dec={f_pdec_final:.3f} (>0.5) | F_prior_phase={f_pphase_final:.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
