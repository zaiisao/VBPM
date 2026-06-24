"""Train the faithful bar-pointer VAE END-TO-END FROM RANDOM WEIGHTS on real audio.

Objective: the STRICT ELBO (beta=1), exactly as derived. This script intentionally has
NO flags for free-bits, KL annealing, latent supervision, pos_weight, prior-mean audio
correction, tempo clamps, scheduled sampling, or extra latents -- adding any would break
faithfulness. The one schedule present (Gumbel-Softmax temperature 1.0 -> 0.3) is the
relaxation temperature used in the reference notebook, not a change to the ELBO.

The experiment this runs: watch the per-latent KL and the two free-run read-outs. The
expected, scientifically-meaningful outcome is posterior collapse (KL -> ~0, phase-wrap
F ~ metronome floor) -- the proof that the strict ELBO is collapse-prone from random init,
independent of any frozen frontend.

Usage (chart env):
  python -m faithful.train --data_root /home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data \
      --datasets ballroom,beatles,hains,rwc_popular --frames 256 --batch_size 16 \
      --steps 2000 --eval_every 200 --max_eval_songs 12 --out runs/strict_elbo
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import FPS, N_MELS, LogMel, build_train_loader, iter_val_songs
from .elbo import strict_elbo
from .evaluate import evaluate
from .model import BarPointerVAE


def parse_args():
    p = argparse.ArgumentParser(description="Faithful strict-ELBO bar-pointer VAE (end-to-end, random init)")
    p.add_argument("--data_root", required=True)
    p.add_argument("--datasets", default="ballroom,beatles,hains,rwc_popular")
    p.add_argument("--frames", type=int, default=256, help="sequence length T (frames @ 86 fps)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num_meters", type=int, default=4)
    p.add_argument("--examples_per_epoch", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--max_eval_songs", type=int, default=12, help="val songs per dataset")
    p.add_argument("--max_eval_frames", type=int, default=4000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latent_only", action="store_true",
                   help="DOCUMENTED DEVIATION: drop h from the decoder (paper §5.4 reads h)")
    p.add_argument("--out", default="runs/strict_elbo")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 78)
    print("FAITHFUL STRICT-ELBO RUN  (end-to-end from random weights)")
    print(f"  decoder reads h: {not args.latent_only}  (latent_only={args.latent_only})")
    print("  beta=1 | NO free-bits | NO KL-anneal | NO latent-sup | NO pos_weight |")
    print("  NO prior-mean audio-correction | NO tempo clamps | NO scheduled-sampling | 3 latents")
    print(f"  datasets={keys} frames={args.frames} batch={args.batch_size} steps={args.steps} lr={args.lr}")
    print("=" * 78, flush=True)

    logmel = LogMel().to(device)
    model = BarPointerVAE(h_dim=N_MELS, hidden=args.hidden,
                          num_meters=args.num_meters, latent_only=args.latent_only).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = build_train_loader(args.data_root, keys, args.frames, args.batch_size,
                                examples_per_epoch=args.examples_per_epoch,
                                num_workers=args.num_workers, seed=args.seed)
    print("materialising val songs ...", flush=True)
    val_songs = list(iter_val_songs(args.data_root, keys, max_per_dataset=args.max_eval_songs, seed=args.seed))
    print(f"  {len(val_songs)} val songs", flush=True)

    metrics_path = out / "metrics.jsonl"
    mf = metrics_path.open("w")
    best_pw = -1.0
    step = 0
    t0 = time.time()
    model.train()
    data_iter = iter(loader)
    while step < args.steps:
        try:
            audio, beats, _db = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            audio, beats, _db = next(data_iter)
        step += 1
        temperature = 1.0 + (0.3 - 1.0) * min(step / max(args.steps, 1), 1.0)  # Gumbel temp anneal

        audio = audio.to(device)
        h = logmel(audio)[:, :args.frames]               # [B, T, n_mels]
        beats = beats[:, :args.frames].to(device)
        if h.shape[1] != beats.shape[1]:
            T = min(h.shape[1], beats.shape[1])
            h, beats = h[:, :T], beats[:, :T]

        opt.zero_grad()
        loss, info = strict_elbo(model, h, beats, temperature=temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % 20 == 0 or step == 1:
            sps = step / (time.time() - t0)
            print(f"step {step:5d} | L={info['loss']:8.2f} recon={info['recon']:7.2f} "
                  f"KL={info['kl']:6.3f} (m={info['kl_meter']:.3f} phi={info['kl_phase']:.3f} "
                  f"tau={info['kl_tempo']:.3f}) | T={temperature:.2f} | {sps:.2f} it/s", flush=True)
            rec = {"step": step, "phase": "train", **{k: info[k] for k in
                   ("loss", "recon", "kl", "kl_meter", "kl_phase", "kl_tempo")}, "temp": temperature}
            mf.write(json.dumps(rec) + "\n"); mf.flush()

        if step % args.eval_every == 0 or step == args.steps:
            summary, _ = evaluate(model, logmel, val_songs, device, fps=FPS,
                                  max_frames=args.max_eval_frames)
            print(f"  [eval @ {step}] phase_wrap_F={summary['phase_wrap']:.3f} "
                  f"decoder_F={summary['decoder']:.3f} metronome_F={summary['metronome']:.3f} "
                  f"(n={summary['n_songs']})", flush=True)
            mf.write(json.dumps({"step": step, "phase": "eval", **summary}) + "\n"); mf.flush()
            torch.save({"model": model.state_dict(), "args": vars(args), "step": step},
                       out / "final.pt")
            if summary["phase_wrap"] == summary["phase_wrap"] and summary["phase_wrap"] > best_pw:
                best_pw = summary["phase_wrap"]
                torch.save({"model": model.state_dict(), "args": vars(args), "step": step,
                            "phase_wrap_F": best_pw}, out / "best.pt")

    mf.close()
    print(f"DONE. best phase_wrap_F={best_pw:.3f} | metrics -> {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
