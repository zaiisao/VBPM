"""Cache frozen-frontend activations to disk for the fast-proxy harness.

The frontend (WaveBeat/Beat This) is FROZEN, so h_t is a pure function of the
audio. Running it on full audio is the dominant cost of every CHART experiment
(esp. the Beat This transformer). This script runs the frontend ONCE per song
and dumps {activations, beat_targets, downbeat_targets, fps} to a .pt file, so
that fast_proxy.py can train the SVT on tiny cached tensors at seconds/epoch.

Run:
    python -m tests.cache_activations \
        --extractor wavebeat --extractor_ckpt wavebeat_epoch=98-step=24749.ckpt \
        --dataset_root /home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data \
        --dataset_include ballroom --split val --max_songs 60 \
        --out_dir cache/acts/wavebeat_ballroom_val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.extractors import get_extractor_backend


def _build_args(cli: argparse.Namespace) -> argparse.Namespace:
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
    a.examples_per_epoch = 4096
    a.preload = False
    a.augment = False
    a.dry_run = False
    a.batch_size = 1
    # WaveBeat uses the .ckpt; Beat This must fall through to --beat_this_checkpoint
    # (load_checkpoint prefers extractor_ckpt, so passing the wavebeat ckpt to Beat This
    # would try to unpickle a WaveBeat PL checkpoint as Beat This -> crash).
    a.extractor_ckpt = cli.extractor_ckpt if cli.extractor == "wavebeat" else None
    # Beat This knobs (ignored by WaveBeat backend).
    a.beat_this_checkpoint = cli.beat_this_checkpoint
    a.extractor_fps_mode = cli.extractor_fps_mode
    a.beat_this_root = None
    a.beat_this_loss_tolerance = 3
    a.dist_rank = 0
    a.dist_world_size = 1
    return a


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--extractor", default="wavebeat", choices=["wavebeat", "beat_this"])
    p.add_argument("--extractor_ckpt", default="wavebeat_epoch=98-step=24749.ckpt")
    p.add_argument("--beat_this_checkpoint", default="final0")
    p.add_argument("--extractor_fps_mode", default="resample")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--dataset_include", default="ballroom")
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--max_songs", type=int, default=60)
    p.add_argument("--max_frames", type=int, default=2048)
    p.add_argument("--rich", action="store_true",
                   help="cache the frontend's penultimate [T,D] features (D=512 for Beat This) "
                        "instead of the collapsed [T,2] activations; keeps [T,2] as 'act2'.")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out_dir", required=True)
    cli = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fps = 22050 / 256
    args = _build_args(cli)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = get_extractor_backend(cli.extractor)
    loader = (backend.build_val_dataloader(args) if cli.split == "val"
              else backend.build_dataloader(args))
    if loader is None:
        print(f"[cache] no {cli.split} split for {cli.dataset_include}")
        return 1
    extractor = backend.build_model(args, device)
    backend.load_checkpoint(extractor, args, device)
    extractor.eval()
    for prm in extractor.parameters():
        prm.requires_grad = False

    def crop(x, n_):
        s = (x.shape[1] - n_) // 2
        return x[:, s:s + n_]

    n = 0
    with torch.no_grad():
        for batch in loader:
            if n >= cli.max_songs:
                break
            audio = batch["audio"].to(device)
            ext_target = batch["extractor_target"].to(device)
            beat_targets = batch["beat_targets"].to(device)

            if cli.rich:
                if cli.extractor != "beat_this":
                    raise SystemExit("--rich is implemented for the beat_this backend only")
                activations, act2 = backend.compute_hidden_and_activations(
                    extractor, audio, ext_target,
                )                                   # [1,T,D], [1,T,2]
            else:
                _, activations = backend.compute_loss_and_activations(
                    model=extractor, audio=audio, target=ext_target, frozen=True,
                )
                act2 = None
            T_ext = activations.shape[1]
            bt = crop(beat_targets, T_ext) if beat_targets.shape[1] > T_ext else beat_targets
            db = ext_target[:, 1, :]
            db = crop(db, T_ext) if db.shape[1] > T_ext else db

            # FIX (2026-06-21): crop to the BEAT REGION of the FULL activation, NOT the
            # first `max_frames`. Short songs are symmetric-padded to train_length, so the
            # real music is CENTERED in the 8192-frame clip; truncating to the first 4096
            # frames discarded the music's second half (~40% of beats) and pushed the rest
            # to the window edge -> empty center-crops downstream. Cropping to
            # [first_beat - margin, last_beat + margin] recovers the WHOLE song + drops
            # the silence padding. Pass --max_frames >= T_ext (e.g. 9000) so the cap never
            # re-truncates a genuinely long (music-throughout) clip.
            _bi = (bt[0] > 0.5).nonzero(as_tuple=True)[0]
            if _bi.numel() > 0:
                _margin = 80
                s = max(int(_bi.min()) - _margin, 0)
                e = min(int(_bi.max()) + _margin + 1, T_ext)
                if e - s > cli.max_frames:          # safety cap for very long clips
                    e = s + cli.max_frames
            else:
                s, e = 0, min(T_ext, cli.max_frames)
            acts_crop = activations[0, s:e].contiguous().cpu()           # [T, D]
            if cli.rich:
                acts_crop = acts_crop.half()                             # fp16: ~13GB/2000 songs
            rec = {
                "activations": acts_crop,                                # [T, D] (D=2 or 512)
                "beat_targets": bt[0, s:e].contiguous().cpu(),           # [T]
                "downbeat_targets": db[0, s:e].contiguous().cpu(),       # [T]
                "fps": fps,
            }
            if act2 is not None:
                rec["act2"] = act2[0, s:e].contiguous().cpu()            # [T, 2] for peak-pick ceiling
            # Structured GT latents (for the supervision-crutch baseline; the
            # AudioPhaseBridge provides these whenever phases_dir resolves).
            for key in ("phase_prev", "log_tempo_prev", "meter_onehot_prev"):
                if key in batch:
                    v = batch[key].to(device)
                    vc = crop(v, T_ext) if v.shape[1] > T_ext else v
                    rec[key] = vc[0, s:e].contiguous().cpu()
            if rec["beat_targets"].sum() < 2:
                continue
            torch.save(rec, out_dir / f"song_{n:04d}.pt")
            n += 1
            if n % 10 == 0:
                print(f"  cached {n} songs (last T={e - s})")

    print(f"[cache] wrote {n} songs to {out_dir}")
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
