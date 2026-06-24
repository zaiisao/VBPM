"""VALIDATE the clean bar-pointer DBN against madmom on SMC.

Correctness check (NOT a hyperparameter search): run our re-implementation
(models/bar_pointer_dbn.py) at madmom's EXACT defaults (transition_lambda=100,
observation_lambda=16, 55-215 BPM, all integer tempi) on SMC, and compare its
beat-F to (a) madmom's own DBNBeatTrackingProcessor on the same Beat This beat
activation and (b) peak-picking. If ours matches madmom, the prior is faithful.

    python tests/dbn_eval_smc.py --max_songs 40
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.extractors import get_extractor_backend
from models.bar_pointer_dbn import BarPointerDBN
from evaluation.phase_converter import extract_beat_timestamps
from evaluation.score import evaluate_beats
from madmom.features.beats import DBNBeatTrackingProcessor


def _load_ref_beats(annot_path):
    arr = np.loadtxt(annot_path)
    if arr.ndim == 2:
        arr = arr[:, 0]
    return np.atleast_1d(arr).astype(np.float64)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smc_root", default="/home/sogang/jaehoon/Analyze-SMC/SMC_MIREX")
    p.add_argument("--beat_this_checkpoint", default="final0")
    p.add_argument("--extractor_fps_mode", default="resample")
    p.add_argument("--wavebeat_root", default="extractors/wavebeat")
    p.add_argument("--max_songs", type=int, default=40)
    p.add_argument("--max_frames", type=int, default=6000)
    # madmom DBNBeatTracker defaults
    p.add_argument("--min_bpm", type=float, default=55.0)
    p.add_argument("--max_bpm", type=float, default=215.0)
    p.add_argument("--transition_lambda", type=float, default=100.0)
    p.add_argument("--observation_lambda", type=int, default=16)
    cli = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fps = 22050 / 256

    smc_root = Path(cli.smc_root)
    audio_dir, annot_dir = smc_root / "SMC_MIREX_Audio", smc_root / "SMC_MIREX_Annotations"
    assert audio_dir.is_dir() and annot_dir.is_dir(), f"SMC dirs missing under {smc_root}"

    backend = get_extractor_backend("beat_this")
    ext_args = argparse.Namespace(
        beat_this_root=None, wavebeat_root=cli.wavebeat_root,
        extractor_fps_mode=cli.extractor_fps_mode, target_factor=None,
        audio_sample_rate=22050, beat_this_loss_tolerance=3,
        extractor_ckpt=None, beat_this_checkpoint=cli.beat_this_checkpoint)
    extractor = backend.build_model(ext_args, device)
    backend.load_checkpoint(extractor, ext_args, device)
    extractor.eval()

    sys.path.insert(0, str(Path(cli.wavebeat_root).resolve()))
    from wavebeat.data import DownbeatDataset  # type: ignore
    ds = DownbeatDataset(
        audio_dir=str(audio_dir), annot_dir=str(annot_dir), dataset="smc",
        audio_sample_rate=22050, target_factor=256, subset="full-val",
        length=2097152, preload=False, augment=False, examples_per_epoch=1000,
        half=False, dry_run=False)

    # ours @ madmom defaults (all integer tempi -> num_intervals=None)
    dbn = BarPointerDBN(fps=fps, beats_only=True, num_intervals=None, learnable_lambda=False,
                        init_lambda=cli.transition_lambda, observation_lambda=cli.observation_lambda,
                        min_bpm=cli.min_bpm, max_bpm=cli.max_bpm).to(device)
    # madmom's own DBN (same params)
    mm = DBNBeatTrackingProcessor(min_bpm=cli.min_bpm, max_bpm=cli.max_bpm, fps=fps,
                                  transition_lambda=cli.transition_lambda,
                                  observation_lambda=cli.observation_lambda)
    print(f"[SMC-validate] ours {dbn.num_states} states @ lambda={cli.transition_lambda} "
          f"obs={cli.observation_lambda} {cli.min_bpm:g}-{cli.max_bpm:g} BPM")

    ours, madmom_f, peak = [], [], []
    n = 0
    with torch.no_grad():
        for idx in range(len(ds)):
            if n >= cli.max_songs:
                break
            audio, target, _ = ds[idx]
            ref_beats = _load_ref_beats(ds.annot_files[idx])
            if len(ref_beats) < 2:
                continue
            audio = audio.float().unsqueeze(0).to(device)
            tgt = target.float().unsqueeze(0).to(device)
            max_samples = cli.max_frames * 256
            if audio.shape[-1] > max_samples:
                audio = audio[..., :max_samples]
                tgt = tgt[..., :min(tgt.shape[-1], cli.max_frames)]
            _, act = backend.compute_loss_and_activations(
                model=extractor, audio=audio, target=tgt, frozen=True)
            Tc = min(act.shape[1], cli.max_frames)
            act = act[0, :Tc].float()
            dur = Tc / fps
            ref = ref_beats[ref_beats < dur]
            if len(ref) < 2:
                continue
            n += 1
            beat_act = act[:, 0].cpu().numpy()
            peak.append(evaluate_beats(ref, extract_beat_timestamps(beat_act, fps=fps))["F-measure"])
            bfr, _ = dbn.decode(act)
            ours.append(evaluate_beats(ref, bfr.cpu().numpy().astype(float) / fps)["F-measure"])
            madmom_f.append(evaluate_beats(ref, np.asarray(mm(beat_act)))["F-measure"])

    print(f"\n[SMC-validate] {n} songs  (Beat This published SMC F = 0.62)\n")
    print(f"  peak-pick (no dynamics):        {np.mean(peak):.3f}")
    print(f"  madmom DBNBeatTracker:          {np.mean(madmom_f):.3f}")
    print(f"  OURS (clean reimpl, same args): {np.mean(ours):.3f}   <- should match madmom")
    print(f"\n  |ours - madmom| per-song mean = {np.mean(np.abs(np.array(ours)-np.array(madmom_f))):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
