"""PF-eval on SMC — the hard, held-out "blind spot" set (beats only, NOT trained on).

SMC_MIREX is expressive/rubato music where discriminative trackers octave-error and
drift (Holzapfel et al. 2012; "the SMC blind spot"). It is NOT in CHART's training
set (ballroom,beatles,hains,rwc_popular). This scores the closed-loop particle
filter vs the open-loop prior rollout on full-length SMC excerpts, beats only.

Audio is loaded via the WaveBeat DownbeatDataset (exact preprocessing); ground-truth
beat times are read straight from the SMC annotation .txt (no frame round-trip).

Run:
    python tests/pf_eval_smc.py \
        --checkpoint checkpoints/ou5_dir1/chart_ep001_f0.0000.pt \
        --extractor_ckpt wavebeat_epoch=98-step=24749.ckpt \
        --smc_root /home/sogang/jaehoon/Analyze-SMC/SMC_MIREX \
        --max_songs 40 --obs_sigma 0.15,0.25 --n_particles 300
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
from evaluation.score import evaluate_beats
from training.extractors import get_extractor_backend

_BEAT_KEYS = ("F-measure", "CMLt", "AMLt")


def _const_baseline(dur_sec, bpm=120.0):
    period = 60.0 / bpm
    n = int(dur_sec / period)
    return np.arange(n, dtype=np.float64) * period


def _tempo_oracle_baseline(ref_beats, dur_sec):
    if len(ref_beats) < 2:
        return np.zeros(0, dtype=np.float64)
    period = float(np.median(np.diff(ref_beats)))
    if period <= 0:
        return np.zeros(0, dtype=np.float64)
    start = float(ref_beats[0])
    n = int((dur_sec - start) / period)
    return start + np.arange(max(n, 0), dtype=np.float64) * period


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
        bar_phase=saved.get("bar_phase", False),
        meter_ste=saved.get("meter_ste", False),
    ).to(device)
    model.load_state_dict(ckpt["svt_model"] if "svt_model" in ckpt else ckpt, strict=True)
    model.eval()
    print(f"[SMC] ckpt={ckpt_path} anchor={model.tempo_anchor_mode} "
          f"alpha={model.tempo_reversion_alpha} latent_only={not model.decoder_use_h_prior} "
          f"audio_emission={model.audio_emission}")
    if not model.audio_emission:
        raise SystemExit("[SMC] checkpoint has no audio_emission head — PF needs Dir 1A.")
    return model


class Acc:
    def __init__(self):
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, prefix, scores, keys=_BEAT_KEYS):
        for k in keys:
            self.sums[prefix + k] = self.sums.get(prefix + k, 0.0) + scores[k]
            self.counts[prefix + k] = self.counts.get(prefix + k, 0) + 1

    def get(self, key):
        return self.sums.get(key, 0.0) / max(self.counts.get(key, 1), 1)


def _readout(out, ref_beats, fps, acc, prefix):
    phase = out.get("phase_mu", out["phase"])[0].cpu().numpy()
    bprobs = torch.sigmoid(out["beat_logits"][0, :, 0]).cpu().numpy()
    acc.add(prefix + "phase_", evaluate_beats(
        ref_beats, extract_beats_from_phase_trajectory(phase, fps=fps)))
    acc.add(prefix + "dec_", evaluate_beats(
        ref_beats, extract_beat_timestamps(bprobs, fps=fps)))
    if "beat_activation" in out:
        ba = out["beat_activation"][0].cpu().numpy()
        ba = ba / (ba.max() + 1e-8)
        acc.add(prefix + "wrap_", evaluate_beats(
            ref_beats, extract_beat_timestamps(ba, fps=fps)))


def _load_ref_beats(annot_path):
    arr = np.loadtxt(annot_path)
    if arr.ndim == 2:          # (time, beat_idx) -> take the time column
        arr = arr[:, 0]
    return np.atleast_1d(arr).astype(np.float64)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--frontend", default="wavebeat", choices=["wavebeat", "beat_this"],
                   help="acoustic frontend the CHART checkpoint was trained on")
    p.add_argument("--extractor_ckpt", default=None,
                   help="WaveBeat extractor ckpt (wavebeat frontend only)")
    p.add_argument("--beat_this_checkpoint", default="final0",
                   help="Beat This pretrained shortname/path (beat_this frontend only)")
    p.add_argument("--extractor_fps_mode", default="resample", choices=["resample", "native"])
    p.add_argument("--smc_root", default="/home/sogang/jaehoon/Analyze-SMC/SMC_MIREX")
    p.add_argument("--max_songs", type=int, default=40)
    p.add_argument("--max_frames", type=int, default=6000)  # SMC excerpts ~40s ~3400 frames
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--n_particles", type=int, default=300)
    p.add_argument("--obs_sigma", default="0.15,0.25")
    p.add_argument("--ess_frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wavebeat_root", default="extractors/wavebeat")
    cli = p.parse_args()

    torch.manual_seed(cli.seed)
    import random
    random.seed(cli.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fps = 22050 / 256
    sigmas = [float(s) for s in cli.obs_sigma.split(",") if s.strip()]

    smc_root = Path(cli.smc_root)
    audio_dir = smc_root / "SMC_MIREX_Audio"
    annot_dir = smc_root / "SMC_MIREX_Annotations"
    assert audio_dir.is_dir() and annot_dir.is_dir(), f"SMC dirs missing under {smc_root}"

    # ---- extractor backend (frontend the CHART ckpt was trained on) ----
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
            raise SystemExit("--extractor_ckpt is required for the wavebeat frontend")
        ext_args = argparse.Namespace(
            wavebeat_root=cli.wavebeat_root, extractor_ckpt=cli.extractor_ckpt,
        )
    extractor = backend.build_model(ext_args, device)
    backend.load_checkpoint(extractor, ext_args, device)
    extractor.eval()
    print(f"[SMC] frontend={cli.frontend}"
          + (f" ({cli.beat_this_checkpoint}, {cli.extractor_fps_mode})" if cli.frontend == "beat_this" else ""))

    # ---- SMC dataset (raw WaveBeat loader; full-val = all files, no crop) ----
    sys.path.insert(0, str(Path(cli.wavebeat_root).resolve()))
    from wavebeat.data import DownbeatDataset  # type: ignore
    ds = DownbeatDataset(
        audio_dir=str(audio_dir), annot_dir=str(annot_dir), dataset="smc",
        audio_sample_rate=22050, target_factor=256, subset="full-val",
        length=2097152, preload=False, augment=False, examples_per_epoch=1000,
        half=False, dry_run=False,
    )
    model = _build_model(cli.checkpoint, device)

    acc = Acc()
    n = 0
    with torch.no_grad():
        for idx in range(len(ds)):
            if n >= cli.max_songs:
                break
            audio, target, _meta = ds[idx]
            ref_beats = _load_ref_beats(ds.annot_files[idx])
            if len(ref_beats) < 2:
                continue
            audio = audio.float().unsqueeze(0).to(device)      # [1, 1, samples]
            tgt = target.float().unsqueeze(0).to(device)
            # Crop the AUDIO to the eval window before the extractor. Beat This is a
            # transformer (cost grows with audio length), so feeding full ~80s SMC
            # excerpts is the bottleneck. We only score the first max_frames frames
            # anyway (ref beats are clipped to the window below), so this is exact.
            max_samples = cli.max_frames * 256
            if audio.shape[-1] > max_samples:
                audio = audio[..., :max_samples]
                if tgt.shape[-1] > cli.max_frames:
                    tgt = tgt[..., :cli.max_frames]
            _, activations = backend.compute_loss_and_activations(
                model=extractor, audio=audio, target=tgt, frozen=True,
            )
            Tc = min(activations.shape[1], cli.max_frames)
            activations = activations[:, :Tc, :]
            dur = Tc / fps
            # clip refs to the evaluated window
            ref = ref_beats[ref_beats < dur]
            if len(ref) < 2:
                continue

            # Raw WaveBeat baseline: peak-pick the frontend's beat-activation channel
            # directly (no structured tracker). Isolates what CHART/PF ADDS over its
            # own discriminative frontend on this out-of-distribution set.
            rawwb = activations[0, :, 0].cpu().numpy()
            acc.add("rawwb_", evaluate_beats(ref, extract_beat_timestamps(rawwb, fps=fps)))

            _readout(model.sample_from_prior(activations, temperature=cli.temperature),
                     ref, fps, acc, "openloop.")
            for s in sigmas:
                out = model.sample_from_prior_pf(
                    activations, n_particles=cli.n_particles, obs_sigma=s,
                    temperature=cli.temperature, ess_frac=cli.ess_frac,
                )
                _readout(out, ref, fps, acc, f"pf{s}.")
            acc.add("base_", evaluate_beats(ref, _const_baseline(dur)))
            acc.add("oracle_", evaluate_beats(ref, _tempo_oracle_baseline(ref, dur)))
            n += 1
            if n % 5 == 0:
                print(f"  ...scored {n} songs")

    n = max(n, 1)
    print(f"\n[SMC] scored {n} held-out SMC songs (beats only), "
          f"N={cli.n_particles} ess_frac={cli.ess_frac}\n")
    print("  {:<16s} {:>9s} {:>7s} {:>7s}".format("METHOD/readout", "F", "CMLt", "AMLt"))

    def row(label, prefix):
        print("  {:<16s} {:>9.3f} {:>7.3f} {:>7.3f}".format(
            label, acc.get(prefix + "F-measure"), acc.get(prefix + "CMLt"),
            acc.get(prefix + "AMLt")))

    row("rawWaveBeat", "rawwb_")
    row("openloop phase", "openloop.phase_")
    row("openloop dec", "openloop.dec_")
    for s in sigmas:
        row(f"PF s={s} phase", f"pf{s}.phase_")
        row(f"PF s={s} dec", f"pf{s}.dec_")
        row(f"PF s={s} wrap", f"pf{s}.wrap_")
    row("baseline120", "base_")
    row("tempoOracle", "oracle_")

    rawwb = acc.get("rawwb_CMLt")
    ol = max(acc.get("openloop.phase_CMLt"), acc.get("openloop.dec_CMLt"))
    pf = max(
        max(acc.get(f"pf{s}.phase_CMLt"), acc.get(f"pf{s}.dec_CMLt"), acc.get(f"pf{s}.wrap_CMLt"))
        for s in sigmas
    )
    rawwb_f = acc.get("rawwb_F-measure")
    pf_f = max(max(acc.get(f"pf{s}.phase_F-measure"), acc.get(f"pf{s}.dec_F-measure")) for s in sigmas)
    print(f"\n[SMC] CMLt: rawWaveBeat={rawwb:.3f}  open-loop={ol:.3f}  particle-filter={pf:.3f}")
    print(f"[SMC] F:    rawWaveBeat={rawwb_f:.3f}  particle-filter={pf_f:.3f}  "
          f"(does the structured tracker beat its own frontend? ΔCMLt={pf - rawwb:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
