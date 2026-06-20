"""Deep free-run diagnosis: load a (collapsed) checkpoint and, on each held-out
song, run BOTH the teacher-forced posterior rollout (forward) and the deployed
free-run prior rollout (sample_from_prior) on the SAME activations, then dissect
WHERE the free-run fails. Answers four questions per song and in aggregate:

  Q1 READOUT vs DYNAMICS:
     phase_F  (beats read off phase WRAPS) and dec_F (beats off the DECODER),
     for teacher-forced (TF) and free-run (FR). If TF is high and FR is low,
     the free-run *trajectory* (dynamics) is broken, not the decoder/readout.
  Q2 TEMPO: free-run BPM(t) vs teacher-forced BPM(t) vs GT BPM. Does free-run
     tempo lock or random-walk away? (CMLt~0 says it drifts.)
  Q3 PHASE ADVANCE: # of free-run phase wraps vs GT beat count; effective
     free-run period vs true period.
  Q4 DIVERGENCE: first frame where circular |phase_FR - phase_TF| > pi/2, and
     mean circular error in 1st vs 2nd half of the song.
  Q5 AUDIO CORRECTION: magnitude of the prior-mean nudges g^phi(h), g^tau(h)
     relative to the per-frame increment they are supposed to correct.
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.svt_core import SVTModel, TWO_PI
from evaluation.phase_converter import (
    extract_beat_timestamps,
    extract_beats_from_phase_trajectory,
)
from evaluation.score import evaluate_beats, frames_to_beat_times


def _wrap_pi(x):
    return (x + np.pi) % TWO_PI - np.pi


def build_from_ckpt(path, device):
    ck = torch.load(path, map_location=device)
    a = ck["args"]
    m = SVTModel(
        hidden_dim=128, nhead=4, num_layers=2,
        num_meter_classes=a["num_meter_classes"],
        phase_corr_scale=a["phase_corr_scale"],
        tempo_corr_scale=a["tempo_corr_scale"],
        decoder_use_h_prior=not a["decoder_latent_only"],
        tempo_anchor_mode=a["tempo_anchor_mode"],
        tempo_reversion_alpha=a["tempo_reversion_alpha"],
        audio_emission=a["audio_emission"],
        bar_phase=a["bar_phase"],
        meter_ste=a["meter_ste"],
        delta_vae=a.get("delta_vae", False),
        delta_vae_rate=a.get("delta_vae_rate", 0.1),
        dvbf=a.get("dvbf", False),
    ).to(device)
    m.load_state_dict(ck["svt_model"])
    m.eval()
    return m, a


def beatF(ref, est):
    if len(ref) < 2 or len(est) < 1:
        return 0.0
    return evaluate_beats(ref, est)["F-measure"]


def main() -> int:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "cache/diag/ckpt_faith_BZ.pt"
    cache = sys.argv[2] if len(sys.argv) > 2 else "cache/acts/bt_val"
    nsongs = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    maxT = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, a = build_from_ckpt(ckpt, device)
    print(f"[diag] ckpt={ckpt}  phase_corr_scale={a['phase_corr_scale']} "
          f"tempo_corr_scale={a['tempo_corr_scale']} bar_phase={a['bar_phase']}")

    fs = sorted(glob.glob(str(Path(cache) / "**" / "*.pt"), recursive=True))[:nsongs]
    agg = {k: [] for k in ("phaseF_tf", "phaseF_fr", "decF_tf", "decF_fr",
                            "bpm_gt", "bpm_fr_mean", "bpm_fr_std", "bpm_tf_std",
                            "fr_wraps", "gt_beats", "fr_period", "gt_period",
                            "div_frame", "cerr_1st", "cerr_2nd",
                            "phcorr_abs", "tcorr_abs", "incr_mean")}

    for f in fs:
        r = torch.load(f, map_location=device)
        act = r["activations"][:maxT].unsqueeze(0).to(device)
        T = act.shape[1]
        fps = float(r["fps"])
        bt = r["beat_targets"][:maxT].cpu().numpy()
        ref = frames_to_beat_times(bt, fps)
        if len(ref) < 3:
            continue
        gt_idx = np.where(bt > 0.5)[0]
        gt_period = float(np.median(np.diff(gt_idx))) if len(gt_idx) > 2 else float("nan")
        bpm_gt = fps * 60.0 / gt_period if gt_period > 0 else float("nan")

        with torch.no_grad():
            out_tf = model(act)                              # teacher-forced (posterior)
            out_fr = model.sample_from_prior(act, temperature=0.1)  # free-run (prior)
            # audio-driven prior-mean corrections on this song's h
            h_prior = model.encode_prior(act)
            ph_corr, t_corr = model.prior_mean_corrections(h_prior)

        ph_tf = out_tf["posterior"]["phase_mu"][0].cpu().numpy()
        ph_fr = out_fr["phase_mu"][0].cpu().numpy()
        lt_tf = out_tf["samples"]["log_tempo"][0].cpu().numpy()
        lt_fr = out_fr["log_tempo"][0].cpu().numpy()
        dec_tf = torch.sigmoid(out_tf["beat_logits"][0, :, 0]).cpu().numpy()
        dec_fr = torch.sigmoid(out_fr["beat_logits"][0, :, 0]).cpu().numpy()

        # Q1 readout vs dynamics
        agg["phaseF_tf"].append(beatF(ref, extract_beats_from_phase_trajectory(ph_tf, fps=fps)))
        agg["phaseF_fr"].append(beatF(ref, extract_beats_from_phase_trajectory(ph_fr, fps=fps)))
        agg["decF_tf"].append(beatF(ref, extract_beat_timestamps(dec_tf, fps=fps)))
        agg["decF_fr"].append(beatF(ref, extract_beat_timestamps(dec_fr, fps=fps)))

        # Q2 tempo (rad/frame -> BPM)
        bpm_fr = fps * 60.0 * np.exp(lt_fr) / TWO_PI
        bpm_tf = fps * 60.0 * np.exp(lt_tf) / TWO_PI
        agg["bpm_gt"].append(bpm_gt)
        agg["bpm_fr_mean"].append(float(np.mean(bpm_fr)))
        agg["bpm_fr_std"].append(float(np.std(bpm_fr)))
        agg["bpm_tf_std"].append(float(np.std(bpm_tf)))

        # Q3 phase advance: count wraps (positive crossings) of free-run phase
        incr_fr = _wrap_pi(np.diff(ph_fr))
        fr_wraps = float(np.sum(np.clip(incr_fr, 0, None)) / TWO_PI)
        agg["fr_wraps"].append(fr_wraps)
        agg["gt_beats"].append(float(len(gt_idx)))
        mean_incr = float(np.mean(incr_fr[incr_fr > 0])) if np.any(incr_fr > 0) else float("nan")
        agg["incr_mean"].append(mean_incr)
        agg["fr_period"].append(TWO_PI / mean_incr if mean_incr and mean_incr > 0 else float("nan"))
        agg["gt_period"].append(gt_period)

        # Q4 divergence FR vs TF phase
        n = min(len(ph_tf), len(ph_fr))
        cerr = np.abs(_wrap_pi(ph_fr[:n] - ph_tf[:n]))
        over = np.where(cerr > np.pi / 2)[0]
        agg["div_frame"].append(float(over[0]) if len(over) else float(n))
        half = n // 2
        agg["cerr_1st"].append(float(np.mean(cerr[:half])))
        agg["cerr_2nd"].append(float(np.mean(cerr[half:])))

        # Q5 correction magnitude vs required increment
        agg["phcorr_abs"].append(float(ph_corr.abs().mean().cpu()))
        agg["tcorr_abs"].append(float(t_corr.abs().mean().cpu()))

    def m(k):
        v = [x for x in agg[k] if not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(v)) if v else float("nan")

    print(f"\n[diag] aggregated over {len(agg['decF_fr'])} songs (maxT={maxT})\n")
    print("Q1 READOUT vs DYNAMICS  (beat F-measure)")
    print(f"   phase-wrap readout :  TF={m('phaseF_tf'):.3f}   FR={m('phaseF_fr'):.3f}")
    print(f"   decoder readout    :  TF={m('decF_tf'):.3f}   FR={m('decF_fr'):.3f}")
    print(f"   -> TF high & FR low => free-run DYNAMICS broken (not the decoder).")
    print(f"   -> phaseF_FR vs decF_FR tells if FR phase is right but readout fails.\n")
    print("Q2 TEMPO  (BPM)")
    print(f"   GT mean BPM        :  {m('bpm_gt'):.1f}")
    print(f"   FR  mean BPM       :  {m('bpm_fr_mean'):.1f}   (per-song std over time={m('bpm_fr_std'):.2f})")
    print(f"   TF  BPM std/time   :  {m('bpm_tf_std'):.2f}   (low std = locked)")
    print(f"   -> large FR BPM std or FR!=GT => tempo random-walk / wrong octave.\n")
    print("Q3 PHASE ADVANCE")
    print(f"   FR wraps           :  {m('fr_wraps'):.1f}   vs GT beats {m('gt_beats'):.1f}")
    print(f"   FR period (frames) :  {m('fr_period'):.1f}   vs GT period {m('gt_period'):.1f}")
    print(f"   -> wraps<<beats or period>>GT => free-run advances too slowly.\n")
    print("Q4 DIVERGENCE (FR vs TF phase)")
    print(f"   first |dphase|>pi/2:  frame {m('div_frame'):.0f}")
    print(f"   circ err 1st half  :  {m('cerr_1st'):.3f} rad   2nd half {m('cerr_2nd'):.3f} rad")
    print(f"   -> early divergence => exposure bias compounds fast.\n")
    print("Q5 AUDIO CORRECTION (the only audio coupling at inference)")
    print(f"   |g_phase(h)| mean  :  {m('phcorr_abs'):.4f} rad   (cap=phase_corr_scale={a['phase_corr_scale']})")
    print(f"   |g_tempo(h)| mean  :  {m('tcorr_abs'):.4f}        (cap=tempo_corr_scale={a['tempo_corr_scale']})")
    print(f"   FR mean increment  :  {m('incr_mean'):.4f} rad/frame")
    print(f"   -> if |g_phase| << per-beat error, audio can't re-anchor a drifting prior.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
