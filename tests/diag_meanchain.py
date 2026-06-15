"""Diagnostic: does the trained model's deterministic MEAN phase trajectory
advance (clean sawtooth) or is it frozen (tempo collapse)? Inspects tempo and
per-frame phase advance on real held-out songs."""
from __future__ import annotations
import argparse, sys, math
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.svt_core import SVTModel, TWO_PI
from training.extractors import get_extractor_backend
from evaluation.score import evaluate_beats, frames_to_beat_times
from evaluation.phase_converter import extract_beats_from_phase_trajectory

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--extractor_ckpt", required=True)
p.add_argument("--dataset_root", required=True)
p.add_argument("--n", type=int, default=3)
cli = p.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse as _a
args = _a.Namespace(wavebeat_root="extractors/wavebeat", dataset_root=cli.dataset_root,
    dataset_include="ballroom", phases_dir=None, audio_dir=None, annot_dir=None,
    wavebeat_dataset="ballroom", audio_sample_rate=22050, target_factor=256,
    train_length=2097152, num_workers=2, examples_per_epoch=1000, preload=False,
    augment=False, dry_run=False, batch_size=1, extractor_ckpt=cli.extractor_ckpt,
    dist_rank=0, dist_world_size=1)
backend = get_extractor_backend("wavebeat")
vl = backend.build_val_dataloader(args)
ext = backend.build_model(args, dev); backend.load_checkpoint(ext, args, dev); ext.eval()

ck = torch.load(cli.checkpoint, map_location=dev, weights_only=False)
saved = ck.get("args", {})
m = SVTModel(hidden_dim=128, nhead=4, num_layers=2,
    num_meter_classes=saved.get("num_meter_classes", 8),
    phase_corr_scale=saved.get("phase_corr_scale", math.pi),
    tempo_corr_scale=saved.get("tempo_corr_scale", 1.0),
    decoder_use_h_prior=not saved.get("decoder_latent_only", False),
    posterior_phase_recursive=saved.get("posterior_phase_recursive", False),
    tempo_anchor_mode=saved.get("tempo_anchor_mode", "none"),
    tempo_reversion_alpha=saved.get("tempo_reversion_alpha", 0.0),
    tempo_anchor_ema_beta=saved.get("tempo_anchor_ema_beta", 0.02)).to(dev)
m.load_state_dict(ck["svt_model"] if "svt_model" in ck else ck, strict=True); m.eval()
fps = 22050/256
print(f"INIT_LOG_TEMPO ref = {math.log(120/60*TWO_PI/fps):.3f} (=120BPM, ~{TWO_PI/(120/60*TWO_PI/fps):.0f} frames/beat)")

n = 0
with torch.no_grad():
    for batch in vl:
        if n >= cli.n: break
        audio = batch["audio"].to(dev); tgt = batch["extractor_target"].to(dev)
        _, act = backend.compute_loss_and_activations(model=ext, audio=audio, target=tgt, frozen=True)
        act = act[:, :2048]
        # ground-truth beats for scoring
        bt = batch["beat_targets"][0].cpu().numpy()
        T = act.shape[1]
        s = (len(bt) - T) // 2
        if s > 0: bt = bt[s:s+T]
        bt = bt[:T]
        ref = frames_to_beat_times(bt, fps)

        out = m.sample_from_prior(act, temperature=0.1)
        pm = out["phase_mu"][0].cpu().numpy()
        # Reconstruct mean phase under a BOUNDED tempo and a CONSTANT init tempo,
        # using the same per-frame phase corrections the model produced.
        h_prior = m.encode_prior(act)
        pcorr, tcorr = m.prior_mean_corrections(h_prior)
        pcorr = pcorr[0].cpu().numpy(); tcorr = tcorr[0].cpu().numpy()
        ltmu = float(out["log_tempo"][0, 0].cpu())  # model's own init log-tempo
        def build(mode):
            ph = pm[0]; lt = ltmu; traj = [ph]
            for t in range(1, T):
                if mode == "clamp":
                    lt = min(max(lt + tcorr[t], -3.5), -0.7)
                elif mode == "const120":
                    lt = math.log(120/60*TWO_PI/fps)  # fixed 120 BPM advance
                else:  # const-init
                    lt = ltmu
                ph = (ph + math.exp(min(lt,10)) + pcorr[t]) % TWO_PI
                traj.append(ph)
            return np.array(traj)
        if len(ref) >= 2:
            for tag, traj in [("raw-mean", pm), ("clamp[-3.5,-0.7]", build("clamp")),
                              ("const-init", build("ci")), ("const-120", build("const120"))]:
                est = extract_beats_from_phase_trajectory(traj, fps=fps)
                sc = evaluate_beats(ref, est)
                print(f"    {tag:18s} nbeats={len(est):3d}  F={sc['F-measure']:.3f} CMLt={sc['CMLt']:.3f} AMLt={sc['AMLt']:.3f}")
        ps = out["phase"][0].cpu().numpy()
        lt = out["log_tempo"][0].cpu().numpy()
        tempo_lin = np.exp(np.clip(lt, None, 10))
        # per-frame advance of the MEAN (unwrapped diff)
        adv = np.diff(np.unwrap(pm))
        wraps_mu = np.sum(np.diff(pm % TWO_PI) < -math.pi)
        wraps_s = np.sum(np.diff(ps % TWO_PI) < -math.pi)
        print(f"\nsong {n}: T={len(pm)}")
        print(f"  log_tempo: mean={lt.mean():.3f} start={lt[0]:.3f} end={lt[-1]:.3f} min={lt.min():.3f} max={lt.max():.3f}")
        print(f"  tempo_lin(rad/frame): mean={tempo_lin.mean():.5f} (=> {tempo_lin.mean()*fps/TWO_PI*60:.1f} BPM)  frames/beat~{TWO_PI/max(tempo_lin.mean(),1e-9):.0f}")
        print(f"  MEAN phase advance/frame: mean={adv.mean():.5f} std={adv.std():.5f}  total={adv.sum():.1f} rad (={adv.sum()/TWO_PI:.1f} cycles)")
        print(f"  wraps: mean-traj={wraps_mu}  stochastic={wraps_s}")
        n += 1
