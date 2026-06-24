"""JOINT eval + SNR sweep: does the structured CHART model beat a discriminative peak-pick
on the outputs it was built for -- DOWNBEATS and TEMPO -- and does that change as the
frontend DEGRADES?

The clean-data verdict was decisive (peak-pick wins beats/downbeats/tempo). The one
unsettled question is the LOW-SNR regime the architecture was theorised to own: as the
frontend's evidence degrades, does CHART's structural prior start closing the gap (or
winning)? We test it by adding gaussian noise (x per-feature std) to BOTH the model's rich
input AND the peak-pick's act2, and sweeping the level.

    python tests/eval_joint.py --ckpt cache/diag/rich_pure_best.pt --cache cache/acts/bt_val_rich \
        --noise_levels 0,0.5,1,2,4 --max_songs 60 --max_frames 1024
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.svt_core import SVTModel, TWO_PI
from evaluation.phase_converter import extract_beat_timestamps, extract_beats_from_phase_trajectory
from evaluation.score import evaluate_beats, evaluate_downbeats, frames_to_beat_times


def _build(margs, device):
    return SVTModel(
        input_dim=margs["input_dim"], hidden_dim=128, nhead=4, num_layers=2,
        num_meter_classes=margs["num_meter_classes"],
        phase_corr_scale=margs["phase_corr_scale"], tempo_corr_scale=margs["tempo_corr_scale"],
        decoder_use_h_prior=not margs["decoder_latent_only"],
        tempo_anchor_mode=margs["tempo_anchor_mode"], tempo_reversion_alpha=margs["tempo_reversion_alpha"],
        audio_emission=margs["audio_emission"], bar_phase=margs["bar_phase"],
        meter_ste=margs["meter_ste"], delta_vae=margs["delta_vae"],
        delta_vae_rate=margs["delta_vae_rate"], dvbf=margs["dvbf"],
    ).to(device)


def _tempo(times):
    if len(times) < 2:
        return 0.0
    ibi = np.diff(np.sort(times)); ibi = ibi[ibi > 1e-3]
    return float(60.0 / np.median(ibi)) if len(ibi) else 0.0


def _acc1(est, ref, tol=0.04):
    return ref > 0 and abs(est - ref) <= tol * ref


def _acc2(est, ref, tol=0.04):
    if ref <= 0 or est <= 0:
        return False
    return any(abs(est - m * ref) <= tol * m * ref for m in (1.0, 2.0, 3.0, 0.5, 1.0 / 3.0))


def eval_at_noise(model, songs, noise, temperature, dev):
    """Run the full peak-pick-vs-CHART comparison at one frontend-degradation level."""
    A = {k: [] for k in ("pk_beat", "ch_beat", "pk_db", "ch_db")}
    Tm = {k: {"a1": [], "a2": []} for k in ("pk", "ch_lat")}
    with torch.no_grad():
        for s in songs:
            fps = s["fps"]
            act = s["act"].clone()                       # [1,T,512] on device
            act2 = s["act2"].copy()                       # [T,2] np
            if noise > 0:
                act = act + noise * act.std() * torch.randn_like(act)
                act2 = np.clip(act2 + noise * act2.std() * np.random.randn(*act2.shape), 0.0, 1.0)
            ref, ref_db, gt_tempo = s["ref"], s["ref_db"], s["gt_tempo"]

            # ---- baseline: peak-pick the (degraded) frontend ----
            pk_beats = extract_beat_timestamps(act2[:, 0], fps=fps)
            pk_db = extract_beat_timestamps(act2[:, 1], fps=fps)
            A["pk_beat"].append(evaluate_beats(ref, pk_beats)["F-measure"])

            # ---- CHART: deployed free-run on (degraded) rich input ----
            out = model.sample_from_prior(act, temperature=temperature)
            phase = out.get("phase_mu", out["phase"])[0].cpu().numpy()
            bprob = torch.sigmoid(out["beat_logits"][0, :, 0]).cpu().numpy()
            dbprob = torch.sigmoid(out["beat_logits"][0, :, 1]).cpu().numpy()
            ch_dec = extract_beat_timestamps(bprob, fps=fps)
            ch_ph = extract_beats_from_phase_trajectory(phase, fps=fps)
            A["ch_beat"].append(max(evaluate_beats(ref, ch_dec)["F-measure"],
                                    evaluate_beats(ref, ch_ph)["F-measure"]))

            if len(ref_db) >= 2:
                A["pk_db"].append(evaluate_downbeats(ref_db, pk_db)["db_F-measure"])
                ch_db_dec = extract_beat_timestamps(dbprob, fps=fps)
                bar_f = 0.0
                if out.get("bar_phase_mu") is not None:
                    bar_db = extract_beats_from_phase_trajectory(out["bar_phase_mu"][0].cpu().numpy(), fps=fps)
                    bar_f = evaluate_downbeats(ref_db, bar_db)["db_F-measure"]
                A["ch_db"].append(max(evaluate_downbeats(ref_db, ch_db_dec)["db_F-measure"], bar_f))

            # ---- tempo ----
            lt = out["log_tempo"][0].cpu().numpy()
            lat_tempo = float(60.0 * fps * np.median(np.exp(np.clip(lt, -10, 10))) / TWO_PI)
            Tm["pk"]["a1"].append(1.0 if _acc1(_tempo(pk_beats), gt_tempo) else 0.0)
            Tm["pk"]["a2"].append(1.0 if _acc2(_tempo(pk_beats), gt_tempo) else 0.0)
            Tm["ch_lat"]["a1"].append(1.0 if _acc1(lat_tempo, gt_tempo) else 0.0)
            Tm["ch_lat"]["a2"].append(1.0 if _acc2(lat_tempo, gt_tempo) else 0.0)
    m = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {
        "pk_beat": m(A["pk_beat"]), "ch_beat": m(A["ch_beat"]),
        "pk_db": m(A["pk_db"]), "ch_db": m(A["ch_db"]),
        "pk_tempo": m(Tm["pk"]["a2"]), "ch_tempo": m(Tm["ch_lat"]["a2"]),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="cache/diag/rich_pure_best.pt")
    p.add_argument("--cache", default="cache/acts/bt_val_rich")
    p.add_argument("--device", default="cuda")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max_frames", type=int, default=1024)
    p.add_argument("--max_songs", type=int, default=60)
    p.add_argument("--noise_levels", default="0,0.5,1,2,4")
    p.add_argument("--seed", type=int, default=0)
    cli = p.parse_args()
    dev = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cli.seed); np.random.seed(cli.seed)

    ck = torch.load(cli.ckpt, map_location="cpu")
    model = _build(ck["args"], dev); model.load_state_dict(ck["svt_model"]); model.eval()

    files = sorted(glob.glob(cli.cache + "/*.pt"))[:cli.max_songs]
    songs = []
    for f in files:
        r = torch.load(f, map_location="cpu")
        fps = float(r["fps"])
        bt = r["beat_targets"][:cli.max_frames].numpy()
        db = r["downbeat_targets"][:cli.max_frames].numpy()
        ref = frames_to_beat_times(bt, fps)
        if len(ref) < 2:
            continue
        songs.append({
            "act": r["activations"][:cli.max_frames].unsqueeze(0).to(dev).float(),
            "act2": r["act2"][:cli.max_frames].float().numpy(),
            "fps": fps, "ref": ref, "ref_db": frames_to_beat_times(db, fps),
            "gt_tempo": _tempo(ref),
        })
    levels = [float(x) for x in cli.noise_levels.split(",")]
    print(f"[sweep] {len(songs)} songs | ckpt={Path(cli.ckpt).name} | levels={levels}", flush=True)
    print(f"\n  noise |  beat-F (pk / CHART) | downbeat-F (pk / CHART) | tempo-Acc2 (pk / CHART)")
    print(f"  ------+---------------------+-------------------------+----------------------")
    rows = []
    for nz in levels:
        rr = eval_at_noise(model, songs, nz, cli.temperature, dev)
        rows.append((nz, rr))
        print(f"  {nz:4.2f}  |   {rr['pk_beat']:.3f} / {rr['ch_beat']:.3f}    |"
              f"     {rr['pk_db']:.3f} / {rr['ch_db']:.3f}       |"
              f"    {rr['pk_tempo']:.3f} / {rr['ch_tempo']:.3f}", flush=True)
    # does the downbeat gap (pk - CHART) shrink as the frontend degrades?
    print("\n  noise | downbeat gap (pk - CHART)  [shrinking => structure helps under degradation]")
    for nz, rr in rows:
        gap = rr["pk_db"] - rr["ch_db"]
        print(f"  {nz:4.2f}  |  {gap:+.3f}   {'CHART AHEAD' if gap < 0 else ''}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
