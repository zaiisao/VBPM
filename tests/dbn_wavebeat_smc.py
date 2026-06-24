"""Control experiment: the bar-pointer DBN on WAVEBEAT activations over SMC.

WaveBeat (unlike Beat This) was NOT optimized for peak-picking -- its authors report
peak-picking fails and they rely on a DBN. So if Beat This's lambda->0 collapse is because
its activations are peak-pick-optimized (prior redundant), WaveBeat should be the OPPOSITE:
the DBN should BEAT peak-pick, and the best lambda should be a useful NON-zero value.

Runs frozen WaveBeat on SMC, then peak-pick vs a transition-lambda sweep (incl ~0) +
per-song oracle lambda. The key comparison: best-uniform-lambda DBN vs peak-pick.

    python tests/dbn_wavebeat_smc.py
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
import mir_eval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.extractors import get_extractor_backend
from models.bar_pointer_dbn import BarPointerDBN

SMC = "/home/sogang/jaehoon/Analyze-SMC/SMC_MIREX"
FPS = 22050 / 256
LAMBDAS = [0.05, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def _peakpick(prob, width):
    t = torch.from_numpy(np.ascontiguousarray(prob)).float().unsqueeze(0)
    peaks = t.masked_fill(t != F.max_pool1d(t, width, 1, width // 2), -1000.0)
    fr = torch.nonzero(peaks.squeeze(0) > 0.5).numpy()[:, 0]
    if len(fr):
        keep = [fr[0]]
        for x in fr[1:]:
            if x - keep[-1] > 1:
                keep.append(x)
        fr = np.array(keep)
    return fr / FPS


def _load_ref(p):
    a = np.loadtxt(p)
    return (a if a.ndim == 1 else a[:, 0]).astype(np.float64)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max_songs", type=int, default=217)
    p.add_argument("--max_frames", type=int, default=6000)
    p.add_argument("--extractor_ckpt", default="wavebeat_epoch=98-step=24749.ckpt")
    cli = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backend = get_extractor_backend("wavebeat")
    ext_args = argparse.Namespace(wavebeat_root="extractors/wavebeat", extractor_ckpt=cli.extractor_ckpt)
    extractor = backend.build_model(ext_args, dev)
    backend.load_checkpoint(extractor, ext_args, dev)
    extractor.eval()

    sys.path.insert(0, "extractors/wavebeat")
    from wavebeat.data import DownbeatDataset  # type: ignore
    ds = DownbeatDataset(
        audio_dir=SMC + "/SMC_MIREX_Audio", annot_dir=SMC + "/SMC_MIREX_Annotations",
        dataset="smc", audio_sample_rate=22050, target_factor=256, subset="full-val",
        length=2097152, preload=False, augment=False, examples_per_epoch=1000, half=False, dry_run=False)

    dbn = BarPointerDBN(fps=FPS, beats_only=True, num_intervals=48, learnable_lambda=False).to(dev)
    width = max(3, 2 * round(FPS * 0.07) + 1)   # odd kernel so max_pool keeps length
    elps = {l: dbn._edge_logp(log_lambda=torch.tensor([float(np.log(l))], device=dev)) for l in LAMBDAS}
    print(f"[wb-smc] WaveBeat @ {FPS:.1f}fps, DBN {dbn.num_states} states, peakpick width {width}", flush=True)

    pk, per_lambda, oracle, best_lam = [], {l: [] for l in LAMBDAS}, [], []
    n = 0
    with torch.no_grad():
        for idx in range(len(ds)):
            if n >= cli.max_songs:
                break
            audio, target, _ = ds[idx]
            ref = _load_ref(ds.annot_files[idx])
            if len(ref) < 2:
                continue
            audio = audio.float().unsqueeze(0).to(dev); tgt = target.float().unsqueeze(0).to(dev)
            ms = cli.max_frames * 256
            if audio.shape[-1] > ms:
                audio = audio[..., :ms]; tgt = tgt[..., :min(tgt.shape[-1], cli.max_frames)]
            _, act = backend.compute_loss_and_activations(model=extractor, audio=audio, target=tgt, frozen=True)
            Tc = min(act.shape[1], cli.max_frames); act = act[0, :Tc].float()
            dur = Tc / FPS; ref = ref[ref < dur]
            if len(ref) < 2:
                continue
            n += 1
            beat = act[:, 0]
            pk.append(mir_eval.beat.evaluate(ref, _peakpick(beat.cpu().numpy(), width))["F-measure"])
            obs = dbn.observation_logp(act)
            bestF, bestL = -1.0, None
            for l in LAMBDAS:
                path = dbn._viterbi(obs, elp=elps[l]); bfr, _ = dbn._path_to_beats(path, beat_snap=beat)
                f = mir_eval.beat.evaluate(ref, bfr.cpu().numpy().astype(float) / FPS)["F-measure"]
                per_lambda[l].append(f)
                if f > bestF:
                    bestF, bestL = f, l
            oracle.append(bestF); best_lam.append(bestL)
            if n % 10 == 0:
                print(f"  ...{n} tracks ({__import__('datetime').datetime.now().strftime('%H:%M:%S')})", flush=True)

    print(f"\n[wb-smc] {len(pk)} SMC tracks")
    print(f"  peak-pick               = {np.mean(pk):.4f}   (WaveBeat: expected POOR)")
    for l in LAMBDAS:
        print(f"  DBN lambda={l:<6g}        = {np.mean(per_lambda[l]):.4f}")
    bl = max(LAMBDAS, key=lambda l: np.mean(per_lambda[l]))
    print(f"\n  BEST-UNIFORM lambda ({bl:g})  = {np.mean(per_lambda[bl]):.4f}")
    print(f"  PER-SONG ORACLE         = {np.mean(oracle):.4f}")
    vals, cnts = np.unique(best_lam, return_counts=True)
    print(f"  per-song best-lambda dist: " + ", ".join(f"{v:g}:{c}" for v, c in zip(vals, cnts)))
    print(f"\n  >>> DOES THE DBN HELP WAVEBEAT? best-uniform {np.mean(per_lambda[bl]):.3f} vs peak-pick {np.mean(pk):.3f} "
          f"({'DBN HELPS' if np.mean(per_lambda[bl]) > np.mean(pk) else 'DBN hurts'})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
