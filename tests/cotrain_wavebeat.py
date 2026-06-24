"""END-TO-END co-training: unfreeze the feature extractor (WaveBeat) and train the WHOLE
model through the bar-pointer DBN, with the frontend LR set to 0.1x the head LR.

The whole thread says the limitation is the FROZEN frontend + frozen prior. This lets the
frontend co-adapt to lean on the structured prior (and lets a lambda-head pick per-song
lambda). Trained with the CRF/forward-likelihood loss; SMC leave-one-fold-out (frontend
fine-tuned on training folds, evaluated on the held-out fold -> leak-free). Compares to the
FROZEN-frontend + fixed-lambda DBN baseline.

    python tests/cotrain_wavebeat.py --steps 300            # full LOFO
    python tests/cotrain_wavebeat.py --steps 15 --smoke     # quick path/memory check
"""
from __future__ import annotations
import argparse, copy, os, sys
from pathlib import Path
import numpy as np, torch
from torch import nn
import torch.nn.functional as F
import mir_eval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.extractors import get_extractor_backend
from models.bar_pointer_dbn import BarPointerDBN

SMC = "/home/sogang/jaehoon/Analyze-SMC/SMC_MIREX"
SPLIT = "/home/sogang/jaehoon/Analyze-SMC/beat_this_annotations/smc/8-folds.split"
FPS, LAM = 22050 / 256, 16


def _madmom_logdens(beat):
    b = beat.clamp(1e-6, 1 - 1e-6)
    return torch.stack([((1 - b) / (LAM - 1)).clamp_min(1e-6).log(), b.log()], dim=-1)


def _dilate(x, w):
    return (F.conv1d(x.view(1, 1, -1), torch.ones(1, 1, 2 * w + 1, device=x.device), padding=w)[0, 0] > 0.5).float()


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


def crf_loss(dbn, em, ind, w, elp, C=6.0):
    obs = dbn.class_logp_to_states(em); logZ = dbn.forward_logpartition(obs, elp=elp)
    win = _dilate(ind, w) > 0.5; bs = (dbn.obs_ptr >= 1); T = obs.shape[0]
    allowed = torch.where(win[:, None], bs[None, :].expand(T, -1), (~bs)[None, :].expand(T, -1))
    obs_gt = obs + torch.where(allowed, obs.new_zeros(()), obs.new_full((), -C))
    return (logZ - dbn.forward_logpartition(obs_gt, elp=elp)) / T


class LambdaHead(nn.Module):
    """pooled [T,2] activation stats -> log lambda_trans (per-song). Init -> lambda=100."""
    def __init__(self, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hid), nn.ReLU(), nn.Linear(hid, 1))
        nn.init.zeros_(self.net[-1].weight); self.net[-1].bias.data.fill_(float(np.log(100.0)))
    def forward(self, act):
        return self.net(torch.cat([act.mean(0), act.std(0)]))   # [1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--base_lr", type=float, default=1e-3)
    p.add_argument("--frontend_mult", type=float, default=0.1)
    p.add_argument("--frames", type=int, default=1024)
    p.add_argument("--max_frames", type=int, default=5000)
    p.add_argument("--W", type=int, default=6)
    p.add_argument("--extractor_ckpt", default="wavebeat_epoch=98-step=24749.ckpt")
    p.add_argument("--smoke", action="store_true")
    cli = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backend = get_extractor_backend("wavebeat")
    ext_args = argparse.Namespace(wavebeat_root="extractors/wavebeat", extractor_ckpt=cli.extractor_ckpt)
    base_extractor = backend.build_model(ext_args, dev)
    backend.load_checkpoint(base_extractor, ext_args, dev)
    base_state = copy.deepcopy(base_extractor.state_dict())

    sys.path.insert(0, "extractors/wavebeat")
    from wavebeat.data import DownbeatDataset  # type: ignore
    ds = DownbeatDataset(audio_dir=SMC + "/SMC_MIREX_Audio", annot_dir=SMC + "/SMC_MIREX_Annotations",
                         dataset="smc", audio_sample_rate=22050, target_factor=256, subset="full-val",
                         length=2097152, preload=False, augment=False, examples_per_epoch=1000, half=False, dry_run=False)
    split = {}
    for line in open(SPLIT):
        a = line.split()
        if len(a) == 2:
            split[a[0]] = int(a[1])

    print("[cotrain] preloading SMC audio...", flush=True)
    tracks = []
    for idx in range(len(ds)):
        audio, target, _ = ds[idx]
        _st = Path(ds.annot_files[idx]).stem.lower().split("_")
        tid = f"{_st[0]}_{_st[1]}"                                # SMC_174_1_1_1_m -> smc_174
        if tid not in split:
            continue
        ref = np.loadtxt(ds.annot_files[idx]); ref = (ref if ref.ndim == 1 else ref[:, 0]).astype(np.float64)
        if len(ref) < 2:
            continue
        ms = cli.max_frames * 256
        a = audio.float()
        if a.shape[-1] > ms:
            a = a[..., :ms]
        tracks.append({"audio": a, "ref": ref, "fold": split[tid], "tid": tid})
    folds = sorted(set(t["fold"] for t in tracks))
    print(f"[cotrain] {len(tracks)} SMC tracks, {len(folds)} folds, frontend_lr={cli.base_lr*cli.frontend_mult:.1e}", flush=True)

    dbn = BarPointerDBN(fps=FPS, beats_only=True, num_intervals=48, learnable_lambda=False).to(dev)
    width = max(3, 2 * round(FPS * 0.07) + 1)
    Fm = lambda ref, fr: mir_eval.beat.evaluate(ref, fr.cpu().numpy().astype(float) / FPS)["F-measure"]

    @torch.no_grad()
    def activations(extractor, audio):
        a = audio.unsqueeze(0).to(dev)
        tgt = torch.zeros(1, 2, a.shape[-1] // 256 + 8, device=dev)   # >= logits len; cropped down internally
        _, act = backend.compute_loss_and_activations(model=extractor, audio=a, target=tgt, frozen=True)
        return act[0]                                            # [T,2]

    def eval_set(extractor, head, ev_tracks):
        F_dbn, F_pk, lams = [], [], []
        for t in ev_tracks:
            act = activations(extractor, t["audio"]); beat = act[:, 0]
            dur = act.shape[0] / FPS; ref = t["ref"][t["ref"] < dur]
            if len(ref) < 2:
                continue
            with torch.no_grad():
                log_lam = head(act)
                elp = dbn._edge_logp(log_lambda=log_lam)
                em = _madmom_logdens(beat)
                bfr, _ = dbn.decode_emission(em, snap_act=beat, elp=elp)
            F_dbn.append(Fm(ref, bfr)); F_pk.append(mir_eval.beat.evaluate(ref, _peakpick(beat.cpu().numpy(), width))["F-measure"])
            lams.append(float(log_lam.exp()))
        return float(np.mean(F_dbn)), float(np.mean(F_pk)), float(np.mean(lams))

    eval_folds = folds[:1] if cli.smoke else folds
    co, frozen_dbn, pk_all = [], [], []
    for fo in eval_folds:
        tr = [t for t in tracks if t["fold"] != fo]; te = [t for t in tracks if t["fold"] == fo]
        extractor = backend.build_model(ext_args, dev); extractor.load_state_dict(base_state); extractor.train()
        for prm in extractor.parameters():
            prm.requires_grad = True
        head = LambdaHead().to(dev)
        opt = torch.optim.AdamW([
            {"params": extractor.parameters(), "lr": cli.base_lr * cli.frontend_mult},
            {"params": head.parameters(), "lr": cli.base_lr}])

        # FROZEN baseline on this fold (before any co-training)
        extractor.eval()
        fb_dbn, fb_pk, fb_lam = eval_set(extractor, head, te); frozen_dbn.append(fb_dbn); pk_all.append(fb_pk)
        print(f"[cotrain] fold {fo} step   0 (frozen): co-DBN={fb_dbn:.4f} mean-lambda={fb_lam:.1f} (peak-pick {fb_pk:.3f})", flush=True)
        extractor.train()

        for step in range(1, cli.steps + 1):
            t = tr[step % len(tr)]
            a = t["audio"].unsqueeze(0).to(dev)
            tgt = torch.zeros(1, 2, a.shape[-1] // 256 + 8, device=dev)   # >= logits len; cropped down internally
            _, act = backend.compute_loss_and_activations(model=extractor, audio=a, target=tgt, frozen=False)
            act = act[0]; beat = act[:, 0]; T = act.shape[0]
            fr = np.round(t["ref"] * FPS).astype(int); fr = fr[(fr >= 0) & (fr < T)]
            ind = torch.zeros(T, device=dev); ind[fr] = 1
            # crop for memory
            s = max(0, (int(fr.mean()) - cli.frames // 2)) if len(fr) else 0
            e = min(s + cli.frames, T)
            log_lam = head(act)
            elp = dbn._edge_logp(log_lambda=log_lam)
            loss = crf_loss(dbn, _madmom_logdens(beat[s:e]), ind[s:e], cli.W, elp)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(extractor.parameters(), 5.0)
            opt.step()
            if step in (25, 50, 100, cli.steps):
                extractor.eval()
                td, _, tl = eval_set(extractor, head, te)
                print(f"[cotrain] fold {fo} step {step:3d}: co-DBN={td:.4f} mean-lambda={tl:.2f}  loss={loss.item():.3f}", flush=True)
                extractor.train()
        cd = td   # last trajectory point
        co.append(cd)

    print(f"\n[cotrain] LOFO: frozen-frontend DBN={np.mean(frozen_dbn):.4f}  "
          f"CO-TRAINED DBN={np.mean(co):.4f}  peak-pick={np.mean(pk_all):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
