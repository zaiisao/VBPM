"""Extract HELD-OUT rich (512-dim) Beat-This features for SMC.

For each SMC track, load the held-out fold checkpoint (fold0..7, the one that never
trained on it), run Beat This, and tap the 512-dim penultimate (transformer_blocks
output, the input to task_heads) at native 50 fps. Saves [T,512] features + the
derived [T,2] activation + the fold. Validates the derived beat activation against
Analyze-SMC's cached held-out [T,2] (should correlate ~1.0) to confirm fidelity.

    python tests/extract_smc_rich.py
"""
from __future__ import annotations
import os, sys
from collections import defaultdict
import numpy as np, torch, torchaudio
from scipy.special import expit

BT = "/home/sogang/jaehoon/Analyze-SMC/beat_this"
sys.path.insert(0, BT)
import inspect
from beat_this.model.beat_tracker import BeatThis           # soxr-free
from beat_this.utils import replace_state_dict_key
from beat_this.preprocessing import LogMelSpect             # soxr is broken in this env; use torchaudio

# Replicate beat_this.inference.load_model WITHOUT importing inference.py (it imports soxr).
_CKPT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp"

def load_model(name, device):
    ckpt = torch.hub.load_state_dict_from_url(
        f"{_CKPT_URL}/{name}.ckpt", file_name=f"beat_this-{name}.ckpt",
        map_location=device, check_hash=False)
    hp = {k: v for k, v in ckpt["hyper_parameters"].items()
          if k in set(inspect.signature(BeatThis).parameters)}
    m = BeatThis(**hp)
    m.load_state_dict(replace_state_dict_key(ckpt["state_dict"], "model.", ""))
    return m.to(device).eval()

SMC = "/home/sogang/jaehoon/Analyze-SMC"
AUDIO = SMC + "/SMC_MIREX/SMC_MIREX_Audio"
SPLIT = SMC + "/beat_this_annotations/smc/8-folds.split"
CACHE = SMC + "/beat_this_activations_cache"
OUT = "/home/sogang/jaehoon/CHART/cache/acts/smc_rich_heldout"


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spect_proc = LogMelSpect(device=dev)

    split = {}
    for line in open(SPLIT):
        ps = line.split()
        if len(ps) == 2:
            split[ps[0]] = int(ps[1])
    byfold = defaultdict(list)
    for t, fo in split.items():
        byfold[fo].append(t)

    n, corrs = 0, []
    for fo in sorted(byfold):
        model = load_model(f"fold{fo}", dev)
        core = getattr(model, "_orig_mod", model)
        first = True
        for tid in sorted(byfold[fo]):
            wav = f"{AUDIO}/{tid.upper()}.wav"
            if not os.path.exists(wav):
                print(f"  MISSING {wav}"); continue
            wav_t, sr = torchaudio.load(wav)                         # [C, N]
            sig = wav_t.mean(0)                                       # mono
            if sr != 22050:
                sig = torchaudio.functional.resample(sig, sr, 22050)
            spect = spect_proc(sig.to(dev))                          # [T, 128]
            with torch.inference_mode():
                feat = core.transformer_blocks(core.frontend(spect.unsqueeze(0)))   # [1, T, 512]
                out = core.task_heads(feat)
            feat = feat[0].float().cpu()                              # [T, 512]
            beat = torch.sigmoid(out["beat"][0]).float().cpu()
            down = torch.sigmoid(out["downbeat"][0]).float().cpu()
            act2 = torch.stack([beat, down], dim=-1)                  # [T, 2]
            torch.save({"feat": feat.half(), "act2": act2, "tid": tid, "fold": fo},
                       f"{OUT}/{tid}.pt")
            n += 1
            # fidelity check vs cached held-out [T,2]
            cpath = f"{CACHE}/{tid}.npz"
            if os.path.exists(cpath):
                cb = expit(np.load(cpath)["beat"].astype(np.float64))
                m = min(len(cb), len(beat))
                c = np.corrcoef(cb[:m], beat.numpy()[:m].astype(np.float64))[0, 1]
                corrs.append(c)
                if first:
                    print(f"  fold{fo} {tid}: T={feat.shape[0]} dim={feat.shape[1]} "
                          f"| derived-beat vs cached corr={c:.4f}", flush=True)
                    first = False
        print(f"[fold {fo}] done ({len(byfold[fo])} tracks)", flush=True)
    print(f"\n[extract] wrote {n} SMC rich-feature files -> {OUT}")
    print(f"[extract] derived-vs-cached beat corr: mean={np.mean(corrs):.4f} min={np.min(corrs):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
