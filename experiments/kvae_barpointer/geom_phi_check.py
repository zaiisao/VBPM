"""Does phi actually ROTATE, at a REALISTIC tempo? Load the saved diagram CVAE, deploy h-only, and for
each val song measure: net phi revolutions vs GT #bars (downbeats), effective tempo (BPM) from the
per-frame phi advance vs GT tempo, and whether phi advances monotonically (rotation) or oscillates.
A high beat-F is only meaningful if phi rotates ~#bars at a musical tempo.
"""
import sys, math, importlib.util
import numpy as np, torch

ROOT = "/home/sogang/jaehoon/CHART"; sys.path.insert(0, ROOT)
s = importlib.util.spec_from_file_location("da", f"{ROOT}/experiments/diagram_arch/run.py")
da = importlib.util.module_from_spec(s); s.loader.exec_module(da)
BPVAE, rollout, load_pool = da.BPVAE, da.rollout, da.load_pool
DEV = da.DEV; FPS = 86.1328125; TWO_PI = 2*math.pi; M = 4

d = torch.load("checkpoints/diagram_rerun.pt", map_location=DEV)
model = BPVAE(h_dim=d["h_dim"], hidden=64).to(DEV); model.load_state_dict(d["vae"]); model.eval()
val = load_pool("cache/acts/bt_val_rich", 30, seed=2)

rows = []
with torch.no_grad():
    for hh, b, db in val:
        T = min(hh.shape[0], b.shape[0], 1600)
        z0 = torch.zeros(1, T, device=DEV)
        _, pm, _ = rollout(model, hh[:T].unsqueeze(0).to(DEV), z0, z0, sample=False, compute_kl=False)
        phi = pm[0].cpu().numpy()
        dphi = np.diff(phi)
        dwrap = np.where(dphi < -math.pi, dphi + TWO_PI, np.where(dphi > math.pi, dphi - TWO_PI, dphi))
        revs = float(np.sum(dwrap) / TWO_PI)                    # net phi revolutions
        med_adv = float(np.median(dwrap[dwrap > 1e-4])) if np.any(dwrap > 1e-4) else 0.0
        tempo = M * med_adv / TWO_PI * FPS * 60                 # BPM from median forward advance
        mono = float(np.mean(dwrap > 0))                        # fraction advancing (vs oscillating)
        gt_db = int((db.numpy()[:T] > 0.5).sum())               # GT #downbeats = #bars
        bf = np.where(b.numpy()[:T] > 0.5)[0]
        gt_tempo = 60 * FPS / np.median(np.diff(bf)) if len(bf) > 2 else float("nan")
        rows.append((revs, gt_db, tempo, gt_tempo, mono))

a = np.array(rows)
print(f"checkpoint: checkpoints/diagram_rerun.pt | {len(rows)} val songs | M={M}")
print(f"  phi revolutions : mean {a[:,0].mean():.1f}   vs GT #bars mean {a[:,1].mean():.1f}")
print(f"  effective tempo : mean {a[:,2].mean():.0f} BPM  vs GT tempo mean {np.nanmean(a[:,3]):.0f} BPM")
print(f"  monotonic frac  : mean {a[:,4].mean():.2f}  (1.0 = pure rotation, ~0.5 = oscillation)")
print(f"  per-song revs/bars ratio: mean {np.mean(a[:,0]/np.maximum(a[:,1],1)):.2f}  (1.0 = locked rotation)")
print(f"  sample (revs, bars, tempo, gt_tempo): " + ", ".join(f"({r[0]:.0f},{int(r[1])},{r[2]:.0f},{r[3]:.0f})" for r in rows[:6]))
