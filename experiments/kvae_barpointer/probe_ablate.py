"""WHERE does the decoder get beats/tempo? Ablate (time-shuffle) each latent component at the decoder
input and measure decoder beat-F. If shuffling phi destroys it but shuffling lt(=dotphi) doesn't, the
decoder relies on phi as a per-frame beat-marker and phi-dot is redundant (-> never grounded)."""
import sys, math, importlib.util
import numpy as np, torch, torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"; sys.path.insert(0, ROOT)
s = importlib.util.spec_from_file_location("da", f"{ROOT}/experiments/diagram_arch/run.py")
da = importlib.util.module_from_spec(s); s.loader.exec_module(da)
BPVAE, load_pool, peaks, fmeas = da.BPVAE, da.load_pool, da.peaks, da.fmeas
DEV = da.DEV; FPS = 86.1328125


def capture(model, h):
    B, T, _ = h.shape; pc = model.enc_post(h, torch.zeros(B, T, device=DEV), torch.zeros(B, T, device=DEV))
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, 0], z0], -1)))
    m = F.softmax(qm, -1); phi = qpm; lt = qtm
    ms, phis, lts = [m], [phi], [lt]; mprev, phiprev, ltprev = m, phi, lt
    for t in range(1, T):
        zp = model.zfeat(mprev, phiprev, ltprev)
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, t], zp], -1)))
        m = F.softmax(qm, -1); phi = qpm; lt = qtm
        ms.append(m); phis.append(phi); lts.append(lt); mprev, phiprev, ltprev = m, phi, lt
    return torch.cat(ms), torch.cat(phis), torch.cat(lts)   # [T,*]


def beatF(model, m, phi, lt, b, T):
    zf = model.zfeat(m, phi, lt)
    prob = torch.sigmoid(torch.stack([model.decode(zf[t:t+1]) for t in range(T)], 1))[0, :, 0].cpu().numpy()
    ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS
    return fmeas(ref, peaks(prob)) if len(ref) >= 2 else np.nan


def main():
    d = torch.load("checkpoints/diagram_rerun.pt", map_location=DEV)
    model = BPVAE(h_dim=d["h_dim"], hidden=64).to(DEV); model.load_state_dict(d["vae"]); model.eval()
    val = load_pool("cache/acts/bt_val_rich", 40, seed=2)
    full, no_phi, no_lt, no_m = [], [], [], []
    g = torch.Generator(device=DEV)
    with torch.no_grad():
        for hh, b, db in val:
            T = min(hh.shape[0], b.shape[0], 1600)
            m, phi, lt = capture(model, hh[:T].unsqueeze(0).to(DEV))
            perm = torch.randperm(T, device=DEV)
            full.append(beatF(model, m, phi, lt, b, T))
            no_phi.append(beatF(model, m, phi[perm], lt, b, T))      # shuffle phi over time
            no_lt.append(beatF(model, m, phi, lt[perm], b, T))       # shuffle lt(=dotphi) over time
            no_m.append(beatF(model, m[perm], phi, lt, b, T))        # shuffle meter
    mn = lambda x: float(np.nanmean(x))
    print(f"n={len(full)} | decoder beat-F under time-shuffle ablation of each latent component:")
    print(f"  FULL (no ablation)        : {mn(full):.3f}")
    print(f"  shuffle phi  (the phase)  : {mn(no_phi):.3f}   drop {mn(full)-mn(no_phi):+.3f}")
    print(f"  shuffle lt (= dotphi/tempo): {mn(no_lt):.3f}   drop {mn(full)-mn(no_lt):+.3f}")
    print(f"  shuffle m  (the meter)    : {mn(no_m):.3f}   drop {mn(full)-mn(no_m):+.3f}")
    print("\nIF shuffling phi destroys beats but shuffling dotphi does NOT => decoder uses phi as a per-frame")
    print("   beat-marker and dotphi is REDUNDANT -> the recon never pushes gradient into dotphi (root cause).")


if __name__ == "__main__":
    main()
