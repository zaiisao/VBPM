"""IDEA #3 (the synthesis): SURROUNDING BEATS -> TEMPO via a differentiable COMB-FILTER TEMPOGRAM.
  h -> BiGRU -> learned beat-activation a_t (grounded by beat-BCE, the easy part, ~0.84-able)
  comb-filter: response_k(t) = sum_j a[t +/- j*p_k]  (how well period p_k matches the surrounding beats)
  soft tempo selection: phi-dot_t = sum_k softmax(response_k(t)) * adv_k
  phi = offset + integral(phi-dot)   (rotates at the comb-selected tempo)
  geometric emission kcos(M*phi)/kcos(phi); deploy = geometric read-out (phi wraps).
The tempo is COMPUTED from surrounding beats (local, well-conditioned conv) -- not learned through the
ill-conditioned integral gradient. This is the architectural form of 'surrounding beats inform tempo'.
"""
import sys, math, importlib.util, argparse, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"; sys.path.insert(0, ROOT)
s = importlib.util.spec_from_file_location("da", f"{ROOT}/experiments/diagram_arch/run.py")
da = importlib.util.module_from_spec(s); s.loader.exec_module(da)
load_pool, sample_batch, fmeas, phase_beats, phase_downbeats = da.load_pool, da.sample_batch, da.fmeas, da.phase_beats, da.phase_downbeats
DEV = da.DEV; TWO_PI = 2*math.pi; FPS = 86.1328125; M = 4; KAPPA = 8.0; K_T = 24
BPMS = np.exp(np.linspace(math.log(45), math.log(210), K_T))
PERIODS = np.maximum(np.round(60 * FPS / BPMS).astype(int), 2)         # frames per beat
ADV = torch.tensor(TWO_PI / (M * (60 * FPS / BPMS)), dtype=torch.float32, device=DEV)   # bar-phase adv/frame


class CombTempo(nn.Module):
    def __init__(self, h_dim, hid=128):
        super().__init__()
        self.gru = nn.GRU(h_dim, hid, batch_first=True, bidirectional=True, num_layers=1)
        self.beat = nn.Linear(2 * hid, 1)        # learned beat-activation
        self.offset = nn.Linear(2 * hid, 2)      # cos/sin of initial phase offset (per song)
        self.scale = nn.Parameter(torch.tensor(2.0))   # comb softmax sharpness

    def forward(self, h):
        ctx, _ = self.gru(h)                     # [B,T,2hid]
        a_logit = self.beat(ctx).squeeze(-1)            # [B,T] beat logits
        a = torch.sigmoid(a_logit)                       # beat-activation for the comb
        # comb-filter tempogram: response_k(t) = sum_j a[t-j p] + a[t+j p]
        B, T = a.shape; J = 4
        resp = []
        for p in PERIODS:
            r = a.new_zeros(B, T)
            for j in range(1, J + 1):
                r = r + F.pad(a, (j * int(p), 0))[:, :T] + F.pad(a, (0, j * int(p)))[:, j * int(p):]
            resp.append(r)
        resp = torch.stack(resp, -1)             # [B,T,K]
        w = F.softmax(self.scale * resp, -1)
        phidot = (w * ADV).sum(-1)               # [B,T]  comb-selected tempo
        off = self.offset(ctx.mean(1)); off = torch.atan2(off[:, 1], off[:, 0])   # [B] per-song offset
        phi = (off.unsqueeze(1) + torch.cumsum(phidot, 1)) % TWO_PI               # [B,T]
        return a, a_logit, phidot, phi


def geom_logits(phi):
    return torch.stack([KAPPA * torch.cos(M * phi), KAPPA * torch.cos(phi)], -1)


@torch.no_grad()
def evaluate(model, val, h_mode="real", frames=1600):
    model.eval(); gb, gd, revs, bpm = [], [], [], []; n = len(val)
    for i, (hh, b, db) in enumerate(val):
        h_use = val[(i + 1) % n][0] if h_mode == "shuffle" else hh
        T = min(h_use.shape[0], b.shape[0], frames)
        h_in = torch.zeros(1, T, hh.shape[1], device=DEV) if h_mode == "zero" else h_use[:T].unsqueeze(0).to(DEV)
        a, a_logit, phidot, phi = model(h_in); phi = phi[0].cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS; dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2: gb.append(fmeas(ref, phase_beats(phi, M)))
        if len(dref) >= 2: gd.append(fmeas(dref, phase_downbeats(phi)))
        dphi = np.diff(phi); adv = np.where(dphi < -math.pi, dphi + TWO_PI, dphi)
        revs.append(float(np.sum(adv) / TWO_PI)); a2 = adv[adv > 1e-4]
        bpm.append(M * float(np.median(a2)) / TWO_PI * FPS * 60 if len(a2) else 0.0)
    model.train(); mn = lambda x: float(np.nanmean(x)) if x else float("nan")
    return mn(gb), mn(gd), mn(revs), mn(bpm)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--la", type=float, default=3.0); ap.add_argument("--n_train", type=int, default=400); ap.add_argument("--n_val", type=int, default=40)
    a = ap.parse_args(); torch.manual_seed(0); np.random.seed(0); random.seed(0)
    train = load_pool("cache/acts/bt_train_rich", a.n_train, seed=1); val = load_pool("cache/acts/bt_val_rich", a.n_val, seed=2)
    nb = int(np.mean([(d > 0.5).sum() for _, _, d in val]))
    print(f"FAITHFUL v7 (COMB-FILTER tempogram -> tempo; K={K_T}) | train={len(train)} val={len(val)} | GT #bars~{nb}", flush=True)
    model = CombTempo(h_dim=512).to(DEV); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pw = torch.tensor([8.0, 20.0], device=DEV)
    def dilate(x, w=3):
        k = torch.ones(1, 1, 2 * w + 1, device=x.device)
        return (F.conv1d(x.unsqueeze(1), k, padding=w).squeeze(1) > 0.5).float()
    for step in range(1, a.steps + 1):
        h, b, db = sample_batch(train, 256, 16)
        aa, aa_logit, phidot, phi = model(h)
        recon = F.binary_cross_entropy_with_logits(geom_logits(phi), torch.stack([b, db], -1), pos_weight=pw, reduction="none").sum((1, 2))
        beat_loss = F.binary_cross_entropy_with_logits(aa_logit, dilate(b), pos_weight=torch.tensor(8.0, device=DEV), reduction="none").sum(1)   # ground a (pos_weight!)
        loss = (recon + a.la * beat_loss).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % 100 == 0 or step == a.steps:
            gb, gd, rv, bp = evaluate(model, val, "real")
            print(f"  step {step:4d} | recon {float(recon.mean()):.1f} bl {float(beat_loss.mean()):.1f} | GEOM beat {gb:.3f} db {gd:.3f} | phi-revs {rv:.1f} tempo {bp:.0f}BPM", flush=True)
    gb, gd, rv, bp = evaluate(model, val, "real"); gbs, _, _, _ = evaluate(model, val, "shuffle"); gbz, _, _, _ = evaluate(model, val, "zero")
    print(f"\n--- FINAL (comb-filter tempo) ---")
    print(f"  real {gb:.3f} db {gd:.3f} revs {rv:.1f} tempo {bp:.0f}BPM | shuf {gbs:.3f} zero {gbz:.3f}  (must COLLAPSE)")
    print("VERDICT: phi rotates at musical tempo + beat HIGH + leak collapses => surrounding-beats->comb->tempo WORKS")
    print("DONE")


if __name__ == "__main__":
    main()
