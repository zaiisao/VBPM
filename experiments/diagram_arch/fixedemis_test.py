"""FIX TEST 3: integrate phi from tempo + FIXED geometric emission (no learned decoder, no tempo-sup).
beat_logit = kappa*cos(m*phi), db_logit = kappa*cos(phi). BCE on the REAL beats can only drop by making
phi wrap WHERE the audio's beats are -> audio-driven rotation by construction (tempo+offset driven by
data, nothing constant to cheat on). Deploy h-only, geometric read-out + leak controls.
"""
import sys, glob, math, random, importlib.util
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, "/home/sogang/jaehoon/CHART")
s = importlib.util.spec_from_file_location("da", "experiments/diagram_arch/run.py"); da = importlib.util.module_from_spec(s); s.loader.exec_module(da)
BPVAE, peaks, fmeas, phase_beats, phase_downbeats = da.BPVAE, da.peaks, da.fmeas, da.phase_beats, da.phase_downbeats
DEV = da.DEV; TWO_PI = 2 * math.pi; FPS = 86.1328125
KAPPA = 6.0; M = 4
# bounded tempo: clamp the per-frame phase advance to the [40,250] BPM range (Whiteley bounded RW)
LT_MIN = math.log(TWO_PI * 40 / 60 / M / FPS)    # ~ -4.41
LT_MAX = math.log(TWO_PI * 250 / 60 / M / FPS)   # ~ -2.58


def integ_phi(model, h, b_in, db_in):
    """phi = integral of the encoder tempo (tempo CLAMPED to a musical range). Returns phis, tempo_logs."""
    B, T, _ = h.shape
    pc = model.enc_post(h, b_in, db_in)
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, 0], z0], -1)))
    phi = qpm; lt = qtm.clamp(LT_MIN, LT_MAX); phis = [phi]; temps = [lt]; phiprev, ltprev = phi, lt
    mprev = F.softmax(qm, -1)
    for t in range(1, T):
        zp = model.zfeat(mprev, phiprev, ltprev)
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, t], zp], -1)))
        lt = qtm.clamp(LT_MIN, LT_MAX); phi = (phiprev + torch.exp(ltprev)) % TWO_PI
        phis.append(phi); temps.append(lt); mprev = F.softmax(qm, -1); phiprev, ltprev = phi, lt
    return torch.stack(phis, 1), torch.stack(temps, 1)


def emission_logits(phi):
    return torch.stack([KAPPA * torch.cos(M * phi), KAPPA * torch.cos(phi)], -1)   # [B,T,2] beat, downbeat


def gt_barphase(beat_fr, m, T):
    phi = np.zeros(T)
    if len(beat_fr) < 2: return phi
    vals = np.arange(len(beat_fr)) * (TWO_PI / m)
    for k in range(len(beat_fr) - 1):
        a, b = beat_fr[k], beat_fr[k + 1]; phi[a:b] = np.linspace(vals[k], vals[k + 1], b - a, endpoint=False)
    phi[beat_fr[-1]:] = vals[-1]
    return phi % TWO_PI


def load(cache_dir, n, seed):
    fs = sorted(glob.glob(f"{cache_dir}/*.pt")); random.Random(seed).shuffle(fs); out = []
    for f in fs[:n]:
        d = torch.load(f, map_location="cpu"); hh = d["activations"].float()
        if hh.shape[0] < 400 or d["beat_targets"].sum() < 8: continue
        out.append((hh, d["beat_targets"].float(), d["downbeat_targets"].float()))
    return out


def make_batch(songs, frames, bs):
    hs, bs_, ds_ = [], [], []
    while len(hs) < bs:
        hh, b, db = random.choice(songs)
        if hh.shape[0] <= frames: continue
        s0 = random.randint(0, hh.shape[0] - frames); bb = b[s0:s0 + frames]
        if bb.sum() < 2: continue
        hs.append(hh[s0:s0 + frames]); bs_.append(bb); ds_.append(db[s0:s0 + frames])
    return torch.stack(hs).to(DEV), torch.stack(bs_).to(DEV), torch.stack(ds_).to(DEV)


@torch.no_grad()
def evaluate(model, val, h_mode="real", frames=1600):
    model.eval(); gb, gd, revs, bpm = [], [], [], []
    n = len(val)
    for i, (hh, b, db) in enumerate(val):
        h_use = val[(i + 1) % n][0] if h_mode == "shuffle" else hh
        T = min(h_use.shape[0], b.shape[0], frames)
        h_in = torch.zeros(1, T, hh.shape[1], device=DEV) if h_mode == "zero" else h_use[:T].unsqueeze(0).to(DEV)
        z = torch.zeros(1, T, device=DEV)
        phis, temps = integ_phi(model, h_in, z, z); phi = phis[0].cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS; dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2: gb.append(fmeas(ref, phase_beats(phi, 4)))
        if len(dref) >= 2: gd.append(fmeas(dref, phase_downbeats(phi)))
        dphi = np.diff(phi); revs.append(float(np.sum(np.where(dphi < -math.pi, dphi + TWO_PI, dphi)) / TWO_PI))
        bpm.append(60 * FPS * 4 * float(np.exp(temps[0].cpu().numpy()).mean()) / TWO_PI)
    model.train(); m = lambda x: float(np.nanmean(x)) if x else float("nan")
    return m(gb), m(gd), m(revs), m(bpm)


def main():
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    b_drop = 0.5
    print(f"phi=INTEGRAL of tempo | FIXED emission kappa*cos(m*phi)/cos(phi) | NO learned decoder, NO tempo-sup | b_drop={b_drop}", flush=True)
    train = load("cache/acts/bt_train_rich", 200, 1); val = load("cache/acts/bt_val_rich", 40, 2)
    print(f"train={len(train)} val={len(val)}", flush=True)
    model = BPVAE(h_dim=512, hidden=64).to(DEV); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pw = torch.tensor([8.0, 20.0], device=DEV)
    for step in range(1, 601):
        h, b, db = make_batch(train, 256, 16)
        keep = (torch.rand(h.shape[0], 1, device=DEV) >= b_drop).float()
        phis, temps = integ_phi(model, h * 1.0, b * keep, db * keep)
        logits = emission_logits(phis)
        loss = F.binary_cross_entropy_with_logits(logits, torch.stack([b, db], -1), pos_weight=pw)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % 200 == 0 or step == 600:
            gb, gd, rv, bpm = evaluate(model, val, "real")
            print(f"\nstep {step} | loss {float(loss):.3f} | H-ONLY geometric: beat {gb:.3f} downbeat {gd:.3f} | "
                  f"phi-revs {rv:.1f} | tempo ~{bpm:.0f}BPM", flush=True)
    gb, gd, rv, bpm = evaluate(model, val, "real")
    gbs, gds, _, _ = evaluate(model, val, "shuffle"); gbz, gdz, _, _ = evaluate(model, val, "zero")
    print("\n--- FINAL (h-only deploy, GEOMETRIC bar-pointer read-out) ---")
    print(f"  real     : beat {gb:.3f}  downbeat {gd:.3f}  phi-revs {rv:.1f}  tempo ~{bpm:.0f}BPM")
    print(f"  shuffled : beat {gbs:.3f}  downbeat {gds:.3f}   (must COLLAPSE for it to be real)")
    print(f"  zero     : beat {gbz:.3f}  downbeat {gdz:.3f}   (must COLLAPSE for it to be real)")
    print("VERDICT: fixed emission -> if geometric beat/db HIGH and shuffled/zero COLLAPSE -> audio-driven rotation FIXED")


if __name__ == "__main__":
    main()
