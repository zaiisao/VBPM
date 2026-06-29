"""DECISIVE re-test after finding the clamp/init bug: in phi_capacity the tempo head inits near 0, which
is ABOVE the log-tempo range [-4.4,-2.57], so it pinned at the LT_MAX clamp (248 BPM) with ZERO gradient
-> the 'flat loss' was largely a dead-gradient artifact, not proof continuous-phi can't lock.

Fix: SOFT, always-differentiable tempo map  omega = exp(LT_MIN + (LT_MAX-LT_MIN)*sigmoid(raw))  (init mid
-range ~100 BPM, gradient everywhere). Then test three objectives on phi = cumsum(omega) from the EXACT
filtered latent:
  1. pure phase-sup  : (1-cos(phi - GP))            (the phi_capacity objective, now un-clamped)
  2. rate-sup        : MSE(log_omega, log(GT advance))  (LOCAL, well-conditioned target)
  3. rate + offset   : rate-sup + weak (1-cos) to anchor absolute phase
Report geometric beat-F, phi-revs, tempo(BPM), leak. If any locks (beats high, revs~#bars, tempo~real,
leak collapses) -> continuous-phi IS salvageable, earlier 'dead end' was the bug/objective.
"""
import sys, math, random, importlib.util
import numpy as np
import torch, torch.nn as nn

ROOT = "/home/sogang/jaehoon/CHART"
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/third_party/kalman-vae")
kr = importlib.util.spec_from_file_location("kr", f"{ROOT}/experiments/kvae_barpointer/kvae_run.py")
m = importlib.util.module_from_spec(kr); kr.loader.exec_module(m)
KVAEBarPointer, kvae_elbo, load, batch = m.KVAEBarPointer, m.kvae_elbo, m.load, m.batch
da = m.da; phase_beats, phase_downbeats, fmeas = da.phase_beats, da.phase_downbeats, da.fmeas
from kvae.sample_control import SampleControl
DEV = m.DEV; TWO_PI = 2*math.pi; FPS = 86.1328125; M = 4
LT_MIN = math.log(TWO_PI*40/60/M/FPS); LT_MAX = math.log(TWO_PI*250/60/M/FPS)


def gt_barphase(bf, mm, T):
    phi = np.zeros(T)
    if len(bf) < 2: return phi
    vals = np.arange(len(bf))*(TWO_PI/mm)
    for k in range(len(bf)-1):
        a, b = bf[k], bf[k+1]; phi[a:b] = np.linspace(vals[k], vals[k+1], b-a, endpoint=False)
    phi[bf[-1]:] = vals[-1]; return phi % TWO_PI


def soft_logomega(raw):                              # always-differentiable; no dead clamp
    return LT_MIN + (LT_MAX - LT_MIN) * torch.sigmoid(raw)


def integ(z, th):
    lo = soft_logomega(th(z).squeeze(-1))           # [.,.]
    return torch.cumsum(torch.exp(lo), 0) % TWO_PI, lo


@torch.no_grad()
def ev(model, th, val, h_mode="real", frames=1600):
    model.eval(); th.eval(); sc = SampleControl(encoder="mean",decoder="mean",state_transition="mean",observation="mean")
    gb, gd, rev, bpm = [], [], [], []; n=len(val)
    for i,(hh,b,db) in enumerate(val):
        h_use = val[(i+1)%n][0] if h_mode=="shuffle" else hh
        T = min(h_use.shape[0], b.shape[0], frames)
        h_in = torch.zeros(T,1,hh.shape[1],device=DEV) if h_mode=="zero" else h_use[:T].unsqueeze(1).to(DEV)
        a = model.encoder(h_in.reshape(-1,hh.shape[1])).mean.view(T,1,model.a_dim)
        fm,*_ = model.ssm.kalman_filter(a, sample_control=sc)
        phi, lo = integ(fm, th); phi = phi[:,0].cpu().numpy()
        ref = np.where(b.numpy()[:T]>0.5)[0]/FPS; dref = np.where(db.numpy()[:T]>0.5)[0]/FPS
        if len(ref)>=2: gb.append(fmeas(ref, phase_beats(phi, M)))
        if len(dref)>=2: gd.append(fmeas(dref, phase_downbeats(phi)))
        d = np.diff(phi); rev.append(float(np.sum(np.where(d<-math.pi,d+TWO_PI,d))/TWO_PI))
        bpm.append(60*FPS*M*float(torch.exp(lo[:,0]).mean().cpu())/TWO_PI)
    model.train(); th.train(); f=lambda x: float(np.nanmean(x)) if x else float("nan")
    return f(gb), f(gd), f(rev), f(bpm)


def run(train, val, mode, steps, tag):
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    model = KVAEBarPointer(h_dim=512, a_dim=8, z_dim=8, K=5).to(DEV)
    th = nn.Sequential(nn.Linear(8,32), nn.ReLU(), nn.Linear(32,1)).to(DEV)
    opt = torch.optim.Adam(list(model.parameters())+list(th.parameters()), lr=1e-3)
    sc = SampleControl(encoder="sample",decoder="mean",state_transition="sample",observation="sample")
    for step in range(1, steps+1):
        H, Bt, Dt = batch(train, 256, 16)
        fm,fc,fnm,fnc,mA,mC,_,_ = model.ssm.kalman_filter(
            model.encoder(H.reshape(-1,512)).rsample().view(*H.shape[:2],model.a_dim), sample_control=sc)
        phi, lo = integ(fm, th)
        GP = torch.zeros_like(phi)
        for j in range(Bt.shape[1]):
            bf = torch.where(Bt[:,j]>0.5)[0].cpu().numpy()
            GP[:,j] = torch.tensor(gt_barphase(bf, M, Bt.shape[0]), device=DEV, dtype=phi.dtype)
        # GT per-frame advance (local rate target)
        adv = torch.diff(GP, dim=0) % TWO_PI
        adv = adv.clamp(math.exp(LT_MIN), math.exp(LT_MAX))
        loss_phase = (1.0 - torch.cos(phi - GP)).mean()
        loss_rate = ((lo[1:] - torch.log(adv))**2).mean()
        if mode == "phase": loss = loss_phase
        elif mode == "rate": loss = loss_rate
        else: loss = loss_rate + 0.5*loss_phase
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(th.parameters()),5.0); opt.step()
        if step % 300 == 0 or step == steps:
            gb, gd, rv, bp = ev(model, th, val)
            print(f"  [{tag}] step {step} | loss {float(loss):.3f} | beat {gb:.3f} db {gd:.3f} | revs {rv:.1f} tempo {bp:.0f}BPM", flush=True)
    gb,gd,rv,bp = ev(model, th, val); gbs,_,_,_ = ev(model,th,val,"shuffle"); gbz,_,_,_ = ev(model,th,val,"zero")
    print(f"  [{tag}] FINAL beat {gb:.3f} db {gd:.3f} revs {rv:.1f} tempo {bp:.0f} | shuf {gbs:.3f} zero {gbz:.3f}", flush=True)
    return gb, gbs, gbz, rv, bp


def main():
    train = load("cache/acts/bt_train_rich", 300, 1); val = load("cache/acts/bt_val_rich", 30, 2)
    print(f"PHI-RATE (soft tempo, no dead clamp) | train={len(train)} val={len(val)}", flush=True)
    nb = int(np.mean([(d>0.5).sum() for _,_,d in val]))
    print(f"  (val avg #downbeats ~ {nb}; a locked phi should have revs ~ that)\n", flush=True)
    print("1. PURE PHASE-SUP (un-clamped):"); P = run(train, val, "phase", 1200, "phase")
    print("\n2. RATE-SUP (local log-advance MSE):"); R = run(train, val, "rate", 1200, "rate")
    print("\n3. RATE + weak phase-offset:"); RO = run(train, val, "rateoff", 1200, "rateoff")
    print("\n==== VERDICT ====")
    for tag, r in (("phase", P), ("rate", R), ("rate+off", RO)):
        print(f"  {tag:9s}: beat {r[0]:.3f}  revs {r[3]:.1f}  tempo {r[4]:.0f}BPM  | shuf {r[1]:.3f} zero {r[2]:.3f}")
    print("  ANY high beat + revs~#bars + tempo~real + leak collapse => continuous-phi SALVAGEABLE (bug/objective, not structural)")
    print("DONE")


if __name__ == "__main__":
    main()
