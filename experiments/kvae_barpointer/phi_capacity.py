"""DECISIVE geometric test: CAN the integrate-tempo phi track the GT bar-phase AT ALL, given maximal
supervision? Train ONLY (1 - cos(phi - phi_GT)) (no ELBO, no geom-BCE) -- pure phase regression through
the exact filter. If beats then jump high + phi-revs ~ #bars -> representation/read-out are FINE, the
earlier failure was the OBJECTIVE (fixable). If it STILL can't track GT -> structural (the filtered
latent + tempo head can't carry a clean rotating phi).
"""
import sys, math, random, importlib.util
import numpy as np
import torch, torch.nn as nn

ROOT = "/home/sogang/jaehoon/CHART"
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/third_party/kalman-vae")
kr = importlib.util.spec_from_file_location("kr", f"{ROOT}/experiments/kvae_barpointer/kvae_run.py")
m = importlib.util.module_from_spec(kr); kr.loader.exec_module(m)
KVAEBarPointer, load, batch = m.KVAEBarPointer, m.load, m.batch
da = m.da; phase_beats, phase_downbeats, fmeas = da.phase_beats, da.phase_downbeats, da.fmeas
from kvae.sample_control import SampleControl
DEV = m.DEV; TWO_PI = 2*math.pi; FPS = 86.1328125; M = 4
LT_MIN = math.log(TWO_PI*40/60/M/FPS); LT_MAX = math.log(TWO_PI*250/60/M/FPS)

def gt_barphase(bf, mm, T):
    phi = np.zeros(T)
    if len(bf) < 2: return phi
    vals = np.arange(len(bf))*(TWO_PI/mm)
    for k in range(len(bf)-1):
        a,b = bf[k], bf[k+1]; phi[a:b] = np.linspace(vals[k], vals[k+1], b-a, endpoint=False)
    phi[bf[-1]:] = vals[-1]; return phi % TWO_PI

def integ(z, th):
    lt = th(z).squeeze(-1).clamp(LT_MIN, LT_MAX); return torch.cumsum(torch.exp(lt), 0) % TWO_PI

@torch.no_grad()
def ev(model, th, val, frames=1600):
    model.eval(); th.eval(); sc = SampleControl(encoder="mean",decoder="mean",state_transition="mean",observation="mean")
    gb, rev = [], []
    for hh,b,db in val:
        T = min(hh.shape[0], b.shape[0], frames)
        a = model.encoder(hh[:T].to(DEV)).mean.view(T,1,model.a_dim)
        fm,*_ = model.ssm.kalman_filter(a, sample_control=sc)
        phi = integ(fm, th)[:,0].cpu().numpy()
        ref = np.where(b.numpy()[:T]>0.5)[0]/FPS
        if len(ref)>=2: gb.append(fmeas(ref, phase_beats(phi, M)))
        d = np.diff(phi); rev.append(float(np.sum(np.where(d<-math.pi,d+TWO_PI,d))/TWO_PI))
    model.train(); th.train(); f=lambda x: float(np.nanmean(x)) if x else float("nan")
    return f(gb), f(rev)

def main():
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    train = load("cache/acts/bt_train_rich", 300, 1); val = load("cache/acts/bt_val_rich", 30, 2)
    print(f"PHI CAPACITY (pure phi-supervision, no ELBO) | train={len(train)} val={len(val)}", flush=True)
    model = KVAEBarPointer(h_dim=512, a_dim=8, z_dim=8, K=5).to(DEV)
    th = nn.Sequential(nn.Linear(8,32), nn.ReLU(), nn.Linear(32,1)).to(DEV)
    opt = torch.optim.Adam(list(model.parameters())+list(th.parameters()), lr=1e-3)
    sc = SampleControl(encoder="sample",decoder="mean",state_transition="sample",observation="sample")
    for step in range(1, 1201):
        H, Bt, Dt = batch(train, 256, 16)
        fm,fc,fnm,fnc,mA,mC,_,_ = model.ssm.kalman_filter(model.encoder(H.reshape(-1,512)).rsample().view(*H.shape[:2],model.a_dim), sample_control=sc)
        phi = integ(fm, th)
        GP = torch.zeros_like(phi)
        for j in range(Bt.shape[1]):
            bf = torch.where(Bt[:,j]>0.5)[0].cpu().numpy()
            GP[:,j] = torch.tensor(gt_barphase(bf, M, Bt.shape[0]), device=DEV, dtype=phi.dtype)
        loss = (1.0 - torch.cos(phi - GP)).mean()      # PURE phase supervision
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(th.parameters()),5.0); opt.step()
        if step % 300 == 0 or step == 1200:
            gb, rv = ev(model, th, val)
            print(f"  step {step} | sup-loss {float(loss):.3f} | GEOM beat {gb:.3f} | phi-revs {rv:.1f}", flush=True)
    print("VERDICT: high beat + revs~#bars => representation FINE, earlier failure was the OBJECTIVE; low => STRUCTURAL")
    print("DONE")

if __name__ == "__main__":
    main()
