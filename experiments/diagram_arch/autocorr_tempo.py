"""B: differentiable AUTOCORRELATION tempo head on h (the correct 'derivative of h' = periodicity).
onset(h) -> autocorrelation over lags -> soft-argmax lag = period = tempo. Tests whether the RATE is
recoverable from h with the right operator (windowed, not pointwise). Expect: yes, with octave errors.
Trained against GT beat period (CE over lag bins). Reports per-song tempo r (raw + octave-tolerant).
"""
import sys, math, importlib.util, random, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT="/home/sogang/jaehoon/CHART"; sys.path.insert(0,ROOT)
s=importlib.util.spec_from_file_location("da",f"{ROOT}/experiments/diagram_arch/run.py")
da=importlib.util.module_from_spec(s); s.loader.exec_module(da)
load_pool=da.load_pool; sample_batch=da.sample_batch; DEV=da.DEV; FPS=86.1328125
LAGS=np.arange(20,140)                                   # beat period frames: ~40-258 BPM
LAGS_T=torch.tensor(LAGS,dtype=torch.long)


class TempoNet(nn.Module):
    def __init__(self,h_dim=512):
        super().__init__()
        self.onset=nn.Sequential(nn.Linear(h_dim,64),nn.ReLU(),nn.Linear(64,1))
        self.scale=nn.Parameter(torch.tensor(5.0))
    def tempogram(self,h):                               # h [B,T,H] -> logits over lags [B,nlag]
        o=F.softplus(self.onset(h).squeeze(-1))          # [B,T] learned onset
        o=o-o.mean(1,keepdim=True)
        T=o.shape[1]; energy=o.pow(2).mean(1)+1e-6
        acs=[]
        for L in LAGS:
            acs.append((o[:,:T-L]*o[:,L:]).mean(1)/energy)
        ac=torch.stack(acs,1)                            # [B,nlag] normalized autocorrelation
        return ac*F.softplus(self.scale)
    def forward(self,h): return self.tempogram(h)


def gt_period_frames(bvec):
    bf=np.where(bvec>0.5)[0]
    return (np.median(np.diff(bf)) if len(bf)>=2 else np.nan)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=1500)
    a=ap.parse_args()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    train=load_pool("cache/acts/bt_train_rich",400,seed=1); val=load_pool("cache/acts/bt_val_rich",40,seed=2)
    print(f"AUTOCORR tempo head | train={len(train)} val={len(val)} | lags {LAGS[0]}-{LAGS[-1]} frames\n",flush=True)
    net=TempoNet(512).to(DEV); opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    for step in range(1,a.steps+1):
        h,b,db=sample_batch(train,256,16)
        logits=net(h)                                    # [B,nlag]
        # target lag bin = nearest LAG to GT period per sample
        tgt=[]; valid=[]
        bn=b.cpu().numpy()
        for j in range(bn.shape[0]):
            p=gt_period_frames(bn[j])
            if np.isnan(p): tgt.append(0); valid.append(0.0)
            else: tgt.append(int(np.argmin(np.abs(LAGS-p)))); valid.append(1.0)
        tgt=torch.tensor(tgt,device=DEV); valid=torch.tensor(valid,device=DEV)
        ce=F.cross_entropy(logits,tgt,reduction="none")
        loss=(ce*valid).sum()/valid.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step%300==0 or step==a.steps:
            print(f"  step {step} loss {float(loss):.3f}",flush=True)
    # eval per-song tempo correlation
    net.eval(); mod,gt=[],[]
    with torch.no_grad():
        for hh,b,db in val:
            T=min(hh.shape[0],b.shape[0],1600)
            p=gt_period_frames(b.cpu().numpy()[:T])
            if np.isnan(p): continue
            lg=net(hh[:T].unsqueeze(0).to(DEV))[0]
            lagpred=LAGS[int(lg.argmax())]
            mod.append(FPS*60/lagpred); gt.append(FPS*60/p)
    mod=np.array(mod); gt=np.array(gt)
    oa=np.array([min([m*f for f in(0.5,1,2)],key=lambda z:abs(z-g)) for m,g in zip(mod,gt)])
    print(f"\n==== AUTOCORR TEMPO (B) ====")
    print(f"  per-song tempo: r raw {np.corrcoef(mod,gt)[0,1]:.3f} oct {np.corrcoef(oa,gt)[0,1]:.3f}")
    print(f"  model std {mod.std():.0f} GT std {gt.std():.0f} | model median {np.median(mod):.0f} GT {np.median(gt):.0f}")
    print(f"  sample (gt->mod): "+", ".join(f"{g:.0f}->{m:.0f}" for g,m in list(zip(gt,mod))[:10]))
    print("  HIGH r + std~GT => tempo IS recoverable from h via autocorrelation (compute, not learn-as-latent)")
    print("DONE-B")


if __name__=="__main__": main()
