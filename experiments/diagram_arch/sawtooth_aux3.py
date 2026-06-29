"""UNIFIED geometric target + GEOMETRIC EMISSION (keep ELBO; dissolve the bar-vs-beat fight).
- ONE phase target phi_gt: 0 at downbeats AND 2pi*k/M at beat k (encodes beats+downbeats, no competition).
- LIKELIHOOD = geometric emission (faithful bar-pointer): beat ~ a_b*cos(M*phi), downbeat ~ a_d*cos(phi),
  replacing the free MLP decoder so the likelihood and the phase target pull phi the SAME way.
ELBO intact (KL meter/phase/tempo from rollout). Eval = posterior phi geometric read-out (sa2 helpers).
"""
import sys, math, importlib.util, random, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT="/home/sogang/jaehoon/CHART"; sys.path.insert(0,ROOT)
s2=importlib.util.spec_from_file_location("sa2",f"{ROOT}/experiments/diagram_arch/sawtooth_aux2.py")
sa2=importlib.util.module_from_spec(s2); s2.loader.exec_module(sa2)
da=sa2.da; BPVAE=da.BPVAE; rollout=da.rollout; load_pool=da.load_pool; sample_batch=da.sample_batch
DEV=da.DEV; FPS=86.1328125; M=4; TWO_PI=2*math.pi


def gt_unified(bvec, dbvec, T):
    """Single ramp: cumulative phase +2pi/M per beat, anchored so downbeats land at multiples of 2pi."""
    phi=np.zeros(T,np.float32); mask=np.zeros(T,np.float32)
    beats=np.where(bvec>0.5)[0]; dbf=set(np.where(dbvec>0.5)[0].tolist())
    if len(beats)<2: return phi,mask
    j0=next((j for j,bt in enumerate(beats) if bt in dbf),0)         # first downbeat among beats
    Phi=np.array([(j-j0)*(TWO_PI/M) for j in range(len(beats))],np.float32)  # unwrapped cumulative
    for k in range(len(beats)-1):
        a,c=beats[k],beats[k+1]
        phi[a:c]=np.linspace(Phi[k],Phi[k+1],c-a,endpoint=False); mask[a:c]=1.0
    phi=phi%TWO_PI
    return phi,mask


def gt_batch_uni(b,db):
    B,T=b.shape; P=np.zeros((B,T),np.float32); Mk=np.zeros((B,T),np.float32)
    bn=b.cpu().numpy(); dn=db.cpu().numpy()
    for j in range(B): P[j],Mk[j]=gt_unified(bn[j],dn[j],T)
    return torch.from_numpy(P).to(b.device),torch.from_numpy(Mk).to(b.device)


class Geom(nn.Module):                                   # geometric emission params (faithful likelihood)
    def __init__(self):
        super().__init__(); self.ab=nn.Parameter(torch.tensor(1.0)); self.ad=nn.Parameter(torch.tensor(1.0))
        self.cb=nn.Parameter(torch.tensor(-1.0)); self.cd=nn.Parameter(torch.tensor(-1.0))
    def logits(self,phi):
        beat=F.softplus(self.ab)*torch.cos(M*phi)+self.cb
        db=F.softplus(self.ad)*torch.cos(phi)+self.cd
        return torch.stack([beat,db],-1)


def elbo_geom(model,geom,h,b,db,temp,lam,fb=0.1,b_drop=0.5,pw_b=8.0,pw_db=20.0):
    B,T,_=h.shape
    keep=(torch.rand(B,1,device=h.device)>=b_drop).float()
    (klm,klp,klt),phase_mu,_=rollout(model,h,b*keep,db*keep,temp,sample=True,compute_kl=True)
    logits=geom.logits(phase_mu)
    pw=torch.tensor([pw_b,pw_db],device=h.device)
    recon=F.binary_cross_entropy_with_logits(logits,torch.stack([b,db],-1),pos_weight=pw,reduction="none").sum((1,2))
    klm=klm.clamp(min=fb*T); klp=klp.clamp(min=fb*T); klt=klt.clamp(min=fb*T)
    phi_gt,mask=gt_batch_uni(b,db)
    Lph=(((1-torch.cos(phase_mu-phi_gt))*mask).sum(1)/mask.sum(1).clamp(min=1.0))
    loss=(recon+klm+klp+klt+lam*T*Lph).mean()
    return loss,{"recon":float(recon.mean()),"Lph":float(Lph.mean())}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=1000); ap.add_argument("--lam",type=float,default=0.5)
    a=ap.parse_args()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    train=load_pool("cache/acts/bt_train_rich",400,seed=1); val=load_pool("cache/acts/bt_val_rich",40,seed=2)
    nb=int(np.mean([(d>0.5).sum() for _,_,d in val]))
    print(f"UNIFIED geom-emission | lam={a.lam} | train={len(train)} val={len(val)} GT#bars~{nb}\n",flush=True)
    model=BPVAE(h_dim=512,hidden=64).to(DEV); geom=Geom().to(DEV)
    opt=torch.optim.Adam(list(model.parameters())+list(geom.parameters()),lr=1e-3)
    for step in range(1,a.steps+1):
        temp=1.0+(0.3-1.0)*min(step/a.steps,1.0)
        h,b,db=sample_batch(train,256,16)
        loss,info=elbo_geom(model,geom,h,b,db,temp,a.lam)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(geom.parameters()),5.0); opt.step()
        if step%200==0 or step==a.steps:
            gb,gd=sa2.evaluate(model,val,"real")
            print(f"  step {step:4d} | recon {info['recon']:.0f} Lph {info['Lph']:.3f} | beat {gb:.3f} db {gd:.3f}",flush=True)
    gb,gd=sa2.evaluate(model,val,"real"); gbs,gds=sa2.evaluate(model,val,"shuffle"); gbz,gdz=sa2.evaluate(model,val,"zero")
    r,ra,ms,gs=sa2.tempo_corr(model,val)
    print(f"\n==== UNIFIED GEOM-EMISSION VERDICT ====")
    print(f"  real beat {gb:.3f} db {gd:.3f} | shuf {gbs:.3f}/{gds:.3f} | zero {gbz:.3f}/{gdz:.3f}")
    print(f"  tempo r raw {r:.3f} oct {ra:.3f} | model std {ms:.0f} GT std {gs:.0f}")
    print(f"  geom params: ab {float(F.softplus(geom.ab)):.2f} ad {float(F.softplus(geom.ad)):.2f}")
    print(f"  (bar-only: beat .618 db .771 r .62/.90 | bar+beat: beat .842 db .476 r .84/.99)")
    print("  WIN = db RECOVERS to ~.77 WITHOUT losing beat/tempo => unified target dissolved the fight")
    print("DONE")


if __name__=="__main__": main()
