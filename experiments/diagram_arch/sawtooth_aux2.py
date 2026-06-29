"""BAR + BEAT phase supervision. Bar-phase phi -> bar-sawtooth (downbeats); beat-phase (M*phi mod 2pi)
-> beat-sawtooth (beats). Beat-phase pins the within-bar subdivision -> should fix octave errors + beat-F.
Reports geometric beat/db F + LEAK + per-song tempo correlation (the collapse-catch test).
"""
import sys, math, importlib.util, random, argparse
import numpy as np, torch, torch.nn.functional as F
ROOT="/home/sogang/jaehoon/CHART"; sys.path.insert(0,ROOT)
s=importlib.util.spec_from_file_location("sa",f"{ROOT}/experiments/diagram_arch/sawtooth_aux.py")
sa=importlib.util.module_from_spec(s); s.loader.exec_module(sa)
da=sa.da; BPVAE=da.BPVAE; rollout=da.rollout; load_pool=da.load_pool; sample_batch=da.sample_batch
phase_beats=da.phase_beats; phase_downbeats=da.phase_downbeats; fmeas=da.fmeas
gt_batch=sa.gt_batch; DEV=da.DEV; FPS=86.1328125; M=4; TWO_PI=2*math.pi


def elbo_aux2(model,h,b,db,temp,lam_bar,lam_beat,pw_b=8.0,pw_db=20.0,fb=0.1,b_drop=0.5):
    B,T,_=h.shape
    keep=(torch.rand(B,1,device=h.device)>=b_drop).float()
    (klm,klp,klt),phase_mu,logits=rollout(model,h,b*keep,db*keep,temp,sample=True,compute_kl=True)
    pw=torch.tensor([pw_b,pw_db],device=h.device)
    recon=F.binary_cross_entropy_with_logits(logits,torch.stack([b,db],-1),pos_weight=pw,reduction="none").sum((1,2))
    klm=klm.clamp(min=fb*T); klp=klp.clamp(min=fb*T); klt=klt.clamp(min=fb*T)
    bar_gt,bar_m=gt_batch(db); beat_gt,beat_m=gt_batch(b)
    Lbar=(((1-torch.cos(phase_mu-bar_gt))*bar_m).sum(1)/bar_m.sum(1).clamp(min=1.0))
    Lbeat=(((1-torch.cos((M*phase_mu)%TWO_PI-beat_gt))*beat_m).sum(1)/beat_m.sum(1).clamp(min=1.0))
    loss=(recon+klm+klp+klt+lam_bar*T*Lbar+lam_beat*T*Lbeat).mean()
    return loss,{"recon":float(recon.mean()),"Lbar":float(Lbar.mean()),"Lbeat":float(Lbeat.mean())}


@torch.no_grad()
def evaluate(model,val,h_mode="real",frames=1600):
    model.eval(); gb,gd=[],[]; n=len(val)
    for i,(hh,b,db) in enumerate(val):
        h_use=val[(i+1)%n][0] if h_mode=="shuffle" else hh
        T=min(h_use.shape[0],b.shape[0],frames)
        h_in=torch.zeros(1,T,hh.shape[1],device=DEV) if h_mode=="zero" else h_use[:T].unsqueeze(0).to(DEV)
        z=torch.zeros(1,T,device=DEV)
        _,phase_mu,_=rollout(model,h_in,z,z,sample=False,compute_kl=False)
        phi=phase_mu[0].cpu().numpy()
        ref=np.where(b.numpy()[:T]>0.5)[0]/FPS; dref=np.where(db.numpy()[:T]>0.5)[0]/FPS
        if len(ref)>=2: gb.append(fmeas(ref,phase_beats(phi,M)))
        if len(dref)>=2: gd.append(fmeas(dref,phase_downbeats(phi)))
    model.train(); f=lambda x:float(np.nanmean(x)) if x else float("nan")
    return f(gb),f(gd)


@torch.no_grad()
def tempo_corr(model,val,frames=1600):
    model.eval(); mod,gt=[],[]
    for hh,b,db in val:
        T=min(hh.shape[0],b.shape[0],frames); bt=np.where(b.numpy()[:T]>0.5)[0]/FPS
        if len(bt)<2: continue
        gtbpm=60.0/np.median(np.diff(bt))
        _,phase_mu,_=rollout(model,hh[:T].unsqueeze(0).to(DEV),torch.zeros(1,T,device=DEV),torch.zeros(1,T,device=DEV),sample=False,compute_kl=False)
        phi=phase_mu[0].cpu().numpy(); d=np.diff(phi); adv=np.where(d<-math.pi,d+TWO_PI,d); adv=adv[adv>1e-4]
        if len(adv)==0: continue
        mod.append(M*float(np.median(adv))/TWO_PI*FPS*60); gt.append(gtbpm)
    model.train(); mod=np.array(mod); gt=np.array(gt)
    oa=np.array([min([m*f for f in(0.5,1,2)],key=lambda x:abs(x-g)) for m,g in zip(mod,gt)])
    return np.corrcoef(mod,gt)[0,1],np.corrcoef(oa,gt)[0,1],mod.std(),gt.std()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=1000)
    ap.add_argument("--lam_bar",type=float,default=0.5); ap.add_argument("--lam_beat",type=float,default=0.5)
    a=ap.parse_args()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    train=load_pool("cache/acts/bt_train_rich",400,seed=1); val=load_pool("cache/acts/bt_val_rich",40,seed=2)
    print(f"BAR+BEAT sawtooth | lam_bar={a.lam_bar} lam_beat={a.lam_beat} | train={len(train)} val={len(val)}\n",flush=True)
    model=BPVAE(h_dim=512,hidden=64).to(DEV); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for step in range(1,a.steps+1):
        temp=1.0+(0.3-1.0)*min(step/a.steps,1.0)
        h,b,db=sample_batch(train,256,16)
        loss,info=elbo_aux2(model,h,b,db,temp,a.lam_bar,a.lam_beat)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        if step%200==0 or step==a.steps:
            gb,gd=evaluate(model,val,"real")
            print(f"  step {step:4d} | recon {info['recon']:.0f} Lbar {info['Lbar']:.3f} Lbeat {info['Lbeat']:.3f} | beat {gb:.3f} db {gd:.3f}",flush=True)
    gb,gd=evaluate(model,val,"real"); gbs,gds=evaluate(model,val,"shuffle"); gbz,gdz=evaluate(model,val,"zero")
    r,ra,ms,gs=tempo_corr(model,val)
    print(f"\n==== BAR+BEAT VERDICT ====")
    print(f"  real beat {gb:.3f} db {gd:.3f} | shuf {gbs:.3f}/{gds:.3f} | zero {gbz:.3f}/{gdz:.3f}")
    print(f"  tempo r raw {r:.3f} oct-aligned {ra:.3f} | model std {ms:.0f} GT std {gs:.0f}")
    print(f"  (vs bar-only: beat 0.618 db 0.771 r 0.624/0.903)")
    print("DONE")


if __name__=="__main__": main()
