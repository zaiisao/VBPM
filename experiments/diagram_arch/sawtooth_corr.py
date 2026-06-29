"""DECISIVE: is phidot tracking PER-SONG tempo, or collapsed to the dataset-average constant (~115)?
Train lambda=0.5 sawtooth-aux, then per val song compute MODEL tempo (median bar-phase advance from the
posterior phi read-out) vs GT tempo (from beat intervals). Report Pearson r + spread of model tempo.
If r is high and model tempo SPREADS across songs -> genuinely per-song grounded. If model tempo is ~flat
(~115 for everyone) -> it's the old dataset-average collapse, just at a musical value.
"""
import sys, math, importlib.util, random
import numpy as np, torch
ROOT="/home/sogang/jaehoon/CHART"; sys.path.insert(0,ROOT)
s=importlib.util.spec_from_file_location("sa",f"{ROOT}/experiments/diagram_arch/sawtooth_aux.py")
sa=importlib.util.module_from_spec(s); s.loader.exec_module(sa)
da=sa.da; BPVAE=da.BPVAE; rollout=da.rollout; load_pool=da.load_pool; sample_batch=da.sample_batch
DEV=da.DEV; FPS=86.1328125; M=4; TWO_PI=2*math.pi

torch.manual_seed(0); np.random.seed(0); random.seed(0)
train=load_pool("cache/acts/bt_train_rich",400,seed=1); val=load_pool("cache/acts/bt_val_rich",40,seed=2)
model=BPVAE(h_dim=512,hidden=64).to(DEV); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
STEPS=1000
for step in range(1,STEPS+1):
    temp=1.0+(0.3-1.0)*min(step/STEPS,1.0)
    h,b,db=sample_batch(train,256,16)
    loss,info=sa.elbo_aux(model,h,b,db,temp,0.5)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
    if step%250==0: print(f"  step {step} recon {info['recon']:.0f} Lph {info['Lphase']:.3f}",flush=True)

model.eval(); mod=[]; gt=[]
with torch.no_grad():
    for hh,b,dbv in val:
        T=min(hh.shape[0],b.shape[0],1600)
        bt=np.where(b.numpy()[:T]>0.5)[0]/FPS
        if len(bt)<2: continue
        gtbpm=60.0/np.median(np.diff(bt))
        _,phase_mu,_=rollout(model,hh[:T].unsqueeze(0).to(DEV),torch.zeros(1,T,device=DEV),torch.zeros(1,T,device=DEV),sample=False,compute_kl=False)
        phi=phase_mu[0].cpu().numpy(); d=np.diff(phi); adv=np.where(d<-math.pi,d+TWO_PI,d); adv=adv[adv>1e-4]
        if len(adv)==0: continue
        modbpm=M*float(np.median(adv))/TWO_PI*FPS*60
        mod.append(modbpm); gt.append(gtbpm)
mod=np.array(mod); gt=np.array(gt)
r=np.corrcoef(mod,gt)[0,1]
# octave-tolerant: also correlate after mapping model to nearest octave of gt
def oct_align(m,g):
    best=m
    for f in (0.5,2.0):
        if abs(m*f-g)<abs(best-g): best=m*f
    return best
moda=np.array([oct_align(m,g) for m,g in zip(mod,gt)])
ra=np.corrcoef(moda,gt)[0,1]
print(f"\nPER-SONG TEMPO ({len(mod)} songs):")
print(f"  GT    : median {np.median(gt):.0f} std {gt.std():.0f} range {gt.min():.0f}-{gt.max():.0f}")
print(f"  MODEL : median {np.median(mod):.0f} std {mod.std():.0f} range {mod.min():.0f}-{mod.max():.0f}")
print(f"  Pearson r (raw)           = {r:.3f}")
print(f"  Pearson r (octave-aligned)= {ra:.3f}")
print(f"  >>> high r + model std comparable to GT std => PER-SONG grounded; model std~0 => dataset-avg collapse")
print("  sample (gt -> model):", ", ".join(f"{g:.0f}->{m:.0f}" for g,m in list(zip(gt,mod))[:12]))
