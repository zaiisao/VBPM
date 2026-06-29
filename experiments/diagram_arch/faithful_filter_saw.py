"""A: FILTER (predict+correct) + sawtooth. phi_t = circular_blend(predict=phi_{t-1}+exp(phidot), data=qpm, gain g).
g=1 -> pure correction (free-phi, works but phidot unused); g=0 -> pure integrator (fails, drift). Learnable
gain g in between = Kalman filter: uses phidot in the prediction AND re-anchors to audio -> should ground phidot
WITHOUT drift. Sawtooth supervises phi. DECISIVE: does tempo-from-LATENT exp(lt) now correlate with GT?
"""
import sys, math, importlib.util, random, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT="/home/sogang/jaehoon/CHART"; sys.path.insert(0,ROOT)
v2s=importlib.util.spec_from_file_location("v2",f"{ROOT}/experiments/kvae_barpointer/faithful_v2.py")
v2=importlib.util.module_from_spec(v2s); v2s.loader.exec_module(v2)
s3=importlib.util.spec_from_file_location("sa3",f"{ROOT}/experiments/diagram_arch/sawtooth_aux3.py")
sa3=importlib.util.module_from_spec(s3); s3.loader.exec_module(sa3)
da=v2.da; BPVAE=da.BPVAE; load_pool=da.load_pool; sample_batch=da.sample_batch
phase_beats=da.phase_beats; phase_downbeats=da.phase_downbeats; fmeas=da.fmeas
soft_lt=v2.soft_lt; gt_batch_uni=sa3.gt_batch_uni
kl_von_mises=da.kl_von_mises; kl_log_normal=da.kl_log_normal; kl_categorical=da.kl_categorical
DEV=da.DEV; TWO_PI=2*math.pi; FPS=86.1328125; M=4


def blend(phi_pred,phi_data,g):                          # circular Kalman-style blend
    cb=(1-g)*torch.cos(phi_pred)+g*torch.cos(phi_data)
    sb=(1-g)*torch.sin(phi_pred)+g*torch.sin(phi_data)
    return torch.atan2(sb,cb)%TWO_PI


def rollout_filt(model,gain,h,b_in,db_in,temp=0.5,sample=True,compute_kl=True):
    B,T,_=h.shape
    pc=model.enc_post(h,b_in,db_in); pr=model.enc_prior(h) if compute_kl else None
    klm=klp=klt=(h.new_zeros(B) if compute_kl else None)
    z0=model.z0.unsqueeze(0).expand(B,-1)
    qm,qpm,qpk,qtm,qts=model.unpack(model.post_head(torch.cat([pc[:,0],z0],-1)))
    m=da.gumbel_softmax(qm,temp) if sample else F.softmax(qm,-1)
    lt=soft_lt(qtm); phi=qpm%TWO_PI
    if compute_kl:
        pm,ppm,ppk,ptm,pts=model.unpack(model.prior_init(pr.mean(1)))
        klm=klm+kl_categorical(torch.log_softmax(qm,-1),torch.log_softmax(pm,-1))
        klp=klp+kl_von_mises(phi,qpk,ppm,ppk); klt=klt+kl_log_normal(qtm,qts,ptm,pts)
    g=torch.sigmoid(gain)
    zf=[model.zfeat(m,phi,lt)]; phis=[phi]; lts=[lt]; mprev,phiprev,ltprev=m,phi,lt
    for t in range(1,T):
        zp=model.zfeat(mprev,phiprev,ltprev)
        qm,qpm,qpk,qtm,qts=model.unpack(model.post_head(torch.cat([pc[:,t],zp],-1)))
        m=da.gumbel_softmax(qm,temp) if sample else F.softmax(qm,-1)
        lt=soft_lt(qtm)
        phi_pred=(phiprev+torch.exp(ltprev))%TWO_PI       # PREDICT via phidot
        phi=blend(phi_pred,qpm%TWO_PI,g)                  # CORRECT toward audio (gain g)
        if compute_kl:
            ppm=phi_pred; ppk=F.softplus(model.prior_pk(pr[:,t]).squeeze(-1))+0.01
            ptm=ltprev; pts=F.softplus(model.prior_ts(pr[:,t]).squeeze(-1))+1e-3
            klm=klm+kl_categorical(torch.log_softmax(qm,-1),model.meter_logp(mprev,phi,phiprev,pr[:,t]))
            klp=klp+kl_von_mises(phi,qpk,ppm,ppk); klt=klt+kl_log_normal(qtm,qts,ptm,pts)
        zf.append(model.zfeat(m,phi,lt)); phis.append(phi); lts.append(lt); mprev,phiprev,ltprev=m,phi,lt
    logits=torch.stack([model.decode(zf[t]) for t in range(T)],1)
    kl=(klm,klp,klt) if compute_kl else None
    return kl,torch.stack(phis,1),torch.stack(lts,1),logits


def elbo(model,gain,h,b,db,temp,lam,fb=0.1,b_drop=0.5,pw_b=8.0,pw_db=20.0):
    B,T,_=h.shape
    keep=(torch.rand(B,1,device=h.device)>=b_drop).float()
    (klm,klp,klt),phi,lt,logits=rollout_filt(model,gain,h,b*keep,db*keep,temp,True,True)
    pw=torch.tensor([pw_b,pw_db],device=h.device)
    recon=F.binary_cross_entropy_with_logits(logits,torch.stack([b,db],-1),pos_weight=pw,reduction="none").sum((1,2))
    klm=klm.clamp(min=fb*T); klp=klp.clamp(min=fb*T); klt=klt.clamp(min=fb*T)
    phi_gt,mask=gt_batch_uni(b,db)
    Lph=(((1-torch.cos(phi-phi_gt))*mask).sum(1)/mask.sum(1).clamp(min=1.0))
    loss=(recon+klm+klp+klt+lam*T*Lph).mean()
    return loss,{"recon":float(recon.mean()),"Lph":float(Lph.mean()),"g":float(torch.sigmoid(gain))}


@torch.no_grad()
def evaluate(model,gain,val,h_mode="real",frames=1600):
    model.eval(); gb,gd=[],[]; n=len(val)
    for i,(hh,b,db) in enumerate(val):
        h_use=val[(i+1)%n][0] if h_mode=="shuffle" else hh
        T=min(h_use.shape[0],b.shape[0],frames)
        h_in=torch.zeros(1,T,hh.shape[1],device=DEV) if h_mode=="zero" else h_use[:T].unsqueeze(0).to(DEV)
        z=torch.zeros(1,T,device=DEV)
        _,phi,lt,_=rollout_filt(model,gain,h_in,z,z,sample=False,compute_kl=False)
        p=phi[0].cpu().numpy()
        ref=np.where(b.numpy()[:T]>0.5)[0]/FPS; dref=np.where(db.numpy()[:T]>0.5)[0]/FPS
        if len(ref)>=2: gb.append(fmeas(ref,phase_beats(p,M)))
        if len(dref)>=2: gd.append(fmeas(dref,phase_downbeats(p)))
    model.train(); f=lambda x:float(np.nanmean(x)) if x else float("nan")
    return f(gb),f(gd)


@torch.no_grad()
def tempo_corr(model,gain,val,frames=1600):
    model.eval(); lat,pha,gt=[],[],[]
    for hh,b,db in val:
        T=min(hh.shape[0],b.shape[0],frames); bt=np.where(b.numpy()[:T]>0.5)[0]/FPS
        if len(bt)<2: continue
        gt.append(60.0/np.median(np.diff(bt)))
        _,phi,lt,_=rollout_filt(model,gain,hh[:T].unsqueeze(0).to(DEV),torch.zeros(1,T,device=DEV),torch.zeros(1,T,device=DEV),False,False)
        lat.append(M*float(torch.exp(lt[0]).median())/TWO_PI*FPS*60)
        p=phi[0].cpu().numpy(); d=np.diff(p); adv=np.where(d<-math.pi,d+TWO_PI,d); adv=adv[adv>1e-4]
        pha.append(M*float(np.median(adv))/TWO_PI*FPS*60 if len(adv) else 0.0)
    lat=np.array(lat); pha=np.array(pha); gt=np.array(gt)
    def rr(x):
        oa=np.array([min([v*f for f in(0.5,1,2)],key=lambda z:abs(z-g)) for v,g in zip(x,gt)])
        return np.corrcoef(x,gt)[0,1],np.corrcoef(oa,gt)[0,1]
    return rr(lat),rr(pha),lat.std(),gt.std()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=1000); ap.add_argument("--lam",type=float,default=0.5)
    a=ap.parse_args()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    train=load_pool("cache/acts/bt_train_rich",400,seed=1); val=load_pool("cache/acts/bt_val_rich",40,seed=2)
    print(f"FILTER (predict+correct, learnable gain) + sawtooth | lam={a.lam}\n",flush=True)
    model=BPVAE(h_dim=512,hidden=64).to(DEV); gain=nn.Parameter(torch.tensor(0.0,device=DEV))
    opt=torch.optim.Adam(list(model.parameters())+[gain],lr=1e-3)
    for step in range(1,a.steps+1):
        temp=1.0+(0.3-1.0)*min(step/a.steps,1.0)
        h,b,db=sample_batch(train,256,16)
        loss,info=elbo(model,gain,h,b,db,temp,a.lam)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters())+[gain],5.0); opt.step()
        if step%200==0 or step==a.steps:
            gb,gd=evaluate(model,gain,val,"real")
            print(f"  step {step:4d} | recon {info['recon']:.0f} Lph {info['Lph']:.3f} gain {info['g']:.2f} | beat {gb:.3f} db {gd:.3f}",flush=True)
    gb,gd=evaluate(model,gain,val,"real"); gbs,gds=evaluate(model,gain,val,"shuffle"); gbz,gdz=evaluate(model,gain,val,"zero")
    (lr,lro),(pr_,pro),ls,gs=tempo_corr(model,gain,val)
    print(f"\n==== FILTER + SAWTOOTH (A) ====")
    print(f"  real beat {gb:.3f} db {gd:.3f} | shuf {gbs:.3f}/{gds:.3f} | zero {gbz:.3f}/{gdz:.3f} | final gain {float(torch.sigmoid(gain)):.2f}")
    print(f"  tempo from LATENT exp(lt): r raw {lr:.3f} oct {lro:.3f} std {ls:.0f} (GT {gs:.0f})  <<< phidot grounded?")
    print(f"  tempo from PHASE-advance : r raw {pr_:.3f} oct {pro:.3f}")
    print(f"  (free-phi: beat .84, latent garbage | integrator: FAILED, std0)")
    print("  WIN = beat high + LATENT tempo r high (gain<1 => phidot used & grounded, no drift)")
    print("DONE-A")


if __name__=="__main__": main()
