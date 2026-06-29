"""FAITHFUL step back + keep the gain: INTEGRATOR posterior (phi = integral of phidot, the derived ELBO
dynamics) + SAWTOOTH phase supervision. Because phi=cumsum(exp(lt)), supervising phi to the unified
sawtooth pushes its SLOPE = phidot toward the true tempo => grounds the TEMPO LATENT itself (not bypassed).
DECISIVE test: does tempo read from the LATENT exp(lt) now correlate with GT (vs free-phi version where
latent gave 2184 BPM garbage)? Also report tempo from phase-advance (should now AGREE with latent),
beat/db geometric, and leak.
"""
import sys, math, importlib.util, random, argparse
import numpy as np, torch, torch.nn.functional as F
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


def rollout_int(model,h,b_in,db_in,temp=0.5,sample=True,compute_kl=True):
    """Integrator: phi_t = phi_{t-1} + exp(lt_{t-1}); returns kl, phi[B,T], lt[B,T], logits."""
    B,T,_=h.shape
    pc=model.enc_post(h,b_in,db_in); pr=model.enc_prior(h) if compute_kl else None
    klm=klp=klt=(h.new_zeros(B) if compute_kl else None)
    z0=model.z0.unsqueeze(0).expand(B,-1)
    qm,qpm,qpk,qtm,qts=model.unpack(model.post_head(torch.cat([pc[:,0],z0],-1)))
    m=da.gumbel_softmax(qm,temp) if sample else F.softmax(qm,-1)
    lt=soft_lt(qtm); phi=qpm%TWO_PI
    if sample: phi=da.sample_von_mises(phi,qpk)%TWO_PI
    if compute_kl:
        pm,ppm,ppk,ptm,pts=model.unpack(model.prior_init(pr.mean(1)))
        klm=klm+kl_categorical(torch.log_softmax(qm,-1),torch.log_softmax(pm,-1))
        klp=klp+kl_von_mises(phi,qpk,ppm,ppk); klt=klt+kl_log_normal(qtm,qts,ptm,pts)
    zf=[model.zfeat(m,phi,lt)]; phis=[phi]; lts=[lt]; mprev,phiprev,ltprev=m,phi,lt
    for t in range(1,T):
        zp=model.zfeat(mprev,phiprev,ltprev)
        qm,qpm,qpk,qtm,qts=model.unpack(model.post_head(torch.cat([pc[:,t],zp],-1)))
        m=da.gumbel_softmax(qm,temp) if sample else F.softmax(qm,-1)
        lt=soft_lt(qtm)
        phi_mean=(phiprev+torch.exp(ltprev))%TWO_PI            # INTEGRATOR (phi <- phidot)
        phi=da.sample_von_mises(phi_mean,qpk)%TWO_PI if sample else phi_mean
        if compute_kl:
            ppm=(phiprev+torch.exp(ltprev))%TWO_PI
            ppk=F.softplus(model.prior_pk(pr[:,t]).squeeze(-1))+0.01
            ptm=ltprev; pts=F.softplus(model.prior_ts(pr[:,t]).squeeze(-1))+1e-3
            klm=klm+kl_categorical(torch.log_softmax(qm,-1),model.meter_logp(mprev,phi,phiprev,pr[:,t]))
            klp=klp+kl_von_mises(phi_mean,qpk,ppm,ppk); klt=klt+kl_log_normal(qtm,qts,ptm,pts)
        zf.append(model.zfeat(m,phi,lt)); phis.append(phi); lts.append(lt); mprev,phiprev,ltprev=m,phi,lt
    logits=torch.stack([model.decode(zf[t]) for t in range(T)],1)
    kl=(klm,klp,klt) if compute_kl else None
    return kl,torch.stack(phis,1),torch.stack(lts,1),logits


def elbo(model,h,b,db,temp,lam,fb=0.1,b_drop=0.5,pw_b=8.0,pw_db=20.0):
    B,T,_=h.shape
    keep=(torch.rand(B,1,device=h.device)>=b_drop).float()
    (klm,klp,klt),phi,lt,logits=rollout_int(model,h,b*keep,db*keep,temp,sample=True,compute_kl=True)
    pw=torch.tensor([pw_b,pw_db],device=h.device)
    recon=F.binary_cross_entropy_with_logits(logits,torch.stack([b,db],-1),pos_weight=pw,reduction="none").sum((1,2))
    klm=klm.clamp(min=fb*T); klp=klp.clamp(min=fb*T); klt=klt.clamp(min=fb*T)
    phi_gt,mask=gt_batch_uni(b,db)
    Lph=(((1-torch.cos(phi-phi_gt))*mask).sum(1)/mask.sum(1).clamp(min=1.0))
    loss=(recon+klm+klp+klt+lam*T*Lph).mean()
    return loss,{"recon":float(recon.mean()),"Lph":float(Lph.mean()),"klt":float(klt.mean())}


@torch.no_grad()
def evaluate(model,val,h_mode="real",frames=1600):
    model.eval(); gb,gd=[],[]; n=len(val)
    for i,(hh,b,db) in enumerate(val):
        h_use=val[(i+1)%n][0] if h_mode=="shuffle" else hh
        T=min(h_use.shape[0],b.shape[0],frames)
        h_in=torch.zeros(1,T,hh.shape[1],device=DEV) if h_mode=="zero" else h_use[:T].unsqueeze(0).to(DEV)
        z=torch.zeros(1,T,device=DEV)
        _,phi,lt,_=rollout_int(model,h_in,z,z,sample=False,compute_kl=False)
        p=phi[0].cpu().numpy()
        ref=np.where(b.numpy()[:T]>0.5)[0]/FPS; dref=np.where(db.numpy()[:T]>0.5)[0]/FPS
        if len(ref)>=2: gb.append(fmeas(ref,phase_beats(p,M)))
        if len(dref)>=2: gd.append(fmeas(dref,phase_downbeats(p)))
    model.train(); f=lambda x:float(np.nanmean(x)) if x else float("nan")
    return f(gb),f(gd)


@torch.no_grad()
def tempo_corr(model,val,frames=1600):
    model.eval(); lat,pha,gt=[],[],[]
    for hh,b,db in val:
        T=min(hh.shape[0],b.shape[0],frames); bt=np.where(b.numpy()[:T]>0.5)[0]/FPS
        if len(bt)<2: continue
        gt.append(60.0/np.median(np.diff(bt)))
        _,phi,lt,_=rollout_int(model,hh[:T].unsqueeze(0).to(DEV),torch.zeros(1,T,device=DEV),torch.zeros(1,T,device=DEV),sample=False,compute_kl=False)
        latbpm=M*float(torch.exp(lt[0]).median())/TWO_PI*FPS*60       # tempo from LATENT exp(lt)
        p=phi[0].cpu().numpy(); d=np.diff(p); adv=np.where(d<-math.pi,d+TWO_PI,d); adv=adv[adv>1e-4]
        phabpm=M*float(np.median(adv))/TWO_PI*FPS*60 if len(adv) else 0.0  # tempo from phase-advance
        lat.append(latbpm); pha.append(phabpm)
    model.train(); lat=np.array(lat); pha=np.array(pha); gt=np.array(gt)
    def rr(x):
        oa=np.array([min([v*f for f in(0.5,1,2)],key=lambda z:abs(z-g)) for v,g in zip(x,gt)])
        return np.corrcoef(x,gt)[0,1], np.corrcoef(oa,gt)[0,1]
    return rr(lat),rr(pha),lat.std(),pha.std(),gt.std()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=1000); ap.add_argument("--lam",type=float,default=0.5)
    a=ap.parse_args()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    train=load_pool("cache/acts/bt_train_rich",400,seed=1); val=load_pool("cache/acts/bt_val_rich",40,seed=2)
    print(f"FAITHFUL integrator + sawtooth | lam={a.lam} | train={len(train)} val={len(val)}\n",flush=True)
    model=BPVAE(h_dim=512,hidden=64).to(DEV); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for step in range(1,a.steps+1):
        temp=1.0+(0.3-1.0)*min(step/a.steps,1.0)
        h,b,db=sample_batch(train,256,16)
        loss,info=elbo(model,h,b,db,temp,a.lam)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        if step%200==0 or step==a.steps:
            gb,gd=evaluate(model,val,"real")
            print(f"  step {step:4d} | recon {info['recon']:.0f} Lph {info['Lph']:.3f} klt {info['klt']:.1f} | beat {gb:.3f} db {gd:.3f}",flush=True)
    gb,gd=evaluate(model,val,"real"); gbs,gds=evaluate(model,val,"shuffle"); gbz,gdz=evaluate(model,val,"zero")
    (lr,lro),(pr,pro),ls,ps,gs=tempo_corr(model,val)
    print(f"\n==== FAITHFUL INTEGRATOR + SAWTOOTH ====")
    print(f"  real beat {gb:.3f} db {gd:.3f} | shuf {gbs:.3f}/{gds:.3f} | zero {gbz:.3f}/{gdz:.3f}")
    print(f"  tempo from LATENT exp(lt): r raw {lr:.3f} oct {lro:.3f}  std {ls:.0f} (GT std {gs:.0f})  <<< is phidot GROUNDED?")
    print(f"  tempo from PHASE-advance : r raw {pr:.3f} oct {pro:.3f}  std {ps:.0f}")
    print(f"  (free-phi baseline: latent gave 2184 BPM garbage / 0% direction; phase gave r .62/.99)")
    print("  WIN = LATENT tempo r high + agrees with phase tempo => phidot grounded, faithful dynamics used")
    print("DONE")


if __name__=="__main__": main()
