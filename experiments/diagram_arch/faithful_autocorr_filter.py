"""SYNTHESIS (A+B in the faithful model): phidot's posterior mean = AUTOCORRELATION head (data-informed,
computed from h) -- NOT free-learned. phi = FILTER(predict via phidot, correct toward data qpm, gain g).
Sawtooth supervises phi. Tempo head trained by its own CE (vs GT period). Losses AGREE (tempo grounds
phidot; sawtooth grounds phi; consistent because phi is built FROM phidot). ELBO (KL meter+phase) intact.
This is the user's design: deterministic data-informed mean for phidot, std/coupling learnable.
Reports beat/db, leak, and tempo-from-phidot vs tempo-from-phase vs GT (should all agree + match GT).
"""
import sys, math, importlib.util, random, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT="/home/sogang/jaehoon/CHART"; sys.path.insert(0,ROOT)
s=importlib.util.spec_from_file_location("da",f"{ROOT}/experiments/diagram_arch/run.py")
da=importlib.util.module_from_spec(s); s.loader.exec_module(da)
bm=importlib.util.spec_from_file_location("B",f"{ROOT}/experiments/diagram_arch/autocorr_tempo.py")
B=importlib.util.module_from_spec(bm); bm.loader.exec_module(B)
s3=importlib.util.spec_from_file_location("sa3",f"{ROOT}/experiments/diagram_arch/sawtooth_aux3.py")
sa3=importlib.util.module_from_spec(s3); s3.loader.exec_module(sa3)
BPVAE=da.BPVAE; load_pool=da.load_pool; sample_batch=da.sample_batch
phase_beats=da.phase_beats; phase_downbeats=da.phase_downbeats; fmeas=da.fmeas
kl_von_mises=da.kl_von_mises; kl_categorical=da.kl_categorical; gt_batch_uni=sa3.gt_batch_uni
TempoNet=B.TempoNet; LAGS=B.LAGS
DEV=da.DEV; TWO_PI=2*math.pi; FPS=86.1328125; M=4


def blend(phi_pred,phi_data,g):
    cb=(1-g)*torch.cos(phi_pred)+g*torch.cos(phi_data)
    sb=(1-g)*torch.sin(phi_pred)+g*torch.sin(phi_data)
    return torch.atan2(sb,cb)%TWO_PI


def clip_tempo(tnet,h):
    """per-clip bar-phase advance phidot from autocorr head (argmax lag, detached for the rollout)."""
    lg=tnet(h)                                            # [B,nlag]
    idx=lg.argmax(1)                                      # [B]
    period=torch.tensor(LAGS,device=h.device,dtype=h.dtype)[idx]  # beat period frames
    phidot=TWO_PI/(M*period)                              # bar-phase advance per frame [B]
    return phidot, lg


def rollout(model,tnet,gain,h,b_in,db_in,temp=0.5,sample=True,compute_kl=True):
    Bs,T,_=h.shape
    pc=model.enc_post(h,b_in,db_in); pr=model.enc_prior(h) if compute_kl else None
    klm=klp=(h.new_zeros(Bs) if compute_kl else None)
    phidot,lg=clip_tempo(tnet,h); phidot=phidot.detach()  # phidot from autocorr head (computed mean)
    z0=model.z0.unsqueeze(0).expand(Bs,-1)
    qm,qpm,qpk,qtm,qts=model.unpack(model.post_head(torch.cat([pc[:,0],z0],-1)))
    m=da.gumbel_softmax(qm,temp) if sample else F.softmax(qm,-1); phi=qpm%TWO_PI
    if compute_kl:
        pm,ppm,ppk,ptm,pts=model.unpack(model.prior_init(pr.mean(1)))
        klm=klm+kl_categorical(torch.log_softmax(qm,-1),torch.log_softmax(pm,-1))
        klp=klp+kl_von_mises(phi,qpk,ppm,ppk)
    g=torch.sigmoid(gain)
    ltc=torch.log(phidot)                                  # log bar-phase advance (for zfeat)
    zf=[model.zfeat(m,phi,ltc)]; phis=[phi]; mprev,phiprev=m,phi
    for t in range(1,T):
        zp=model.zfeat(mprev,phiprev,ltc)
        qm,qpm,qpk,qtm,qts=model.unpack(model.post_head(torch.cat([pc[:,t],zp],-1)))
        m=da.gumbel_softmax(qm,temp) if sample else F.softmax(qm,-1)
        phi_pred=(phiprev+phidot)%TWO_PI                   # PREDICT via autocorr phidot
        phi=blend(phi_pred,qpm%TWO_PI,g)                   # CORRECT toward data
        if compute_kl:
            ppk=F.softplus(model.prior_pk(pr[:,t]).squeeze(-1))+0.01
            klm=klm+kl_categorical(torch.log_softmax(qm,-1),model.meter_logp(mprev,phi,phiprev,pr[:,t]))
            klp=klp+kl_von_mises(phi,qpk,phi_pred,ppk)
        zf.append(model.zfeat(m,phi,ltc)); phis.append(phi); mprev,phiprev=m,phi
    logits=torch.stack([model.decode(zf[t]) for t in range(T)],1)
    kl=(klm,klp) if compute_kl else None
    return kl,torch.stack(phis,1),phidot,lg,logits


def elbo(model,tnet,gain,h,b,db,temp,lam_saw=0.5,lam_t=1.0,fb=0.1,b_drop=0.5,pw_b=8.0,pw_db=20.0):
    Bs,T,_=h.shape
    keep=(torch.rand(Bs,1,device=h.device)>=b_drop).float()
    (klm,klp),phi,phidot,lg,logits=rollout(model,tnet,gain,h,b*keep,db*keep,temp,True,True)
    pw=torch.tensor([pw_b,pw_db],device=h.device)
    recon=F.binary_cross_entropy_with_logits(logits,torch.stack([b,db],-1),pos_weight=pw,reduction="none").sum((1,2))
    klm=klm.clamp(min=fb*T); klp=klp.clamp(min=fb*T)
    phi_gt,mask=gt_batch_uni(b,db)
    Lsaw=(((1-torch.cos(phi-phi_gt))*mask).sum(1)/mask.sum(1).clamp(min=1.0))
    # tempo CE on autocorr head vs GT period
    tgt=[]; valid=[]; bn=b.cpu().numpy()
    for j in range(bn.shape[0]):
        p=B.gt_period_frames(bn[j])
        if np.isnan(p): tgt.append(0); valid.append(0.0)
        else: tgt.append(int(np.argmin(np.abs(LAGS-p)))); valid.append(1.0)
    tgt=torch.tensor(tgt,device=h.device); valid=torch.tensor(valid,device=h.device)
    Lt=(F.cross_entropy(lg,tgt,reduction="none")*valid).sum()/valid.sum().clamp(min=1)
    loss=(recon+klm+klp+lam_saw*T*Lsaw).mean()+lam_t*Lt
    return loss,{"recon":float(recon.mean()),"Lsaw":float(Lsaw.mean()),"Lt":float(Lt),"g":float(torch.sigmoid(gain))}


@torch.no_grad()
def evaluate(model,tnet,gain,val,h_mode="real",frames=1600):
    model.eval(); tnet.eval(); gb,gd=[],[]; n=len(val)
    for i,(hh,b,db) in enumerate(val):
        h_use=val[(i+1)%n][0] if h_mode=="shuffle" else hh
        T=min(h_use.shape[0],b.shape[0],frames)
        h_in=torch.zeros(1,T,hh.shape[1],device=DEV) if h_mode=="zero" else h_use[:T].unsqueeze(0).to(DEV)
        z=torch.zeros(1,T,device=DEV)
        _,phi,_,_,_=rollout(model,tnet,gain,h_in,z,z,sample=False,compute_kl=False)
        p=phi[0].cpu().numpy()
        ref=np.where(b.numpy()[:T]>0.5)[0]/FPS; dref=np.where(db.numpy()[:T]>0.5)[0]/FPS
        if len(ref)>=2: gb.append(fmeas(ref,phase_beats(p,M)))
        if len(dref)>=2: gd.append(fmeas(dref,phase_downbeats(p)))
    model.train(); tnet.train(); f=lambda x:float(np.nanmean(x)) if x else float("nan")
    return f(gb),f(gd)


@torch.no_grad()
def tempo_corr(model,tnet,gain,val,frames=1600):
    model.eval(); tnet.eval(); lat,pha,gt=[],[],[]
    for hh,b,db in val:
        T=min(hh.shape[0],b.shape[0],frames); bt=np.where(b.numpy()[:T]>0.5)[0]/FPS
        if len(bt)<2: continue
        gt.append(60.0/np.median(np.diff(bt)))
        _,phi,phidot,_,_=rollout(model,tnet,gain,hh[:T].unsqueeze(0).to(DEV),torch.zeros(1,T,device=DEV),torch.zeros(1,T,device=DEV),False,False)
        lat.append(M*float(phidot[0])/TWO_PI*FPS*60)
        p=phi[0].cpu().numpy(); d=np.diff(p); adv=np.where(d<-math.pi,d+TWO_PI,d); adv=adv[adv>1e-4]
        pha.append(M*float(np.median(adv))/TWO_PI*FPS*60 if len(adv) else 0.0)
    lat=np.array(lat); pha=np.array(pha); gt=np.array(gt)
    def rr(x):
        oa=np.array([min([v*f for f in(0.5,1,2)],key=lambda z:abs(z-g)) for v,g in zip(x,gt)])
        return np.corrcoef(x,gt)[0,1],np.corrcoef(oa,gt)[0,1]
    return rr(lat),rr(pha),lat.std(),gt.std()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=1500); ap.add_argument("--lam_saw",type=float,default=0.5)
    a=ap.parse_args()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    train=load_pool("cache/acts/bt_train_rich",400,seed=1); val=load_pool("cache/acts/bt_val_rich",40,seed=2)
    print(f"FAITHFUL + AUTOCORR phidot + FILTER + sawtooth | lam_saw={a.lam_saw}\n",flush=True)
    model=BPVAE(h_dim=512,hidden=64).to(DEV); tnet=TempoNet(512).to(DEV); gain=nn.Parameter(torch.tensor(0.0,device=DEV))
    opt=torch.optim.Adam(list(model.parameters())+list(tnet.parameters())+[gain],lr=1e-3)
    for step in range(1,a.steps+1):
        temp=1.0+(0.3-1.0)*min(step/a.steps,1.0)
        h,b,db=sample_batch(train,256,16)
        loss,info=elbo(model,tnet,gain,h,b,db,temp,a.lam_saw)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(tnet.parameters())+[gain],5.0); opt.step()
        if step%200==0 or step==a.steps:
            gb,gd=evaluate(model,tnet,gain,val,"real")
            print(f"  step {step:4d} | recon {info['recon']:.0f} Lsaw {info['Lsaw']:.3f} Lt {info['Lt']:.3f} gain {info['g']:.2f} | beat {gb:.3f} db {gd:.3f}",flush=True)
    gb,gd=evaluate(model,tnet,gain,val,"real"); gbs,gds=evaluate(model,tnet,gain,val,"shuffle"); gbz,gdz=evaluate(model,tnet,gain,val,"zero")
    (lr,lro),(pr_,pro),ls,gs=tempo_corr(model,tnet,gain,val)
    print(f"\n==== FAITHFUL + AUTOCORR phidot + FILTER (A+B) ====")
    print(f"  real beat {gb:.3f} db {gd:.3f} | shuf {gbs:.3f}/{gds:.3f} | zero {gbz:.3f}/{gdz:.3f} | gain {float(torch.sigmoid(gain)):.2f}")
    print(f"  tempo from phidot(autocorr): r raw {lr:.3f} oct {lro:.3f} std {ls:.0f} (GT {gs:.0f})")
    print(f"  tempo from PHASE-advance   : r raw {pr_:.3f} oct {pro:.3f}  (should AGREE with phidot)")
    print("  WIN = beat/db high + leak collapse + phidot~phase~GT => faithful filter w/ computed phidot WORKS")
    print("DONE-AB")


if __name__=="__main__": main()
