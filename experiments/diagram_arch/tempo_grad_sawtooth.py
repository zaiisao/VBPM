"""GRADIENT TEST (again), now on the sawtooth-aux model. Mechanism claim: the aux grounds the phase qpm
-> the prior coupling ppm=phiprev+exp(ltprev) then makes KL_phase deliver a STRONG, correctly-DIRECTED
gradient into the tempo mean qtm. Compare NO-AUX (lam=0) vs AUX (lam=0.5): if aux flips the tempo
gradient from weak/misdirected to strong/aligned, that IS the breakthrough mechanism, proven at the gradient.
Measures (w.r.t. log-tempo means qtm_t): grad-norm of recon / KL_phase / KL_tempo, and DIRECTION
correctness (does -grad_total move tempo toward GT?).
"""
import sys, math, importlib.util, random
import numpy as np, torch, torch.nn.functional as F
ROOT="/home/sogang/jaehoon/CHART"; sys.path.insert(0,ROOT)
s=importlib.util.spec_from_file_location("sa",f"{ROOT}/experiments/diagram_arch/sawtooth_aux.py")
sa=importlib.util.module_from_spec(s); s.loader.exec_module(sa)
da=sa.da; BPVAE=da.BPVAE; load_pool=da.load_pool; sample_batch=da.sample_batch
kl_von_mises=da.kl_von_mises; kl_log_normal=da.kl_log_normal; kl_categorical=da.kl_categorical
DEV=da.DEV; TWO_PI=2*math.pi; FPS=86.1328125; M=4


def capture(model, h):
    """Deterministic (means) rollout matching run.py; returns qpm list, qtm list, klp, klt (per-frame summed)."""
    B,T,_=h.shape
    pc=model.enc_post(h, torch.zeros(B,T,device=h.device), torch.zeros(B,T,device=h.device))
    pr=model.enc_prior(h)
    z0=model.z0.unsqueeze(0).expand(B,-1)
    qm,qpm,qpk,qtm,qts=model.unpack(model.post_head(torch.cat([pc[:,0],z0],-1)))
    m=F.softmax(qm,-1); phi=qpm; lt=qtm
    pm,ppm,ppk,ptm,pts=model.unpack(model.prior_init(pr.mean(1)))
    klp=kl_von_mises(qpm,qpk,ppm,ppk); klt=kl_log_normal(qtm,qts,ptm,pts)
    qpms=[qpm]; qtms=[qtm]; zf=[model.zfeat(m,phi,lt)]; mprev,phiprev,ltprev=m,phi,lt
    for t in range(1,T):
        zp=model.zfeat(mprev,phiprev,ltprev)
        qm,qpm,qpk,qtm,qts=model.unpack(model.post_head(torch.cat([pc[:,t],zp],-1)))
        m=F.softmax(qm,-1); phi=qpm; lt=qtm
        ppm=(phiprev+torch.exp(ltprev))%TWO_PI
        ppk=F.softplus(model.prior_pk(pr[:,t]).squeeze(-1))+0.01
        ptm=ltprev; pts=F.softplus(model.prior_ts(pr[:,t]).squeeze(-1))+1e-3
        klp=klp+kl_von_mises(qpm,qpk,ppm,ppk); klt=klt+kl_log_normal(qtm,qts,ptm,pts)
        qpms.append(qpm); qtms.append(qtm); zf.append(model.zfeat(m,phi,lt)); mprev,phiprev,ltprev=m,phi,lt
    logits=torch.stack([model.decode(zf[t]) for t in range(T)],1)
    return qpms,qtms,klp,klt,logits


def gnorm(loss,qtms):
    g=torch.autograd.grad(loss,qtms,retain_graph=True,allow_unused=True)
    g=[x if x is not None else torch.zeros_like(qtms[0]) for x in g]
    return torch.stack(g)  # [T,B]


def run(lam, train, val, steps=500):
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    model=BPVAE(h_dim=512,hidden=64).to(DEV); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for step in range(1,steps+1):
        temp=1.0+(0.3-1.0)*min(step/steps,1.0)
        h,b,db=sample_batch(train,256,16)
        loss,_=sa.elbo_aux(model,h,b,db,temp,lam)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
    model.train()
    hb=[v for v in val if v[0].shape[0]>=256][:10]
    h=torch.stack([v[0][:256] for v in hb]).to(DEV); b=torch.stack([v[1][:256] for v in hb]).to(DEV); db=torch.stack([v[2][:256] for v in hb]).to(DEV)
    qpms,qtms,klp,klt,logits=capture(model,h)
    pw=torch.tensor([8.0,20.0],device=DEV)
    recon=F.binary_cross_entropy_with_logits(logits,torch.stack([b,db],-1),pos_weight=pw,reduction="none").sum((1,2)).mean()
    g_r=gnorm(recon,qtms); g_p=gnorm(klp.mean(),qtms); g_t=gnorm(klt.mean(),qtms)
    g_total=g_r+g_p+g_t
    with torch.no_grad():
        cur=(M*torch.exp(torch.stack(qtms))/TWO_PI*FPS*60).mean(0)  # [B] per-song mean tempo
    gt=[]
    for v in hb:
        bf=np.where(v[1][:256].numpy()>0.5)[0]; gt.append(60*FPS/np.median(np.diff(bf)) if len(bf)>2 else np.nan)
    gt=torch.tensor(gt,device=DEV,dtype=cur.dtype); err=gt-cur; valid=~torch.isnan(err)
    upd=-g_total.mean(0)  # [B] direction the gradient moves each song's tempo
    aligned=(torch.sign(upd[valid])==torch.sign(err[valid])).float().mean()
    print(f"  [lam={lam}] grad-norm into log-tempo:  recon {float(g_r.norm()):.2f} | KL_phase {float(g_p.norm()):.2f} | KL_tempo {float(g_t.norm()):.2f}")
    print(f"  [lam={lam}] DIRECTION correctness (-grad_total -> GT): {float(aligned)*100:.0f}%  | cur {float(cur[valid].mean()):.0f} vs GT {float(gt[valid].mean()):.0f} BPM")
    return float(g_p.norm()), float(aligned)


def main():
    train=load_pool("cache/acts/bt_train_rich",300,seed=1); val=load_pool("cache/acts/bt_val_rich",16,seed=2)
    print("GRADIENT TEST: does the sawtooth aux make KL_phase deliver a strong/aligned tempo gradient?\n",flush=True)
    print("NO-AUX (lam=0):"); n=run(0.0,train,val)
    print("\nAUX (lam=0.5):"); a=run(0.5,train,val)
    print(f"\n==== {('KL_phase tempo-grad %.2f->%.2f, direction %.0f%%->%.0f%%'%(n[0],a[0],n[1]*100,a[1]*100))} ====")
    print("  aux RAISES KL_phase->tempo grad norm AND aligns direction >50% => mechanism proven")
    print("DONE")


if __name__=="__main__": main()
