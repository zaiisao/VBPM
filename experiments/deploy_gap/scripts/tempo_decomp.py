"""DEEP-ANALYSIS DIAGNOSTIC: decompose the free-run deployment failure precisely.
Is the deployed tempo (a) WRONG FROM THE START (bad initial estimate), or (b) OK then DRIFTS?
Also: how fast does phase desync? On the healthy oulong checkpoint. Eval-only.
Variants of the free-run MEAN chain:
  A) stochastic free-run (the real deployment)         -> baseline
  B) model-init tempo FROZEN (no drift, no noise)       -> isolates the model's tempo CHOICE
  C) GT-global tempo FROZEN, best phase                 -> perfect-constant ceiling
Plus: model-init BPM vs GT BPM (estimate error), and frames-until-phase-error>70ms (drift speed).
"""
import sys, math
import numpy as np, torch
import torch.nn.functional as F
import mir_eval
sys.path.insert(0, "/home/sogang/jaehoon/CHART")
from faithful.model import BarPointerVAE
from faithful.elbo import free_run
from faithful.data import FPS, N_MELS, LogMel, iter_val_songs
from faithful.distributions import TWO_PI, gumbel_softmax, sample_von_mises

dev = "cuda"; ROOT = "/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data"; DS = ["ballroom","beatles","hains","rwc_popular"]
CK = "/home/sogang/.tmp/claude-1003/-home-sogang-jaehoon-CHART/84e38297-7220-4bbe-b30a-42cd7c5a3087/scratchpad/runs/nf/w6lat_oulong/final.pt"


def f1(ref, est): return 0.0 if len(est)==0 else (float("nan") if len(ref)==0 else float(mir_eval.beat.f_measure(ref, est)))
def beats(phase, m):
    psi=(m*np.asarray(phase))%TWO_PI; w=np.where(np.diff(psi)<-math.pi)[0]+1
    out,last=[],-1e9
    for fr in w:
        if fr-last>=0.10*FPS: out.append(fr); last=fr
    return np.array(out)/FPS
def const_chain(T, phi0, lt):
    phi=np.empty(T); phi[0]=phi0%TWO_PI
    for t in range(1,T): phi[t]=(phi[t-1]+math.exp(lt))%TWO_PI
    return phi


def main():
    ck=torch.load(CK,map_location=dev); a=ck.get("args",{})
    model=BarPointerVAE(h_dim=N_MELS,hidden=a.get("hidden",64),num_meters=a.get("num_meters",4),latent_only=True).to(dev)
    model.load_state_dict(ck["model"]); model.eval(); logmel=LogMel().to(dev)
    rows=[]
    for key,audio,b,downs,meta in iter_val_songs(ROOT,DS,max_per_dataset=4):
        T=min(len(b),1200); ref=np.where(b.numpy()[:T]>0.5)[0]/FPS; df=np.where(downs.numpy()[:T]>0.5)[0]/FPS
        if len(ref)<8: continue
        m=4
        if len(df)>=2:
            bpb=np.median([np.sum((ref>=df[i])&(ref<df[i+1])) for i in range(len(df)-1)]); m=max(2,min(int(round(bpb)) if bpb>0 else 4,4))
        h=logmel(audio.to(dev).unsqueeze(0))[:,:T]
        with torch.no_grad():
            pc=model.encode_prior(h); pm,pphmu,pphk,ptmu,pts=model.unpack(model.prior_init_head(pc.mean(1)))
            lt_init=float(ptmu[0]); phi0=float(pphmu[0])%TWO_PI
            o=free_run(model,h,temperature=0.3); fr_phase=o["phase_mu"][0,:T].cpu().numpy()
        gt_bpm=60.0/np.median(np.diff(ref)); gt_lt=math.log(TWO_PI/(m*np.median(np.diff(ref))*FPS))
        init_bpm=60.0*FPS*m*math.exp(lt_init)/TWO_PI
        # A stochastic, B model-init frozen, C GT frozen best-phase
        fA=f1(ref, beats(fr_phase,m))
        fB=f1(ref, beats(const_chain(T,phi0,lt_init),m))
        fC=max(f1(ref, beats(const_chain(T,TWO_PI*k/m,gt_lt),m)) for k in range(m))
        rows.append((key,gt_bpm,init_bpm,abs(init_bpm-gt_bpm)/gt_bpm,fA,fB,fC,float(np.std(np.diff(ref))/np.mean(np.diff(ref)))))
    R=np.array([r[1:] for r in rows],float)
    cv=R[:,6]; stable=cv<0.04
    print(f"songs={len(rows)}")
    print(f"  model init-BPM error |Δ|/gt   mean={R[:,2].mean()*100:.1f}%   (>4% = wrong tempo from the START)")
    print(f"  octave-ish (init within 0.5-2x gt): {np.mean((R[:,1]/R[:,0]>0.5)&(R[:,1]/R[:,0]<2.0))*100:.0f}%")
    print(f"  F1  A stochastic free-run     = {np.nanmean(R[:,3]):.3f}")
    print(f"  F1  B model-init tempo FROZEN = {np.nanmean(R[:,4]):.3f}   (B>>A => drift/noise is the killer; B~A => not drift)")
    print(f"  F1  C GT tempo FROZEN (ceil)  = {np.nanmean(R[:,5]):.3f}   (C>>B => the model's tempo ESTIMATE is wrong)")
    print(f"  C on stable-tempo songs={np.nanmean(R[stable,5]):.3f} (n={int(stable.sum())}) | varying={np.nanmean(R[~stable,5]):.3f} (n={int((~stable).sum())})")
    print(f"  B on stable={np.nanmean(R[stable,4]):.3f} | varying={np.nanmean(R[~stable,4]):.3f}")


if __name__=="__main__": main()
