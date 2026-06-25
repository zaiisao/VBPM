"""CORRECTED SMC (referencing official AESMC tuananhle7/aesmc inference.py): per-step weight =
emission log-prob (bootstrap proposal = prior), but ACCUMULATED across frames with adaptive
resampling (resample on ESS<K/2 using the accumulated weight, reset after) -- the standard SMC
recursion my buggy per-frame-softmax version skipped. Re-instrument to VERIFY the fix:
  does ESS now DROP (filter actually reweights/resamples)? does weighted-mean BPM converge to GT?
  F1 under circular-mean and MAP-trajectory readouts.
"""
import sys, math
import numpy as np, torch
import torch.nn.functional as F
import mir_eval
sys.path.insert(0, "/home/sogang/jaehoon/CHART")
sys.path.insert(0, "/home/sogang/.tmp/claude-1003/-home-sogang-jaehoon-CHART/84e38297-7220-4bbe-b30a-42cd7c5a3087/scratchpad")
from dbn_vae import DBNVae, onset_env, beats, f1
from faithful.data import FPS, N_MELS, LogMel, iter_val_songs
from faithful.distributions import TWO_PI

dev="cuda"; ROOT="/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data"; DS=["ballroom","beatles","hains","rwc_popular"]
CKP="/home/sogang/.tmp/claude-1003/-home-sogang-jaehoon-CHART/84e38297-7220-4bbe-b30a-42cd7c5a3087/scratchpad/dbn_vae.pt"


@torch.no_grad()
def smc(model, o, m, gt_bpm, K=800):
    T=o.shape[1]; oo=o[0]
    sig_t=float(F.softplus(model.log_sig_tempo)+1e-3); sig_obs=float(F.softplus(model.log_sig_obs)+1e-2)
    ke=float(F.softplus(model.kappa_em)); we=float(model.w_em); be=float(model.b_em)
    bpm0=np.random.uniform(50,210,K); lt=torch.tensor(np.log(TWO_PI*(bpm0/60)/m/FPS),device=dev,dtype=torch.float32)
    phi=torch.rand(K,device=dev)*TWO_PI
    logw=torch.zeros(K,device=dev)                                   # ACCUMULATED log-weight (the fix)
    phi_steps=np.empty((T,K),dtype=np.float32); res_idx=np.empty((T,K),dtype=np.int64)
    ess=np.empty(T); wmean_bpm=np.empty(T); cmean=np.empty(T); n_resample=0
    for t in range(T):
        if t>0:
            lt=lt+sig_t*torch.randn(K,device=dev); phi=(phi+torch.exp(lt))%TWO_PI
        mu=we*torch.exp(ke*(torch.cos(m*phi)-1.0))+be
        logw=logw+(-((oo[t]-mu)**2)/(2*sig_obs**2))                  # ACCUMULATE emission log-prob
        w=torch.softmax(logw,0)
        phi_steps[t]=phi.cpu().numpy()
        ess[t]=1.0/float((w*w).sum())
        pbpm=60.0*FPS*m*torch.exp(lt)/TWO_PI
        wmean_bpm[t]=float((w*pbpm).sum())
        cmean[t]=math.atan2(float((w*torch.sin(phi)).sum()),float((w*torch.cos(phi)).sum()))%TWO_PI
        if ess[t]<K/2:                                               # adaptive resample on accumulated weight
            idx=torch.multinomial(w,K,replacement=True); res_idx[t]=idx.cpu().numpy()
            phi=phi[idx]; lt=lt[idx]; logw=torch.zeros(K,device=dev); n_resample+=1   # RESET after resample
        else:
            res_idx[t]=np.arange(K)
        if t==T-1: final_w=torch.softmax(logw,0).cpu().numpy()
    def trace(j):
        anc=np.empty(T,dtype=np.int64); anc[T-1]=j
        for t in range(T-2,-1,-1): anc[t]=res_idx[t][anc[t+1]]
        return np.array([phi_steps[t,anc[t]] for t in range(T)])
    map_traj=trace(int(np.argmax(final_w)))
    return dict(cmean=cmean,map_traj=map_traj,ess=ess,wmean_bpm=wmean_bpm,n_resample=n_resample,
                final_bpm=float(np.median(wmean_bpm[-200:])))


def main():
    model=DBNVae().to(dev); model.load_state_dict(torch.load(CKP,map_location=dev)); model.eval(); logmel=LogMel().to(dev)
    np.random.seed(0)
    Fcm=[];Fmap=[];essmed=[];nres=[];bpmerr=[];octok=[]
    for key,audio,b,downs,meta in iter_val_songs(ROOT,DS,max_per_dataset=4):
        T=min(len(b),1000); ref=np.where(b.numpy()[:T]>0.5)[0]/FPS; df=np.where(downs.numpy()[:T]>0.5)[0]/FPS
        if len(ref)<8: continue
        m=4
        if len(df)>=2:
            bpb=np.median([np.sum((ref>=df[i])&(ref<df[i+1])) for i in range(len(df)-1)]); m=max(2,min(int(round(bpb)) if bpb>0 else 4,4))
        gt_bpm=60.0/np.median(np.diff(ref))
        o=onset_env(logmel(audio.to(dev).unsqueeze(0))[:,:T])
        r=smc(model,o,m,gt_bpm,K=800)
        Fcm.append(f1(ref,beats(r["cmean"],m))); Fmap.append(f1(ref,beats(r["map_traj"],m)))
        essmed.append(np.median(r["ess"])/800); nres.append(r["n_resample"])
        be=abs(r["final_bpm"]-gt_bpm)/gt_bpm; bpmerr.append(be)
        octok.append(float(any(abs(r["final_bpm"]-gt_bpm*x)/(gt_bpm*x)<0.08 for x in (.5,1,2))))
    print(f"songs={len(Fcm)}  CORRECTED SMC (accumulated weights + adaptive resample, AESMC-style)")
    print(f"  ESS check: median ESS/K={np.mean(essmed):.3f} (was 0.999=never filtered)  resamples/song={np.mean(nres):.0f}")
    print(f"  TEMPO: final weighted-mean BPM err={np.mean(bpmerr)*100:.0f}% (was 54%)  octave-correct={np.mean(octok):.3f} (was 0.25)")
    print(f"  F1: circular-mean={np.nanmean(Fcm):.3f}  MAP-trajectory={np.nanmean(Fmap):.3f}")
    print(f"  refs: VAE free-run 0.40 | broken PF 0.335 | classic tempo+phase 0.66")


if __name__=="__main__": main()
