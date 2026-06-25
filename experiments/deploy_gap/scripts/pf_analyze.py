"""DEEP ANALYSIS of WHY the particle filter underperformed. Instruments the PF (ancestral tracking)
to test three hypotheses on the trained DBN-VAE:
  H1 (readout): the filter HOLDS a good trajectory but the circular-MEAN readout destroys it.
       -> compare F1 of: circular-mean vs MAP-particle ancestral trajectory vs tempo-oracle particle.
  H2 (degeneracy): ESS collapses after the sharp emission -> filter loses diversity, locks wrong.
       -> ESS/K over time; fraction of frames with ESS<K/10.
  H3 (tempo found): the filter DOES identify the right tempo (only phase readout fails).
       -> weighted-mean BPM vs GT over time; octave-correct rate.
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
def instrumented_pf(model, o, m, gt_bpm, K=800):
    T=o.shape[1]; oo=o[0]
    sig_t=float(F.softplus(model.log_sig_tempo)+1e-3); sig_obs=float(F.softplus(model.log_sig_obs)+1e-2)
    ke=float(F.softplus(model.kappa_em)); we=float(model.w_em); be=float(model.b_em)
    bpm0=np.random.uniform(50,210,K); lt=torch.tensor(np.log(TWO_PI*(bpm0/60)/m/FPS),device=dev,dtype=torch.float32)
    phi=torch.rand(K,device=dev)*TWO_PI
    phi_steps=np.empty((T,K),dtype=np.float32); lt_steps=np.empty((T,K),dtype=np.float32); res_idx=np.empty((T,K),dtype=np.int64)
    ess=np.empty(T); wmean_bpm=np.empty(T); cover=np.empty(T); cmean=np.empty(T)
    for t in range(T):
        if t>0:
            lt=lt+sig_t*torch.randn(K,device=dev); phi=(phi+torch.exp(lt))%TWO_PI
        mu=we*torch.exp(ke*(torch.cos(m*phi)-1.0))+be
        w=torch.softmax(-((oo[t]-mu)**2)/(2*sig_obs**2),0)
        phi_steps[t]=phi.cpu().numpy(); lt_steps[t]=lt.cpu().numpy()
        ess[t]=1.0/float((w*w).sum())
        pbpm=60.0*FPS*m*torch.exp(lt)/TWO_PI
        wmean_bpm[t]=float((w*pbpm).sum()); cover[t]=float((pbpm-gt_bpm).abs().min())
        cmean[t]=math.atan2(float((w*torch.sin(phi)).sum()),float((w*torch.cos(phi)).sum()))%TWO_PI
        if ess[t]<K/2:
            idx=torch.multinomial(w,K,replacement=True); res_idx[t]=idx.cpu().numpy(); phi=phi[idx]; lt=lt[idx]
        else:
            res_idx[t]=np.arange(K)
        if t==T-1: final_w=w.cpu().numpy()
    def trace(j):
        anc=np.empty(T,dtype=np.int64); anc[T-1]=j
        for t in range(T-2,-1,-1): anc[t]=res_idx[t][anc[t+1]]
        return np.array([phi_steps[t,anc[t]] for t in range(T)])
    map_j=int(np.argmax(final_w)); map_traj=trace(map_j)
    # tempo-oracle: particle whose median tempo is closest to GT (did a right-tempo particle survive?)
    med_bpm=60.0*FPS*m*np.exp(np.median(lt_steps,0))/TWO_PI; orc_j=int(np.argmin(np.abs(med_bpm-gt_bpm))); orc_traj=trace(orc_j)
    return dict(cmean=cmean,map_traj=map_traj,orc_traj=orc_traj,ess=ess,wmean_bpm=wmean_bpm,cover=cover,
                final_wmean_bpm=float(wmean_bpm[-200:].mean()))


def main():
    model=DBNVae().to(dev); model.load_state_dict(torch.load(CKP,map_location=dev)); model.eval(); logmel=LogMel().to(dev)
    np.random.seed(0)
    Fcm=[];Fmap=[];Forc=[];essmed=[];essfrac=[];bpmerr=[];octok=[];covok=[]
    for key,audio,b,downs,meta in iter_val_songs(ROOT,DS,max_per_dataset=3):
        T=min(len(b),800); ref=np.where(b.numpy()[:T]>0.5)[0]/FPS; df=np.where(downs.numpy()[:T]>0.5)[0]/FPS
        if len(ref)<8: continue
        m=4
        if len(df)>=2:
            bpb=np.median([np.sum((ref>=df[i])&(ref<df[i+1])) for i in range(len(df)-1)]); m=max(2,min(int(round(bpb)) if bpb>0 else 4,4))
        gt_bpm=60.0/np.median(np.diff(ref))
        o=onset_env(logmel(audio.to(dev).unsqueeze(0))[:,:T])
        r=instrumented_pf(model,o,m,gt_bpm)
        Fcm.append(f1(ref,beats(r["cmean"],m))); Fmap.append(f1(ref,beats(r["map_traj"],m))); Forc.append(f1(ref,beats(r["orc_traj"],m)))
        essmed.append(np.median(r["ess"])); essfrac.append(float(np.mean(r["ess"]<80)))   # <K/10
        be=abs(r["final_wmean_bpm"]-gt_bpm)/gt_bpm; bpmerr.append(be)
        octok.append(float(any(abs(r["final_wmean_bpm"]-gt_bpm*x)/(gt_bpm*x)<0.08 for x in (.5,1,2))))
        covok.append(float(np.median(r["cover"])<0.08*gt_bpm))
    print(f"songs={len(Fcm)}  (K=800)")
    print(f"H1 READOUT:  circular-mean F1={np.nanmean(Fcm):.3f}  |  MAP-trajectory F1={np.nanmean(Fmap):.3f}  |  tempo-oracle-particle F1={np.nanmean(Forc):.3f}")
    print(f"   -> H1 {'CONFIRMED (mean destroys a good trajectory)' if np.nanmean(max(Fmap,Forc) if False else Fmap)>np.nanmean(Fcm)+0.05 or np.nanmean(Forc)>np.nanmean(Fcm)+0.05 else 'NOT confirmed'}")
    print(f"H2 DEGENERACY: median ESS/K={np.mean(essmed)/800:.3f}  frac frames ESS<K/10={np.mean(essfrac):.3f}")
    print(f"H3 TEMPO FOUND: final weighted-mean BPM error={np.mean(bpmerr)*100:.0f}%  octave-correct={np.mean(octok):.3f}  GT-tempo-covered={np.mean(covok):.3f}")


if __name__=="__main__": main()
