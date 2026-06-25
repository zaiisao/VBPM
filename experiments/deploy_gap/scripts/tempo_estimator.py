"""SIMPLE TEMPO ESTIMATOR experiment: does supplying a tempo fix the deployment gap?
Deep-dive said: the prior can't estimate tempo from audio (852% err); GT-tempo-frozen = 0.51 vs model 0.36.
Test: a CLASSIC autocorrelation tempo estimator (onset envelope -> autocorr -> BPM; no learning,
periodicity-aware -- exactly what prior_init's mean-pooling destroyed). Then:
  1. tempo accuracy of the estimator vs GT (Acc1 within 4%, Acc2 within 4% of any of /3,/2,x2,x3)
  2. F1 of a constant metronome at the ESTIMATED tempo (best phase)  -> does a simple estimate reach ~0.51?
  3. F1 of the oulong VAE free-run with tempo FROZEN at the estimate  -> VAE phase/meter + estimator tempo
Compare to: model free-run 0.356 | GT-tempo frozen 0.510.
"""
import sys, math
import numpy as np, torch
import torch.nn.functional as F
import mir_eval
sys.path.insert(0, "/home/sogang/jaehoon/CHART")
from faithful.model import BarPointerVAE
from faithful.data import FPS, N_MELS, LogMel, iter_val_songs
from faithful.distributions import TWO_PI

dev = "cuda"; ROOT = "/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data"; DS = ["ballroom","beatles","hains","rwc_popular"]
CK = "/home/sogang/.tmp/claude-1003/-home-sogang-jaehoon-CHART/84e38297-7220-4bbe-b30a-42cd7c5a3087/scratchpad/runs/nf/w6lat_oulong/final.pt"


def f1(ref, est): return 0.0 if len(est)==0 else (float("nan") if len(ref)==0 else float(mir_eval.beat.f_measure(ref, est)))


def estimate_tempo(mel):
    """classic: onset envelope (positive spectral flux) -> autocorrelation -> dominant beat period."""
    flux = np.maximum(0.0, np.diff(mel, axis=0)).sum(1)          # [T-1] onset strength
    flux = flux - flux.mean(); flux = flux / (flux.std() + 1e-8)
    ac = np.correlate(flux, flux, "full")[len(flux)-1:]          # autocorrelation, lags >= 0
    lo = int(round(60.0/220*FPS)); hi = int(round(60.0/50*FPS))  # 50-220 BPM beat-period range (frames)
    hi = min(hi, len(ac)-1)
    lag = lo + int(np.argmax(ac[lo:hi]))
    return 60.0 * FPS / lag                                       # BPM


def metronome(bpm, phase0_s, T):
    period = 60.0/bpm
    return np.arange(phase0_s, T/FPS, period)


def main():
    ck = torch.load(CK, map_location=dev); a = ck.get("args", {})
    model = BarPointerVAE(h_dim=N_MELS, hidden=a.get("hidden",64), num_meters=a.get("num_meters",4), latent_only=True).to(dev)
    model.load_state_dict(ck["model"]); model.eval(); logmel = LogMel().to(dev)
    from faithful.elbo import free_run
    acc1=[]; acc2=[]; fmet=[]; fvae=[]; fmodel=[]
    for key,audio,b,downs,meta in iter_val_songs(ROOT,DS,max_per_dataset=4):
        T=min(len(b),1200); ref=np.where(b.numpy()[:T]>0.5)[0]/FPS; df=np.where(downs.numpy()[:T]>0.5)[0]/FPS
        if len(ref)<8: continue
        m=4
        if len(df)>=2:
            bpb=np.median([np.sum((ref>=df[i])&(ref<df[i+1])) for i in range(len(df)-1)]); m=max(2,min(int(round(bpb)) if bpb>0 else 4,4))
        h=logmel(audio.to(dev).unsqueeze(0))[:,:T]
        mel=h[0].cpu().numpy()
        est_bpm=estimate_tempo(mel); gt_bpm=60.0/np.median(np.diff(ref))
        acc1.append(float(abs(est_bpm-gt_bpm)/gt_bpm<0.04))
        acc2.append(float(any(abs(est_bpm-gt_bpm*r)/(gt_bpm*r)<0.04 for r in (1/3,1/2,1,2,3))))
        # (2) metronome at estimated tempo, best phase
        per=60.0/est_bpm
        fmet.append(max(f1(ref, metronome(est_bpm, off, T)) for off in np.arange(0, per, per/8)))
        # (3) VAE free-run but tempo frozen at the estimate (beat-rate -> bar-advance lt); keep model phase chain
        lt_est = math.log(TWO_PI*(est_bpm/60.0)/m/FPS)
        with torch.no_grad():
            pc=model.encode_prior(h); pm,pphmu,pphk,ptmu,pts=model.unpack(model.prior_init_head(pc.mean(1)))
            phi=float(pphmu[0])%TWO_PI; step=math.exp(lt_est); chain=[phi]
            for t in range(1,T): phi=(phi+step)%TWO_PI; chain.append(phi)
            chain=np.array(chain)
            psi=(m*chain)%TWO_PI; w=np.where(np.diff(psi)<-math.pi)[0]+1
            est_beats=w[np.diff(np.concatenate([[-99],w]))>=0.10*FPS]/FPS
        fvae.append(f1(ref, est_beats))
        fmodel.append(f1(ref, []) if False else None)  # placeholder
    print(f"songs={len(acc1)}")
    print(f"  SIMPLE autocorr tempo estimator:  Acc1(within4%)={np.mean(acc1):.3f}   Acc2(octave-tolerant)={np.mean(acc2):.3f}")
    print(f"     (model prior_init tempo was 852% off, 0% octave -- contrast)")
    print(f"  (2) constant metronome @ ESTIMATED tempo, best phase   F1 = {np.nanmean(fmet):.3f}")
    print(f"  (3) oulong VAE free-run with tempo FROZEN @ estimate    F1 = {np.nanmean(fvae):.3f}")
    print(f"  refs: model free-run 0.356 | GT-tempo frozen 0.510")


if __name__=="__main__": main()
