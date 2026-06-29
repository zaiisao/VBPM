"""OOD eval on SMC MIREX (beats-only, hard; Beat-This published F=0.62). Train the bar+beat sawtooth
model on bt_train_rich (86fps), then deploy the POSTERIOR phi geometric read-out on SMC held-out
features. SMC cache is native 50fps -> resample features to 86fps to match training. GT beats from
SMC_MIREX_Annotations by tid. Reports mean beat F + leak (shuffle) to confirm audio-locked OOD.
"""
import sys, math, glob, importlib.util, random, argparse
import numpy as np, torch, torch.nn.functional as F
ROOT="/home/sogang/jaehoon/CHART"; sys.path.insert(0,ROOT)
s2=importlib.util.spec_from_file_location("sa2",f"{ROOT}/experiments/diagram_arch/sawtooth_aux2.py")
sa2=importlib.util.module_from_spec(s2); s2.loader.exec_module(sa2)
da=sa2.da; BPVAE=da.BPVAE; rollout=da.rollout; load_pool=da.load_pool; sample_batch=da.sample_batch
phase_beats=da.phase_beats; fmeas=da.fmeas
DEV=da.DEV; FPS=86.1328125; SMC_FPS=50.0; M=4; TWO_PI=2*math.pi
ANNOT="/home/sogang/jaehoon/Analyze-SMC/SMC_MIREX/SMC_MIREX_Annotations"


def gt_beats(tid):
    num=tid.split("_")[1]
    fs=glob.glob(f"{ANNOT}/SMC_{num}_*.txt")
    if not fs: return None
    return np.loadtxt(fs[0]).reshape(-1)


def resample_feat(feat):                     # [T,512] @50fps -> [T',512] @86fps (linear interp on time)
    T=feat.shape[0]; Tn=int(round(T*FPS/SMC_FPS))
    x=feat.t().unsqueeze(0).float()          # [1,512,T]
    y=F.interpolate(x,size=Tn,mode="linear",align_corners=False)
    return y[0].t().contiguous()             # [Tn,512]


@torch.no_grad()
def eval_smc(model, files, shuffle=False, frames=2600):
    model.eval(); Fs=[]; n=len(files)
    feats=[resample_feat(torch.load(f,map_location="cpu")["feat"]) for f in files]
    tids=[torch.load(f,map_location="cpu")["tid"] for f in files]
    for i,f in enumerate(files):
        ref=gt_beats(tids[i])
        if ref is None or len(ref)<2: continue
        feat=feats[(i+1)%n] if shuffle else feats[i]
        T=min(feat.shape[0],frames)
        h_in=feat[:T].unsqueeze(0).to(DEV); z=torch.zeros(1,T,device=DEV)
        _,phase_mu,_=rollout(model,h_in,z,z,sample=False,compute_kl=False)
        phi=phase_mu[0].cpu().numpy()
        est=phase_beats(phi,M)               # beat times in seconds (uses run.py FPS=86.13)
        Fs.append(fmeas(ref,est))
    model.train(); return float(np.nanmean(Fs)), len(Fs)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=1000)
    ap.add_argument("--lam_bar",type=float,default=0.5); ap.add_argument("--lam_beat",type=float,default=0.5)
    ap.add_argument("--max_smc",type=int,default=999)
    a=ap.parse_args()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    train=load_pool("cache/acts/bt_train_rich",400,seed=1)
    print(f"SMC OOD eval | training bar+beat (lam_bar={a.lam_bar} lam_beat={a.lam_beat}) on {len(train)} songs @86fps\n",flush=True)
    model=BPVAE(h_dim=512,hidden=64).to(DEV); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for step in range(1,a.steps+1):
        temp=1.0+(0.3-1.0)*min(step/a.steps,1.0)
        h,b,db=sample_batch(train,256,16)
        loss,info=sa2.elbo_aux2(model,h,b,db,temp,a.lam_bar,a.lam_beat)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        if step%250==0: print(f"  step {step} recon {info['recon']:.0f} Lbar {info['Lbar']:.3f} Lbeat {info['Lbeat']:.3f}",flush=True)
    torch.save({"vae":model.state_dict(),"h_dim":512},"experiments/diagram_arch/smc_model.pt")
    files=sorted(glob.glob("cache/acts/smc_rich_heldout/*.pt"))[:a.max_smc]
    print(f"\nevaluating on {len(files)} SMC files (resampled 50->86fps)...",flush=True)
    Freal,nr=eval_smc(model,files,shuffle=False)
    Fshuf,_=eval_smc(model,files,shuffle=True)
    print(f"\n==== SMC MIREX (OOD, beats-only) ====")
    print(f"  ours (posterior phi geom read-out): F = {Freal:.3f}  over {nr} songs")
    print(f"  leak (shuffled audio)             : F = {Fshuf:.3f}  (must be << real => audio-locked)")
    print(f"  reference: Beat-This published SMC F = 0.62 ; madmom DBN ~0.52")
    print("DONE")


if __name__=="__main__": main()
