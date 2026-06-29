"""OOD test of the A+B synthesis on SMC MIREX. Train faithful_autocorr_filter (computed phidot + filter +
sawtooth) on bt_train_rich @86fps, then deploy posterior-phi geometric read-out on SMC (resample 50->86fps).
Beat F vs GT + shuffle leak. Reference: Beat-This act2 peak-pick 0.586; pure-sawtooth model collapsed to 0.19.
"""
import sys, math, glob, importlib.util, random, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT="/home/sogang/jaehoon/CHART"; sys.path.insert(0,ROOT)
ab=importlib.util.spec_from_file_location("AB",f"{ROOT}/experiments/diagram_arch/faithful_autocorr_filter.py")
AB=importlib.util.module_from_spec(ab); ab.loader.exec_module(AB)
da=AB.da; BPVAE=AB.BPVAE; TempoNet=AB.TempoNet; load_pool=AB.load_pool; sample_batch=AB.sample_batch
phase_beats=AB.phase_beats; fmeas=AB.fmeas; rollout=AB.rollout; elbo=AB.elbo
DEV=AB.DEV; FPS=86.1328125; SMC_FPS=50.0; M=4
ANNOT="/home/sogang/jaehoon/Analyze-SMC/SMC_MIREX/SMC_MIREX_Annotations"


def gt_beats(tid):
    fs=glob.glob(f"{ANNOT}/SMC_{tid.split('_')[1]}_*.txt")
    return np.loadtxt(fs[0]).reshape(-1) if fs else None


def resample_feat(feat):
    T=feat.shape[0]; Tn=int(round(T*FPS/SMC_FPS))
    y=F.interpolate(feat.t().unsqueeze(0).float(),size=Tn,mode="linear",align_corners=False)
    return y[0].t().contiguous()


@torch.no_grad()
def eval_smc(model,tnet,gain,files,shuffle=False,frames=2600):
    model.eval(); tnet.eval(); Fs=[]; n=len(files)
    feats=[resample_feat(torch.load(f,map_location="cpu")["feat"]) for f in files]
    tids=[torch.load(f,map_location="cpu")["tid"] for f in files]
    for i in range(len(files)):
        ref=gt_beats(tids[i])
        if ref is None or len(ref)<2: continue
        feat=feats[(i+1)%n] if shuffle else feats[i]
        T=min(feat.shape[0],frames)
        h_in=feat[:T].unsqueeze(0).to(DEV); z=torch.zeros(1,T,device=DEV)
        _,phi,_,_,_=rollout(model,tnet,gain,h_in,z,z,sample=False,compute_kl=False)
        Fs.append(fmeas(ref,phase_beats(phi[0].cpu().numpy(),M)))
    model.train(); tnet.train(); return float(np.nanmean(Fs)),len(Fs)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=1500)
    a=ap.parse_args()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    train=load_pool("cache/acts/bt_train_rich",400,seed=1)
    print(f"SMC OOD (synthesis A+B) | training on {len(train)} songs @86fps\n",flush=True)
    model=BPVAE(h_dim=512,hidden=64).to(DEV); tnet=TempoNet(512).to(DEV); gain=nn.Parameter(torch.tensor(0.0,device=DEV))
    opt=torch.optim.Adam(list(model.parameters())+list(tnet.parameters())+[gain],lr=1e-3)
    for step in range(1,a.steps+1):
        temp=1.0+(0.3-1.0)*min(step/a.steps,1.0)
        h,b,db=sample_batch(train,256,16)
        loss,info=AB.elbo(model,tnet,gain,h,b,db,temp,0.5)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(tnet.parameters())+[gain],5.0); opt.step()
        if step%300==0: print(f"  step {step} recon {info['recon']:.0f} Lsaw {info['Lsaw']:.3f} Lt {info['Lt']:.3f} gain {info['g']:.2f}",flush=True)
    torch.save({"vae":model.state_dict(),"tnet":tnet.state_dict(),"gain":gain.detach()},"experiments/diagram_arch/smc_synth_model.pt")
    files=sorted(glob.glob("cache/acts/smc_rich_heldout/*.pt"))
    print(f"\nevaluating on {len(files)} SMC files (resampled 50->86fps)...",flush=True)
    Fr,nr=eval_smc(model,tnet,gain,files,False); Fs,_=eval_smc(model,tnet,gain,files,True)
    print(f"\n==== SMC OOD (synthesis A+B) ====")
    print(f"  ours (filter+autocorr phidot): F = {Fr:.3f} over {nr} songs")
    print(f"  leak (shuffled audio)        : F = {Fs:.3f}")
    print(f"  reference: Beat-This act2 peak-pick 0.586 | pure-sawtooth model collapsed 0.19")
    print("DONE-SMC-SYNTH")


if __name__=="__main__": main()
