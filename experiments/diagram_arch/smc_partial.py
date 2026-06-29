import sys, glob, importlib.util, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT="/home/sogang/jaehoon/CHART"; sys.path.insert(0,ROOT)
ab=importlib.util.spec_from_file_location("AB",f"{ROOT}/experiments/diagram_arch/faithful_autocorr_filter.py")
AB=importlib.util.module_from_spec(ab); ab.loader.exec_module(AB)
se=importlib.util.spec_from_file_location("SE",f"{ROOT}/experiments/diagram_arch/smc_eval_synth.py")
SE=importlib.util.module_from_spec(se); se.loader.exec_module(SE)
DEV=AB.DEV
ck=torch.load("experiments/diagram_arch/smc_synth_model.pt",map_location=DEV)
model=AB.BPVAE(h_dim=512,hidden=64).to(DEV); model.load_state_dict(ck["vae"])
tnet=AB.TempoNet(512).to(DEV); tnet.load_state_dict(ck["tnet"]); gain=ck["gain"]
files=sorted(glob.glob("cache/acts/smc_rich_heldout/*.pt"))[:30]
Fr,nr=SE.eval_smc(model,tnet,gain,files,False)
Fs,_=SE.eval_smc(model,tnet,gain,files,True)
print(f"PARTIAL SMC (first {nr} songs): real F={Fr:.3f} | shuf F={Fs:.3f} | (BT 0.586, sawtooth-model 0.19)",flush=True)
