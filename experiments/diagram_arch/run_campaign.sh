#!/bin/bash
# Diagram-architecture campaign (runs in tmux, survives disconnect):
#   A: pretrained-frozen Beat-This (4 datasets, cached) + VAE  -> checkpoints/diagram_A.pt
#   B: from-scratch end-to-end TCN(log-mel)+VAE (4 datasets)   -> checkpoints/diagram_B.pt
#   SMC-MIREX eval of both (hard cross-dataset generalization)
# Log: experiments/diagram_arch/campaign.log
cd /home/sogang/jaehoon/CHART
PY=/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python
export CUDA_VISIBLE_DEVICES=0
LOG=experiments/diagram_arch/campaign.log
FILT="UserWarning|warn\(|gmpy|set_audio|FutureWarning|torch.load|weights_only|Selected [0-9]"
{
  echo "######## A: PRETRAINED-FROZEN Beat-This (4 datasets, cached) + VAE, 600 steps ########"
  date
  $PY experiments/diagram_arch/run.py --n_train 400 --n_val 80 --steps 600 --eval_every 300 \
      --frames 256 --save checkpoints/diagram_A.pt 2>&1 | grep -vE "$FILT"
  echo; echo "######## B: FROM-SCRATCH end-to-end TCN(log-mel)+VAE (4 datasets), 1500 steps ########"
  date
  $PY experiments/diagram_arch/e2e.py --val_per_ds 8 --steps 1500 --eval_every 750 \
      --frames 256 --save checkpoints/diagram_B.pt 2>&1 | grep -vE "$FILT"
  echo; echo "######## SMC-MIREX EVAL (A + B) ########"
  date
  $PY experiments/diagram_arch/smc_eval.py --ckpt_A checkpoints/diagram_A.pt \
      --ckpt_B checkpoints/diagram_B.pt 2>&1 | grep -vE "$FILT"
  echo "CAMPAIGN_DONE"; date
} > $LOG 2>&1
