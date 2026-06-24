#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
ROOT=/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data
CH=/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python
cd /home/sogang/jaehoon/CHART
exec $CH -m training.train \
  --mode end2end \
  --extractor wavebeat --extractor_ckpt wavebeat_epoch=98-step=24749.ckpt --freeze_extractor \
  --dataset_root "$ROOT" \
  --dataset_include ballroom,beatles,hains,rwc_popular \
  --wavebeat_root extractors/wavebeat \
  --seq_len 512 --batch_size 8 --num_epochs 14 --lr 0.0003 --num_workers 3 \
  --examples_per_epoch 200 --num_meter_classes 8 \
  --kl_anneal_epochs 3 --bce_pos_weight 5.0 --bce_pos_weight_db 15.0 \
  --free_bits_phase 0.2 --free_bits_tempo 0.1 --free_bits_meter 0.2 --max_grad_norm 1.0 \
  --phase_corr_scale 0.1 --tempo_corr_scale 0.15 --decoder_latent_only \
  --tempo_anchor_mode latent --tempo_reversion_alpha 0.4 --taubar_sup_weight 1.0 \
  --meter_sup_weight 1.0 --phase_sup_weight 1.0 \
  --scheduled_sampling_max 0.5 \
  --val_every 1 --val_subset_batches 16 --no_wandb \
  --save_ckpt_path checkpoints/probe3/chart.pt
