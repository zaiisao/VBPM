#!/bin/bash
# OVERNIGHT: CHART on Beat This + bar_phase + z_forcing, FAITHFUL (ou2-orthodox: latent-only
# decoder + free-bits + OU(ema) + small corr nudge; NO direct latent supervision, NO scheduled
# sampling, NO audio_recon — audio_emission kept ONLY for the z_forcing aux head). The clean,
# defensible config. Eval on clean + SMC in the morning.
export CUDA_VISIBLE_DEVICES=${1:-3}
ROOT=/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data
CH=/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python
cd /home/sogang/jaehoon/CHART
exec $CH -m training.train \
  --mode end2end \
  --extractor beat_this --beat_this_checkpoint final0 --extractor_fps_mode resample --freeze_extractor \
  --dataset_root "$ROOT" \
  --dataset_include ballroom,beatles,hains,rwc_popular \
  --wavebeat_root extractors/wavebeat \
  --seq_len 1024 --batch_size 8 --num_epochs 25 --lr 0.0003 --num_workers 3 \
  --train_length 524288 \
  --examples_per_epoch 1600 --num_meter_classes 8 \
  --kl_anneal_epochs 5 --bce_pos_weight 5.0 --bce_pos_weight_db 15.0 \
  --free_bits_phase 0.2 --free_bits_tempo 0.1 --free_bits_meter 0.2 --free_bits_barphase 0.2 --max_grad_norm 1.0 \
  --phase_corr_scale 0.1 --tempo_corr_scale 0.15 --decoder_latent_only \
  --tempo_anchor_mode ema --tempo_reversion_alpha 0.4 \
  --audio_emission --audio_recon_weight 0.0 \
  --bar_phase --barphase_sup_weight 1.0 \
  --z_forcing_weight 1.0 --z_forcing_offset 8 \
  --val_every 2 --val_subset_batches 24 --no_wandb \
  --save_ckpt_path checkpoints/bz_bt_faith_fix/chart.pt
