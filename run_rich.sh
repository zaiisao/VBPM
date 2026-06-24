#!/bin/bash
# =============================================================================
# RICH-FEATURE suite (the user's instruction: do NOT feed CHART the collapsed
# [T,2] activations; ingest Beat This's 512-dim penultimate representation).
# This is the FULL Deep-Research shortlist re-run on the CORRECT (rich) input.
#
#   1. Cache Beat This [T,512] features (transformer_blocks output) -> train (2000,
#      matches bt_train_fixed) + val, fp16 ~13GB. Skips if already complete.
#   2. Sanity-gate: dim>2 AND beats/song>=45.
#   3. Train CHART on the RICH input across the full matrix, peak ckpts.
#
# THE verdict: rich_base vs the [T,2] baseline (faith_FIXED peak beatF 0.448).
#
# Launch:  tmux new-session -d -s rich 'bash run_rich.sh'
# =============================================================================
cd /home/sogang/jaehoon/CHART || exit 1
CH=/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python
ROOT=/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data
mkdir -p logs/rich cache/diag
ORCH=logs/rich/orch.log; : > "$ORCH"
log(){ echo "[orch $(date +%H:%M:%S)] $*" | tee -a "$ORCH"; }
log "RICH SUITE START"

TRAIN=cache/acts/bt_train_rich
VAL=cache/acts/bt_val_rich
CFLAGS="--extractor beat_this --beat_this_checkpoint final0 --extractor_fps_mode resample \
  --dataset_root $ROOT --dataset_include ballroom,beatles,hains,rwc_popular --rich --max_frames 9000"

# ---- 1. cache (re-cache unless BOTH splits are complete) ----
NTR=$(find "$TRAIN" -name '*.pt' 2>/dev/null | wc -l)
NVA=$(find "$VAL" -name '*.pt' 2>/dev/null | wc -l)
if [ "$NTR" -lt 1900 ] || [ "$NVA" -lt 100 ]; then
  log "caching RICH train (GPU0, 2000) + val (GPU3, all) in parallel  [had train=$NTR val=$NVA] ..."
  CUDA_VISIBLE_DEVICES=0 $CH tests/cache_activations.py $CFLAGS --split train --max_songs 2000 --out_dir "$TRAIN" > logs/rich/cache_train.log 2>&1 &
  CUDA_VISIBLE_DEVICES=3 $CH tests/cache_activations.py $CFLAGS --split val   --max_songs 400  --out_dir "$VAL"   > logs/rich/cache_val.log   2>&1 &
  wait
else
  log "rich cache already complete (train=$NTR val=$NVA), skipping caching"
fi
log "cache: train=$(find "$TRAIN" -name '*.pt'|wc -l) songs, val=$(find "$VAL" -name '*.pt'|wc -l) songs"

# ---- 2. sanity gate ----
$CH - "$TRAIN" <<'PY'
import sys, glob, torch
fs = sorted(glob.glob(sys.argv[1] + "/*.pt"))[:40]
b = [float(torch.load(f, map_location="cpu")["beat_targets"].sum()) for f in fs]
d = torch.load(fs[0], map_location="cpu")["activations"].shape[-1] if fs else 0
m = sum(b) / max(len(b), 1)
print(f"[sanity] {len(fs)} songs, feature dim = {d}, mean beats/song = {m:.1f}")
sys.exit(0 if (d > 2 and m >= 45) else 7)
PY
if [ $? -ne 0 ]; then log "ABORT: rich cache failed sanity (dim<=2 or truncated)"; exit 1; fi

# ---- 3. train CHART on the RICH input -- FULL Deep-Research shortlist ----
#   tag | idea | steps | baseline   (all faithful + bar_phase base unless noted)
EXPERIMENTS=(
  # --- baselines + INDIVIDUAL DR fixes (ablation: which fix helps on rich input?) ---
  "base|bar_phase|250|faithful"                              # reference (cf. [T,2] faith_FIXED 0.448)
  "pure|none|250|faithful"                                   # cleanest faithful floor (no bar_phase)
  "delta|bar_phase+delta_vae|250|faithful"                   # DR#3 delta-VAE (collapse/rate)
  "dvbf|bar_phase+dvbf|250|faithful"                         # DR#5 DVBF (free-run dynamics)
  "freerun|bar_phase+freerun|250|faithful"                   # DR#2 exposure bias (SS=0.8)
  "phasesup|bar_phase+phasesup|250|faithful"                 # phase identifiability
  # --- COMBINATIONS of mechanism-diverse fixes (most likely complementary) ---
  "delta_dvbf|bar_phase+delta_vae+dvbf|250|faithful"                  # collapse + dynamics
  "delta_dvbf_freerun|bar_phase+delta_vae+dvbf+freerun|250|faithful"  # + exposure (3 pillars)
  "all4|bar_phase+delta_vae+dvbf+freerun+phasesup|250|faithful"       # maximal faithful stack
  "delta_phasesup|bar_phase+delta_vae+phasesup|250|faithful"          # collapse + identifiability
  "dvbf_freerun|bar_phase+dvbf+freerun|250|faithful"                  # dynamics + exposure
  # --- secondary singles + references ---
  "bz|bar_phase+z_forcing|250|faithful"                      # z-forcing aux (predicts 512-dim)
  "dir1|bar_phase|250|dir1"                                  # strong non-faithful reference
  "aggr|bar_phase+aggressive_encoder|150|faithful"           # DR#4 He-2019 (expensive, last)
)
log "queued ${#EXPERIMENTS[@]} rich configs"

COMMON="--train_cache $TRAIN --heldout_cache $VAL \
  --eval_every 100 --frames 512 --batch_size 16 --lr 1e-3 --eval_frames 2048 --max_heldout 24"
run(){  # gpu tag idea steps baseline
  local g=$1 tag=$2 idea=$3 steps=$4 base=$5
  log "GPU$g START $tag (idea=$idea base=$base steps=$steps)"
  CUDA_VISIBLE_DEVICES=$g $CH -u tests/fast_proxy.py $COMMON \
    --baseline "$base" --idea "$idea" --tag "rich_$tag" --steps "$steps" \
    --save_best "cache/diag/rich_${tag}_best.pt" --save_ckpt "cache/diag/rich_${tag}_final.pt" \
    > "logs/rich/${tag}.log" 2>&1
  log "GPU$g DONE $tag -> $(grep -h 'BEST {' logs/rich/${tag}.log | tail -1)"
}

# one lane per GPU (full speed, no contention); each lane runs its share sequentially
GPU_ARR=(0 0 0 3 3 3); NG=${#GPU_ARR[@]}   # pack 3/GPU: runs are dispatch-bound (~27% util), not GPU-bound
lane=0
for g in "${GPU_ARR[@]}"; do
  (
    j=$lane
    while [ $j -lt ${#EXPERIMENTS[@]} ]; do
      IFS='|' read -r tag idea steps base <<< "${EXPERIMENTS[$j]}"
      run "$g" "$tag" "$idea" "$steps" "$base"
      j=$((j + NG))
    done
  ) &
  lane=$((lane + 1))
done
wait

# ---- summary ----
log "ALL DONE"
{
  echo "===== RICH-FEATURE SUITE (peak beatF; cf. [T,2] faith_FIXED peak 0.448) ====="
  for e in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r tag idea steps base <<< "$e"
    echo "rich_$tag | $(grep -h 'BEST {' logs/rich/${tag}.log 2>/dev/null | tail -1)"
  done
} | tee -a "$ORCH"
log "RICH SUITE COMPLETE"
