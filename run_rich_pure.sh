#!/bin/bash
# =============================================================================
# RICH-FEATURE suite, PURE base (no bar_phase).
# rich_pure (faithful, no bar_phase, no fixes) beat everything: beatF 0.743 /
# dbF 0.342, vs the base (+bar_phase) configs stuck ~0.37. So bar_phase HURTS on
# rich input -- which means the DR fixes were being tested on a handicapped base.
# This suite re-tests each fix (and combos) stacked on PURE, the actual winner.
#
# Reuses the existing rich cache (cache/acts/bt_train_rich); NO caching step.
# Chained to run AFTER the base suite (tmux 'rich') finishes.
# =============================================================================
cd /home/sogang/jaehoon/CHART || exit 1
CH=/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python
mkdir -p logs/rich_pure cache/diag
ORCH=logs/rich_pure/orch.log; : > "$ORCH"
log(){ echo "[orch $(date +%H:%M:%S)] $*" | tee -a "$ORCH"; }
log "RICH-PURE SUITE START"

TRAIN=cache/acts/bt_train_rich
VAL=cache/acts/bt_val_rich
NTR=$(find "$TRAIN" -name '*.pt' 2>/dev/null | wc -l)
if [ "$NTR" -lt 1900 ]; then log "ABORT: rich cache missing (train=$NTR)"; exit 1; fi
log "using rich cache: train=$NTR val=$(find "$VAL" -name '*.pt'|wc -l)"

#   tag | idea (NO bar_phase) | steps | baseline
EXPERIMENTS=(
  # --- FAST, high-value (non-DVBF) FIRST: freerun is the one that won on base ---
  "freerun|freerun|250|faithful"            # pure + the exposure-bias fix (TOP priority)
  "delta|delta_vae|250|faithful"
  "phasesup|phasesup|250|faithful"          # dense BEAT-phase sup on phi (no bar_phase)
  "bz|z_forcing|250|faithful"               # predicts 512-dim features
  "delta_freerun|delta_vae+freerun|250|faithful"
  "aggr|aggressive_encoder|150|faithful"    # He-2019
  # --- DVBF configs LAST (DVBF is ~50x slower; also a clear loser on base) ---
  "dvbf|dvbf|250|faithful"
  "delta_dvbf|delta_vae+dvbf|250|faithful"
  "dvbf_freerun|dvbf+freerun|250|faithful"
  "delta_dvbf_freerun|delta_vae+dvbf+freerun|250|faithful"
  "all|delta_vae+dvbf+freerun+phasesup|250|faithful"
)
log "queued ${#EXPERIMENTS[@]} pure-based configs"

COMMON="--train_cache $TRAIN --heldout_cache $VAL \
  --eval_every 100 --frames 512 --batch_size 16 --lr 1e-3 --eval_frames 2048 --max_heldout 24"
run(){  # gpu tag idea steps baseline
  local g=$1 tag=$2 idea=$3 steps=$4 base=$5
  log "GPU$g START rp_$tag (idea=$idea steps=$steps)"
  CUDA_VISIBLE_DEVICES=$g $CH -u tests/fast_proxy.py $COMMON \
    --baseline "$base" --idea "$idea" --tag "rp_$tag" --steps "$steps" \
    --save_best "cache/diag/rp_${tag}_best.pt" --save_ckpt "cache/diag/rp_${tag}_final.pt" \
    > "logs/rich_pure/${tag}.log" 2>&1
  log "GPU$g DONE rp_$tag -> $(grep -h 'BEST {' logs/rich_pure/${tag}.log | tail -1)"
}

GPU_ARR=(0 0 0 3 3 3); NG=${#GPU_ARR[@]}   # 3/GPU (dispatch-bound)
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

log "ALL DONE"
{
  echo "===== RICH-PURE SUITE (peak beatF / dbF; cf. rich_pure 0.743/0.342, [T,2] 0.448/0.180) ====="
  for e in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r tag idea steps base <<< "$e"
    echo "rp_$tag | $(grep -h 'BEST {' logs/rich_pure/${tag}.log 2>/dev/null | tail -1)"
  done
} | tee -a "$ORCH"
log "RICH-PURE SUITE COMPLETE"
