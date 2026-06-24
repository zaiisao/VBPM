#!/bin/bash
# =============================================================================
# Deep-Research shortlist A/B suite, on the FIXED cache (post data-bug fixes).
#
# Tests whether the DEEP_RESEARCH.md fixes address the CURRENT (post-fix) issues:
#   floor-pinned KLs (mild posterior collapse) + free-run cap ~0.45 + overfit/
#   degrade-past-peak on small data. Every idea is A/B'd against the SAME faithful
#   baseline + cache + eval protocol, so the ranking is apples-to-apples.
#
# Self-contained: auto-detects the fixed cache, REFUSES to run on truncated data
# (the gate: no number trusted from bad data), auto-picks free GPUs, runs the
# matrix in parallel lanes (one per GPU), saves the PEAK checkpoint per run, and
# prints a summary table. Survives client disconnect (run under tmux on the host).
#
# Launch:  tmux new-session -d -s drsuite 'bash run_dr_suite.sh'
# Override: GPUS="0 1" TRAIN_CACHE=... VAL_CACHE=... FORCE=1 bash run_dr_suite.sh
# =============================================================================
cd /home/sogang/jaehoon/CHART || exit 1
CH=/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python
mkdir -p logs/dr_suite cache/diag
ORCH=logs/dr_suite/orch.log
log() { echo "[orch $(date +%H:%M:%S)] $*" | tee -a "$ORCH"; }
: > "$ORCH"
log "DR SUITE START"

# ---- locate the FIXED train/val cache (env override wins) --------------------
if [ -z "${TRAIN_CACHE:-}" ]; then
  for c in cache/acts/bt_train_fixed cache/acts/bt_train cache/acts/wb_train; do
    [ -n "$(find "$c" -name '*.pt' 2>/dev/null | head -1)" ] && { TRAIN_CACHE="$c"; break; }
  done
fi
if [ -z "${VAL_CACHE:-}" ]; then
  for c in cache/acts/bt_val_fixed cache/acts/bt_val cache/acts/wb_val; do
    [ -n "$(find "$c" -name '*.pt' 2>/dev/null | head -1)" ] && { VAL_CACHE="$c"; break; }
  done
fi
if [ -z "$TRAIN_CACHE" ] || [ -z "$VAL_CACHE" ]; then
  log "ABORT: could not find a train/val cache under cache/acts/. Set TRAIN_CACHE / VAL_CACHE."; exit 1
fi
NTR=$(find "$TRAIN_CACHE" -name '*.pt' 2>/dev/null | wc -l)
NVA=$(find "$VAL_CACHE" -name '*.pt' 2>/dev/null | wc -l)
log "train_cache=$TRAIN_CACHE ($NTR songs)  val_cache=$VAL_CACHE ($NVA songs)"

# ---- sanity gate: REFUSE truncated data (fixed ~80 beats/song, truncated ~30) -
$CH - "$TRAIN_CACHE" <<'PY'
import sys, glob, torch
d = sys.argv[1]
fs = sorted(glob.glob(d + "/**/*.pt", recursive=True))[:60] or sorted(glob.glob(d + "/*.pt"))[:60]
bs = []
for f in fs:
    try: bs.append(float(torch.load(f, map_location="cpu")["beat_targets"].sum()))
    except Exception: pass
m = sum(bs) / max(len(bs), 1)
print(f"[sanity] {d}: sampled {len(bs)} songs, mean beats/song = {m:.1f} "
      f"(fixed cache ~80, truncated ~30)")
sys.exit(0 if m >= 45 else 7)
PY
if [ $? -ne 0 ]; then
  log "ABORT: cache looks TRUNCATED (mean beats/song < 45) -> numbers would be invalid (the gate)."
  [ "${FORCE:-0}" != "1" ] && exit 1
  log "FORCE=1 set -> continuing on suspect cache anyway."
fi

# ---- pick free GPUs (mem.used < 2500 MiB), env GPUS overrides ----------------
pick_gpus() {
  command -v nvidia-smi >/dev/null 2>&1 || return
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 < 2500) print $1}'
}
GPUS="${GPUS:-$(pick_gpus)}"
[ -z "$GPUS" ] && GPUS="0"
GPU_ARR=($GPUS); NG=${#GPU_ARR[@]}
log "GPUs: ${GPU_ARR[*]} ($NG lanes)"

# ---- experiment matrix:  tag | idea | steps | baseline --------------------- #
# All share: faithful baseline + bar_phase+z_forcing (the established config that
# reproduces faith_FIXED) so each DR fix is an additive A/B. dir1 is the strong
# non-faithful reference. aggr (He-2019) is expensive -> fewer steps.
EXPERIMENTS=(
  "base|bar_phase+z_forcing|500|faithful"                       # reference (= faith_FIXED)
  "delta|bar_phase+z_forcing+delta_vae|500|faithful"            # DR#3 delta-VAE  (floor-collapse)
  "dvbf|bar_phase+z_forcing+dvbf|500|faithful"                  # DR#5 DVBF       (free-run dynamics)
  "delta_dvbf|bar_phase+z_forcing+delta_vae+dvbf|500|faithful"  # DR#3+#5 compose
  "freerun|bar_phase+z_forcing+freerun|500|faithful"            # DR#2 exposure bias (SS=0.8)
  "phasesup|bar_phase+z_forcing+phasesup|500|faithful"          # dense beat-phase sup (diag fix)
  "aggr|bar_phase+z_forcing+aggressive_encoder|250|faithful"    # DR#4 He-2019 lagging enc
  "dir1ref|bar_phase+z_forcing|500|dir1"                        # strong non-faithful reference
)
log "queued ${#EXPERIMENTS[@]} experiments"

run_one() {
  local gpu="$1" tag="$2" idea="$3" steps="$4" base="$5"
  local logf="logs/dr_suite/${tag}.log"
  log "GPU$gpu START $tag (idea=$idea base=$base steps=$steps) -> $logf"
  CUDA_VISIBLE_DEVICES="$gpu" $CH -u tests/fast_proxy.py \
    --train_cache "$TRAIN_CACHE" --heldout_cache "$VAL_CACHE" \
    --baseline "$base" --idea "$idea" --tag "dr_$tag" \
    --steps "$steps" --eval_every 50 --frames 512 --batch_size 16 --lr 1e-3 \
    --eval_frames 2048 --max_heldout 60 \
    --save_best "cache/diag/dr_${tag}_best.pt" \
    --save_ckpt "cache/diag/dr_${tag}_final.pt" \
    > "$logf" 2>&1
  log "GPU$gpu DONE  $tag  -> $(grep -h 'BEST {' "$logf" | tail -1)"
}

# ---- launch: one lane per GPU, each lane runs its share sequentially ---------
lane=0
for g in "${GPU_ARR[@]}"; do
  (
    j=$lane
    while [ $j -lt ${#EXPERIMENTS[@]} ]; do
      IFS='|' read -r tag idea steps base <<< "${EXPERIMENTS[$j]}"
      run_one "$g" "$tag" "$idea" "$steps" "$base"
      j=$((j + NG))
    done
  ) &
  lane=$((lane + 1))
done
wait

# ---- summary ----------------------------------------------------------------
log "ALL DONE — summary:"
{
  echo "===== DR SUITE SUMMARY (peak beatF on $NVA-song heldout, $TRAIN_CACHE) ====="
  printf "%-12s %-10s %-40s\n" "tag" "best_step" "best (beatF / downbeatF)"
  for e in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r tag idea steps base <<< "$e"
    b=$(grep -h 'BEST {' "logs/dr_suite/${tag}.log" 2>/dev/null | tail -1)
    echo "  $tag | $b"
  done
  echo "----- (final/degraded RESULT lines for contrast) -----"
  grep -h 'RESULT {' logs/dr_suite/*.log 2>/dev/null
} | tee -a "$ORCH"
log "DR SUITE COMPLETE"
