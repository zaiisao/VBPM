#!/bin/bash
# Self-contained pipeline (runs in tmux, survives disconnect): wait for the Beat This
# activation caching to finish, then train CHART (corrected von Mises sampler, big batch,
# bar_phase + z_forcing) on the cached activations — faithful AND dir1 in parallel.
cd /home/sogang/jaehoon/CHART
CH=/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python

echo "[orch] waiting for caching (CACHE0_DONE + CACHE3_DONE)..."
until grep -q CACHE0_DONE logs/cache/gpu0.log 2>/dev/null && grep -q CACHE3_DONE logs/cache/gpu3.log 2>/dev/null; do
  sleep 20
done
echo "[orch] caching done at $(cat logs/cache/gpu0.log | grep -c wrote) ... launching cached training"
echo "[orch] train songs: $(ls cache/acts/bt_train/*/*.pt 2>/dev/null | wc -l)  val songs: $(ls cache/acts/bt_val/*/*.pt 2>/dev/null | wc -l)"

COMMON="--train_cache cache/acts/bt_train --heldout_cache cache/acts/bt_val \
  --steps 2000 --eval_every 400 --frames 512 --batch_size 48 --lr 1e-3 \
  --eval_frames 2048 --max_heldout 40"

# faithful + B+Z on GPU0 ; dir1 + B+Z on GPU3 (corrected sampler is in models/distributions.py)
CUDA_VISIBLE_DEVICES=0 $CH -u tests/fast_proxy.py $COMMON \
  --baseline faithful --idea "bar_phase+z_forcing" --tag bt_cached_faith_BZ \
  > logs/cache/train_faith_BZ.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $CH -u tests/fast_proxy.py $COMMON \
  --baseline dir1 --idea "bar_phase+z_forcing" --tag bt_cached_dir1_BZ \
  > logs/cache/train_dir1_BZ.log 2>&1 &
wait
echo "[orch] CACHED_TRAINING_DONE"
