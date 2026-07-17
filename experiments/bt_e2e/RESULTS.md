# Preliminary: vanilla Beat Transformer vs Beat Transformer + R2 (end-to-end)

2026-07-18 overnight run. Both arms: official Demixed_DilatedTransformerModel FROM SCRATCH on our
4 CV datasets (fold-0 split: 1,020 train / 146 val songs, meter-representable subset), identical
beat-aligned crops (<=16 s), identical optimizer (RAdam+Lookahead, lr 1e-3, clip .5, batch 1, 30
epochs). Arm difference only in loss/decode:
  vanilla = their BCE (widened targets) -> madmom-as-BT-ships-it (obs_lambda=6, num_tempi=None,
            threshold=0.2), transition_lambda=100
  r2 e2e  = BCE + CRF NLL through the exact structured forward (rungs/r2_learned_dbn.py),
            learning the frontend AND transition_lambda jointly -> same decode, learned lambda

## Final table (FULL fold-0 val, 146 songs, best checkpoints)

| activations           | decode lambda | beat F | downbeat F |
|-----------------------|---------------|--------|------------|
| vanilla               | 100           | 0.9506 | 0.9053     |
| vanilla               | 14.2 (learned)| 0.9467 | 0.8808     |
| r2 e2e                | 100           | 0.9446 | 0.8899     |
| r2 e2e                | 14.2 (learned)| 0.9374 | 0.8637     |
| r2 e2e                | 14.2, threshold=0 | 0.9341 | 0.8600 |
| pretrained fold_0 (LEAKY ref: their folds are not ours) | 100 | 0.9622 | 0.9012 |

## Verdict (preliminary)

VANILLA WINS. The e2e CRF arm does not beat BCE + fixed madmom DBN; best r2 cell trails by
-0.006 beat F / -0.015 downbeat F, and r2's own learned lambda makes it worse, not better.

Decomposed:
1. LEARNED LAMBDA IS LIKELIHOOD-OPTIMAL BUT F-SUBOPTIMAL. lambda converged decisively to ~14.2
   (7x looser than madmom's 100) and was rock-stable for 10+ epochs -- the CRF genuinely prefers
   a flexible tempo kernel (consistent with the heavy-tailed tempo-increment finding). But at
   DECODE, lambda=100 beats lambda=14.2 for BOTH frontends (vanilla: 0.9506 vs 0.9467). F-measure
   rewards tempo continuity more than likelihood does. Likelihood != task metric.
2. THE CRF TERM SLIGHTLY HURT THE FRONTEND: r2 acts decoded at lambda=100 still trail vanilla
   acts (0.9446 vs 0.9506). The structure gradient cost a little BCE-fit without buying F.
3. NO DECODE ARTIFACT: bare (threshold=0) ~= shipped for r2 (0.9341 vs 0.9374), so the BCE anchor
   kept activations calibrated; the loss is real, not a thresholding illusion.
4. OUR PIPELINE IS HEALTHY: from-scratch vanilla (0.9506) nearly matches the pretrained
   reference (0.9622) which SAW some of our val songs in its training (leaky).

## Failure archaeology (what it took to get here)

- Run 1: both arms NaN'd (their model does this; their own train.py carries a NaN-skip guard we
  had not replicated). Fixed: skip nonfinite loss AND nonfinite grad-norm; per-epoch weight
  health check; resume support. vanilla resumed from epoch 7.
- Run 2 (r2): PURE CRF SATURATES. With no calibration pressure, logits hit |55|, 100% of frames
  sigmoid-saturated within ONE epoch; CRF gradient dies (flat under saturation), lambda absorbs
  all remaining gradient, decode freezes (bit-identical F across evals). Fix: hybrid
  CRF + BCE -- BCE's gradient is maximal exactly where CRF's is zero. This also sharpened the
  design: arm delta = the exact-forward structure term alone.

## Caveats

Single fold, single seed, 30 epochs, hybrid weight 1:1 untuned, 16-s crops (short-range; CRF's
long-range structure advantage may need full songs), lambda learned jointly (not decode-tuned).
Skipped 139 songs (no meter/grid representation). The 60-song training-time evals ran ~+0.01 high
vs the full fold (subset bias) -- use this table, not the training logs.

## Follow-ups suggested by the data

1. Decode-time lambda sweep on vanilla acts (is 100 even optimal? maybe 150-300).
2. Frozen-frontend R2 (pure ladder R2): learn lambda on vanilla activations by CRF -- decouples
   the two effects cleanly.
3. Full-song (or 60-s) crops for the CRF arm.
4. Meter set (2,3,4) variant (16 songs excluded today).
5. Multi-seed + all 8 folds before any strong claim.

## ADDENDUM (post-review): the deficit was substantially OUR harness, not the idea

Comprehensive review after the first table found three asymmetries; two measured, all fixed:

1. LR ANNEALING NEVER FIRED FOR R2. The plateau scheduler stepped on the COMBINED loss; r2's
   CRF term kept falling, so its lr stayed 1e-3 for all 30 epochs while vanilla fine-tuned at
   2e-4 from epoch 10 -- and most of vanilla's final margin accrued in its post-anneal phase.
   Fixed: both arms now anneal on the BCE component (identical signal).
2. OBSERVATION-MODEL TRAIN/DECODE MISMATCH. The CRF trained against observation_lambda=16
   (chassis default) but decode is BT-shipped 6. Probe on the full fold:
       vanilla obs=6  lam=100 : 0.9506 | obs=16 lam=100 : 0.9017
       r2      obs=6  lam=100 : 0.9446 | obs=16 lam=100 : 0.8911
       r2      obs=6  lam=14  : 0.9374 | obs=16 lam=14  : 0.9295
   Under the TRAIN-CONSISTENT obs=16 decode, the learned lambda=14 BEATS lambda=100 -- the
   learned factor was optimal for the world it was trained in. THE EARLIER "likelihood-optimal
   but F-suboptimal lambda" INTERPRETATION IS THEREFORE CONFOUNDED AND WITHDRAWN pending the
   obs=6-consistent rematch. (obs=6 remains the right deployment decode for both frontends.)
3. Epoch/data edge to vanilla (35 effective epochs incl. resume vs 30 fresh; early vanilla
   epochs overlapped cache build). Fixed: both arms fresh, 30 epochs, same seed.

Rematch launched 2026-07-18 with all three fixes (r2 chassis at observation_lambda=6, BCE-keyed
scheduler, symmetric fresh runs).
