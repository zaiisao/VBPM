# Deployment-gap experiments (faithful bar-pointer VAE)

Archive of the investigation into **why the faithful bar-pointer VAE can't free-run-deploy**, and
attempts to fix it. All metrics are beat-F1 @ ±70 ms (mir_eval). Preserved here so the scratchpad
(temporary) results are not lost. Harness: `scripts/nextfix.py` (all flags below). Raw per-cell
numbers: `results/<cell>.json`. Narrative + dated log: `AUTONOMOUS_LOG.md`. Synthesis: `AUTONOMOUS_FINDINGS.md`.

## Key metrics (what to read)
- `tf_post_dec` — teacher-forced posterior **decoder** reconstruction (encoder is given the beats). Upper-bound sanity, ~1.0 when healthy. NOT a tracking score.
- `tf_post_lat` — teacher-forced **latent** read-out (beats from posterior phase). How load-bearing the latent is.
- `fr_lat` — **FREE-RUN latent read-out = the actual deployment metric.** This is the wall.
- `tf_tempo_corr` — cross-song corr of posterior tempo vs true tempo. Is tempo identified?

## Headline result
- Read-out + scorer are **sound** (ideal latents → 0.97; `ideal_readout.py`).
- The faithful model deploys as a **constant-tempo metronome, capped ~0.51**; it sits ~0.36 because it
  mis-estimates one global tempo scalar. The deployment failure is **tempo drift** (oracle injection:
  phase-resync useless, tempo-resync recovers to 0.67–0.83; `oracle_inject.py`).
- **Nothing lifted free-run `fr_lat` past ~0.41** with a healthy model: widen, β<1, free-bits, He-2019,
  overshoot, OU, survival likelihood, Tier-B audio-conditioned tempo mean, free-run reconstruction,
  scheduled sampling, word-dropout, and stop-grad tempo distillation all bounce at the floor/ceiling.
- Clean warm-start test (no collapse confound): even from a model whose **posterior tempo is accurate
  (tcorr 0.93)**, free-run stays ~0.36 — because the audio-blind prior mean can't carry that tempo into
  deployment. (A blind 7-agent panel independently reached the same conclusion.)

## Best checkpoints (in `checkpoints/`)
| ckpt | config | tf_post_dec | tf_post_lat | fr_lat | tcorr | note |
|---|---|---|---|---|---|---|
| `w6lat_oulong` | widen6 + latent_only + OU0.05, 1200 steps | 0.99 | 0.61 | 0.356 | 0.78 | best teacher-forced latent; warm-start source |
| `combo_a3s1` | + Tier B a0.3 + survival 0.1 | 0.999 | 0.65 | 0.388 | 0.06 | highest tf_post_lat |
| `dist_a8` | warm-start + Tier B a0.8 + distill | 0.99 | 0.65 | **0.400** | 0.24 | **best free-run (still ~floor)** |
| `widen6` | widen6 (h-reading) | 0.59 | 0.07 | 0.018 | — | widen baseline |
| `w6_latent` | widen6 + latent_only | 0.65 | 0.40 | 0.018 | — | latent_only baseline |
| `ws_frec05` | warm-start + free-run recon | 0.99 | 0.61 | 0.332 | 0.93 | healthiest tempo (tcorr 0.93), deploy still floor |

## Reproduce (scripts/nextfix.py flags)
`--widen 6` (target widening, ±70 ms) · `--latent_only` (latent-only emission, no h-shortcut) ·
`--ou` (OU mean-reverting tempo) · `--overshoot D` (latent overshooting) · `--survival_weight`
(renewal/IOI tempo likelihood) · `--tierb --tierb_anchor a` (audio-conditioned prior tempo mean) ·
`--freerun_weight` (free-run reconstruction) · `--ss_prob` (scheduled sampling) · `--h_dropout`
(word-dropout) · `--distill_weight` (stop-grad distill posterior tempo → prior head) · `--init_from CK`
(warm-start). Eval-only diagnostics: `oracle_inject.py`, `tempo_const_test.py`, `task4_fourway.py`, `pr_eval.py`.

## Open path (route 2 — true trainable DBN)
The pragmatic `h→φ̇` (VRNN readout) route bounces at ~0.40 and corrupts the tempo. The faithful path keeps
tempo a **pure latent** + activation as observable and fixes the **inference** (sequential/particle / FIVO)
so tempo is identified from the activation like the DBN's exact forward-backward — instead of amortized
one-step VI, which provably can't. That is the next build.
