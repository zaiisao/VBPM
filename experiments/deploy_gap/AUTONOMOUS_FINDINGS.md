# Autonomous diagnostic session — findings (2026-06-25, ~04:45–06:30 KST)

Ran the blind-agent Tier 0–5 methodology overnight. All experiment scripts + raw logs are in
this scratchpad (`AUTONOMOUS_LOG.md` has timestamped detail). One authorized fix was committed to
`faithful/`; everything else is temporary/scratchpad.

## TL;DR — the failure is now precisely located

The faithful bar-pointer VAE does **not** fail because the read-out is broken (it isn't) or
because the architecture can't express beats (it can). The real finding is sharper and more
damning than "posterior collapse":

**THE STRUCTURED LATENT IS NEVER USED. The decoder reconstructs beats from the audio feature h
alone; the bar-pointer phase/tempo/meter latent earns nothing.** Evidence, each demonstrated:

1. **The model never fits the full data** — teacher-forced posterior decoder F = **0.000**. The
   widely-quoted "free-run beat-F 0.33" is *higher than the teacher-forced posterior* (0.024),
   so 0.33 is **not inference** — it is a learned-tempo metronome catching ~1/3 of beats by
   periodicity. "Posterior collapse" is a misnomer; the **decoder fails to generalize**.
2. **The architecture CAN express beats — but only via the decoder/h, not the latent.** Overfitting
   a single song drives recon→0 and teacher-forced **decoder** beat-F → **1.000** (at both lr 1e-3
   and 1e-2). Yet the teacher-forced **latent** (phase subdivision) read-out is **0.000 in every
   eval**, even on that one memorized song. So capacity exists, but it lives entirely in the
   audio-reading decoder; the phase latent is inert.
3. **The latent never learns the true dynamics, even on clean known-truth data** — on synthetic
   click-track audio with *known* tempo/meter: **free-run = 0.000** at every step and the recovered
   phase never matches the planted phase (circular corr stays ≈0.17–0.24, once negative). The
   decoder fits the clicks (recon→0) via h; the generative rollout and the phase latent do not.

Plus two mechanisms confirmed:
- The **1e9-BPM tempo blowup is the unbounded log-random-walk prior behaving exactly as defined**
  (empirical Var[log τ] equals the cumulative-σ² prediction, ratio 1.04).
- The unbounded RW tempo is also an **optimization hazard**: at lr 1e-2 the overfit run reached
  decoder-F 1.0 then DIVERGED (tempo-KL exploded 13→1210, recon blew back up). Not just a
  deployment problem.

## Evidence table

| Experiment | Result | Conclusion |
|---|---|---|
| Read-out known-answer (ideal latents) | beat 0.976 / downbeat 0.959 | read-out + mir_eval are SOUND |
| **Four-way F (strict ckpt)** | tf_post_dec **0.000**, tf_post_lat 0.024, fr_lat 0.328 | model never fit; 0.33 is periodicity, not inference |
| Four-way F (overshoot ckpt) | tf_* all 0.000, fr_lat 0.404 | overshoot only improved the metronome |
| **Overfit one song** | TF-dec-F **1.000** (both lr); TF-lat-F **0.000** always | decoder can fit via h; phase latent inert |
| **Synthetic-truth** | **free-run 0.000** always; phase-corr ~0.2 (never recovers) | latent never learns true dynamics, even clean |
| Decoder 2×2 | full dec_dz 0.035 / dec_dh 0.059 (both tiny); latent_only dec_dz 0.415 | decoder is DEAD, not audio-shortcutting |
| Tempo variance growth | emp Var 49.6 ≈ RW pred 47.7 (ratio 1.04) | blowup = unbounded RW prior as-defined |
| He-2019 (axis-1 fix) | posterior 0.046, free-run 0.316 | rescuing encoder doesn't help deployment |
| Overshoot D=4 (axis-2 fix) | beat 0.40, but σ 0.244→**6.69** | helps metronome; inflates stochastic tempo prior |

## Corrections to earlier beliefs (this session overturned two)

- **"Overshoot trains σ down."** WRONG. It pushed prior tempo σ from 0.244 to **6.69** (Task 6).
  KL(stop_grad q ‖ multi-step prior) with a parameter-free RW mean can only cover the d-step-ahead
  posterior by *inflating* σ. The ablation's bpm_std "improvement" (147→46) was the deterministic
  MEAN chain, which is blind to σ — the frozen-mean-metronome caveat biting exactly as warned.
- **"0.33 free-run is the model partially tracking beats."** WRONG. It is below the teacher-forced
  posterior; it is periodicity, not tracking. The honest deployment number for *inference* is ~0.

## What this implies for whether the model class can work

- **Sufficiency (can it express?): YES** — proven by overfit-one-song.
- **Necessity / does it earn its complexity: NOT on current evidence** — on clean known-truth data
  the generative rollout fails (free-run 0.000) and the latent doesn't learn the true phase, while
  a strong frontend + peak-pick already wins (prior joint-eval verdict). The structured latent is
  not carrying the beat information; the decoder memorizes via teacher forcing.

The two binding problems are now named and separable, and BOTH must be fixed for deployment:
- **(A) the decoder/likelihood never fits across diverse data** (majority-class collapse on
  ~1.5%-positive single-frame Bernoulli targets) — points at the likelihood spec (shift-tolerant /
  widened / point-process target), not the latent;
- **(B) the prior rollout is not trained to be a generator and the tempo prior is improper for
  length-T rollout** (unbounded RW) — points at a mean-reverting (OU) tempo prior and a multi-step
  rollout-consistency objective that trains the prior MEAN (overshoot as-is only touches σ/κ, and
  inflates σ).

## Recommended next steps (NOT executed — for the user to choose)

1. Likelihood fix for (A): widened/shift-tolerant beat target (Beat-This-style) or a point-process
   likelihood; re-check whether tf_post_dec rises above 0 on full data.
2. Prior fix for (B): OU/mean-reverting tempo prior (bounds Var) + a rollout objective that puts a
   gradient on the prior MEAN (current overshoot cannot — means are parameter-free).
3. The decisive sanity gate going forward: report **tf_post_dec** (teacher-forced posterior decoder
   F) as the primary training-health metric, NOT free-run F — free-run F is inflated by periodicity.

## UPDATE (09:5x) — new-baseline grid + Tier-1 latent_only: problem narrowed to multi-step rollout

Built two fixes as a factorial (scratchpad/nextfix.py): (A) widen beat target ±W frames
(shift-tolerant), (B) OU/mean-reverting tempo prior. Primary metric tf_post_dec (baseline 0.000).

- **Fix A (widen) is the lever:** tf_post_dec 0.000 → 0.59–0.62 (w5/w6). OU alone does nothing (0.000).
  Eval tolerance is mir_eval ±70 ms = 6.03 frames; widen6 (±70 ms) matches it = the principled setting.
- **But on the h-reading decoder the latent stays inert** — fr_dec ≈ tf_post_dec ≈ 0.62, tf_post_lat ≈ 0
  → just a (weak) discriminative frontend.
- **Tier-1 `latent_only` (remove the audio shortcut) makes the latent LOAD-BEARING:**
  tf_post_lat 0.00/0.07 → **0.40** (w6_latent / w6_latent_ou), tf_post_dec stays 0.65–0.67.
  First time the structured latent carries beats.
- **Deployment still fails, and is now pinned:** four-way on w6_latent = tf_post_dec .654,
  tf_post_lat .399, **tf_prior_lat .476** (one-step prior GOOD), fr_dec .276, **fr_lat .018**
  (multi-step free rollout BAD). One-step prior works; the unrolled chain compounds errors and
  desyncs. The binding problem is now PRECISELY **multi-step prior-rollout consistency** (Hafner/PlaNet
  gap) — and latent overshooting now has a working latent to roll (it was inert before). OU keeps
  free-run at the periodicity floor (.33) instead of collapsing (.018) but does not track.

NET PROGRESS this session: (A) decoder majority-class collapse — FIXED (widen). (latent inert) —
FIXED (latent_only). (B) multi-step deployment — REMAINS, now cleanly isolated and literature-mapped
(latent overshooting / DVBF / KVAE / VRNN audio-conditioned prior). Recommended training baseline:
widen target to ±70 ms + latent_only.

## TIER-A RESULTS (overshoot + OU on widen6+latent_only) — deployment gap NOT closed

Goal: lift fr_lat (multi-step free-run latent F1) off the 0.018/0.33 floor toward the one-step
ceiling (tf_prior_lat~0.48-0.63). Baseline w6_latent: dec .654 / lat .399 / fr_lat .018.

| cell | tf_post_dec | tf_post_lat | fr_lat | note |
|---|---|---|---|---|
| w6_latent (baseline) | 0.654 | 0.399 | 0.018 | one-step prior 0.476 |
| w6lat_oulong (OU.05, 1200 steps) | 0.993 | 0.607 | 0.356 | one-step prior **0.626**; fr_dec 0.359 |
| w6lat_ou10 (OU.10) | 0.556 | 0.329 | 0.344 | safe, no lift |
| w6lat_ou20 (OU.20) | 0.000 | 0.131 | 0.372 | collapsed |
| w6lat_os4 / os8 / os4fn | 0.000 | 0.000 | ~0.33 | overshoot COLLAPSES (starves recon) |
| w6lat_os4ou | 0.000 | 0.000 | 0.000 | total collapse |

**VERDICT: nothing lifted fr_lat above the ~0.33 periodicity floor.** Even the best model (oulong:
near-perfect teacher-forced dec 0.99, latent 0.61, one-step prior 0.63) free-runs at 0.354. The
one-step→multi-step gap (0.63→0.35) is the deployment failure and it is UNTOUCHED by Tier A.
- Overshoot only trains sigma/kappa (phase mean is parameter-free) AND at w=1.0 starves recon → collapse.
- OU bounds tempo variance but mean-reverts to a CONSTANT C, not the song's tempo → no tracking.
Faithful, audio-blind prior fixes are insufficient for multi-step deployment.

## EXP 0.1 (oracle injection) — WHY Tier A fails, and what Tier B must do
Periodically resyncing the free-run state from GT: phase-only (arm A) does NOTHING (0.019→0.031);
phase+TEMPO (arm B) recovers hugely (→0.674 @4s, →0.833 @2s). **The binding failure is TEMPO drift**,
not phase drift. Implication: Tier B's audio-conditioned correction must target the prior TEMPO mean
(g_tau), not just phase — a phase-only innovation prior would fail.

## STATUS: Tier B (audio-conditioned prior TEMPO mean) is now the evidence-justified next lever
(faithfulness-crossing — user decision). Also running: EXP 1.3 renewal/survival IOI likelihood
(faithful attempt to make the tempo latent accurate, which 0.1 says is the lever) + a plain 1200-step
control. Results pending.

## Authorized change committed
- `faithful/evaluate.py` (commit 22a7945): fixed the mis-scaled phase read-out — beats now read at
  the m-subdivision rate scored vs beats; 2π wraps scored vs downbeats; meter estimated from GT.
  Verified on ideal latents (0.976 / 0.959).
