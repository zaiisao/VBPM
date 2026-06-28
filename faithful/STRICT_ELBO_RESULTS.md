# Strict-ELBO collapse experiment — results (corrected von Mises sampler)

**Date:** 2026-06-26 (re-run) · **Branch:** `faithful/strict-elbo` · **Model:** `faithful/` (186K params)
**Setup:** strict ELBO (β=1, no bandages), end-to-end from random weights on a fixed log-mel
front-end, 4 datasets (ballroom/beatles/hains/rwc_popular), frames=256, batch=16, lr=1e-3, 800 steps,
46 held-out val songs. Free-run F-measure @70 ms (`mir_eval`); read-outs are the official
bar-pointer ones (beats = m subdivisions of φ; downbeats = 2π wraps).

---

## 0. CORRECTION — the von Mises sampler was buggy in the earlier version of this doc

The previous numbers in this file (beat-F ≈ 0.33, phase-wrap ≈ 0.42) were produced with a **broken
von Mises sampler**. `best_fisher_rejection` in `faithful/distributions.py` used a wrong acceptance
test, so the sampled phase had an essentially **constant concentration (E[cos] ≈ 0.8) regardless of
κ** instead of the correct `A(κ)=I1(κ)/I0(κ)`. The earlier audit only checked the sample *mean*
direction (which was fine), never the spread, so it slipped through.

The sampler is now **fixed and verified exact against `scipy.stats.vonmises`** across κ=0.2…20
(E[cos] matches A(κ) to <0.01). The numbers below are the **re-run with the corrected sampler**.
**The collapse conclusion is unchanged** — it was always objective/identifiability-driven, not a
sampler artifact — but the specific read-out values are now trustworthy (and, if anything, *cleaner*:
the strict-model latent read-out is ≈0.01, far below the metronome floor, rather than the spurious
0.33–0.42 the broken sampler produced). The main checkpoint of the broken run is superseded.

---

## 1. Headline

The strict ELBO, trained from random init, **posterior-collapses** — every latent's KL decays
toward ~0 while reconstruction plateaus — **and** the faithful per-frame BCE on sparse beats yields
a **dead decoder** (`decoder_F = 0.000`). With the correct sampler the latent read-outs sit **below a
120-BPM metronome**. This is the clean, honest negative result.

## 2. Posterior collapse (main run, decoder reads h)

| term | step 1 | step 800 | ratio |
|---|---:|---:|---:|
| meter KL (Categorical) | 3.06 | **0.010** | ÷306 |
| phase KL (von Mises) | 50.80 | **0.044** | ÷1155 |
| tempo KL (Log-Normal) | 355.72 | **1.609** | ÷221 |
| reconstruction (BCE) | 159.7 | 21.9 (flat from ~step 100) | — |

Eval is **flat across all checkpoints** (free-run F-measure):

| eval step | beat-phase | downbeat-phase | decoder | metronome |
|---:|---:|---:|---:|---:|
| 200 | 0.011 | 0.122 | 0.000 | 0.273 |
| 400 | 0.012 | 0.128 | 0.000 | 0.273 |
| 600 | 0.012 | 0.127 | 0.000 | 0.273 |
| 800 | **0.011** | **0.125** | **0.000** | 0.273 |

The latent read-outs (beat 0.011, downbeat 0.125) are **below the metronome (0.273)**: the collapsed
tempo never finds a musical scale, so the free-running pointer lays down no usable beat grid.

## 3. Two phenomena, two mechanisms

**(a) `decoder_F = 0` — sparse-positive BCE.** Beats are ~1.5% of frames. Faithful BCE with no
`pos_weight` and no shift-tolerance is minimized by predicting ≈0 everywhere → the decoder output
never peaks above threshold → zero recovered beats. (Confirmed by the latent-only control, §4.)

**(b) The latent itself is dead (KL ≈ 0).** The bar-pointer dynamics encode ~nothing per song; the
beat read-out is not a learned per-song grid (it is below even a fixed metronome).

## 4. Latent-only control (corrected sampler) — collapse is deeper than the h-shortcut

Same strict ELBO with `h` removed from the decoder.

| term | step 1 | step 800 |
|---|---:|---:|
| meter / phase / tempo KL | 2.58 / 50.24 / 399.9 | **0.004 / 0.038 / 2.78** |

| eval step | beat-phase | downbeat | decoder | metronome |
|---:|---:|---:|---:|---:|
| 200 | 0.355 | 0.123 | 0.257 | 0.273 |
| 800 | **0.352** | 0.119 | **0.170** | 0.273 |

**Reading:** removing the shortcut does **not** revive the latent KL (still ~0 — so the latent
collapse is optimization/identifiability-driven, not caused by the decoder shortcut). But denied `h`,
the decoder is **forced to express beats through the deterministic phase grid**, so it finally emits
beats (`decoder_F` 0.000 → 0.170) and the beat-phase read-out rises (0.011 → 0.352, now ≈ metronome).
This pins `decoder_F = 0` in the faithful run on the **h-shortcut × sparse-BCE** interaction, distinct
from the latent collapse.

## 5. Confounds

| alternative explanation | verdict | basis |
|---|---|---|
| It's the von Mises sampler bug | **refuted** | re-run with the corrected, scipy-verified sampler reproduces the same collapse |
| It's a learning-rate artifact | **refuted** | loss falls monotonically (≈570 → 22), no divergence/NaN |
| It's a code/gradient bug | **refuted** | faithfulness audit: all param groups get non-zero gradients; loss == recon + KL exactly |
| `decoder_F=0` is a threshold artifact | **supported** | the latent-only control proves a properly-incentivised decoder *does* emit beats (0.170); the h-decoder chose flat |
| the latent learned a useful grid | **refuted** | beat-phase 0.011 < metronome 0.273 and KL_phase ≈ 0.044 ⇒ no per-song structure |

## 6. Conclusion

The faithful strict-ELBO bar-pointer VAE, trained end-to-end from random weights with no frozen
frontend and the **corrected** von Mises sampler, reproduces **both** predicted failures:
(1) **posterior collapse** intrinsic to the strict objective (KL of all three latents → ~0; the
latent read-outs fall below a fixed metronome); and (2) an **unusable decoder** from faithful BCE on
sparse beats (no `pos_weight`, no shift-tolerance) — revived only when `h` is removed, isolating the
cause to the h-shortcut × sparse-BCE interaction. The collapse is a property of the **objective and
the data**, demonstrated without any frozen discriminative frontend to blame, and now on numbers free
of the sampler bug.

**Artifacts:** `runs/strict_elbo_fixed/{final,best}.pt`, `runs/control_latentonly_fixed/`;
metrics `runs/strict_elbo_fixed/metrics.jsonl`. Candidate remedies (for review) are listed in
`notebooks/CANDIDATE_DEVIATIONS.md`; the clean, self-contained reference implementation (corrected
sampler, trained on ballroom) is `notebooks/ELBO_for_DBN.ipynb`.

> **Not re-run:** the earlier *free-bits* control was not repeated with the corrected sampler (the
> faithful `train.py` intentionally has no free-bits flag). Its qualitative conclusion — free-bits
> floors the KL at the rate target without buying useful read-out (rate-without-relevance) — is a
> general result and is left as previously characterized, flagged here rather than re-measured.
