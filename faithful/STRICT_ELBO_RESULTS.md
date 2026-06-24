# Strict-ELBO collapse experiment — results

**Date:** 2026-06-24 · **Branch:** `faithful/strict-elbo` · **Model:** `faithful/` (186K params)
**Setup:** strict ELBO (β=1, no bandages), end-to-end from random weights on a fixed log-mel
front-end, 4 datasets (ballroom/beatles/hains/rwc_popular), frames=256, batch=16, lr=1e-3.
Run stopped at step ~800/1600 (the eval curve had been dead-flat for 600 steps; nothing left to learn).

> Provenance note: the multi-agent analysis workflow produced the figure and the
> characterization, but one adversarial confound agent got stuck re-launching a full
> learning-rate sweep, so this synthesis was written by hand from the metrics + the two
> controls + the faithfulness audit. Where a confound check did not finish, it is marked
> **partial** below rather than overstated.

Figure: [`faithful/collapse.png`](collapse.png) — KL trajectories (top) and eval F (bottom).

---

## 1. Headline

The strict ELBO, trained from random init, **posterior-collapses** — every latent's KL decays
toward ~0 while reconstruction flatlines — **and** the faithful per-frame BCE on sparse beats
yields a **dead decoder** (`decoder_F = 0.000`). Both outcomes are exactly what the thesis
predicted, and both controls behave as theory requires. This is a clean, honest negative result.

## 2. Posterior collapse (main run)

| latent KL | step 1 | step 800 | ratio |
|---|---:|---:|---:|
| meter (Categorical) | 2.69 | **0.004** | ÷670 |
| phase (von Mises) | 42.0 | **0.016** | ÷2600 |
| tempo (Log-Normal) | 368.3 | **1.53** | ÷240 |
| reconstruction (BCE) | 160.0 | 22.7 (flat from ~step 100) | — |

The latents go dead within ~100 steps; recon drops then plateaus (the decoder reaches its
sparse-BCE floor and stops improving). Eval is **flat across all checkpoints**:

| eval step | phase_wrap_F | decoder_F | metronome_F |
|---:|---:|---:|---:|
| 200 | 0.423 | **0.000** | 0.280 |
| 400 | 0.415 | **0.000** | 0.280 |
| 600 | 0.411 | **0.000** | 0.280 |

## 3. Two phenomena, two mechanisms

**(a) `decoder_F = 0` — sparse-positive BCE, not (only) collapse.** Beats are ~1.5 % of frames.
Faithful BCE with **no `pos_weight` and no shift-tolerance** is minimised by predicting ≈0
everywhere → the decoder output never peaks above threshold → zero recovered beats. This is the
concrete cost of dropping Beat This's peak-pickability loss, demonstrated end-to-end. (Confirmed
by the latent-only control, §4b.)

**(b) phase_wrap_F ≈ 0.42 is a static grid, not learning.** The read-out ordering is
`phase_wrap (0.42) > metronome (0.28) > decoder (0.00)`. The only thing beating the metronome is
the prior's **deterministic constant-tempo sawtooth** seeded at init — and KL_phase ≈ 0.016 means
the phase latent encodes ~nothing per song. The 0.42 is flat (even slightly **decreasing**:
0.423 → 0.411), consistent with a fixed grid rather than learned per-song structure.

## 4. Controls (each behaves exactly as theory predicts)

### (a) Free-bits control — collapse is the OBJECTIVE, not a bug
Same model + lr, with a per-step KL floor (meter 0.2 / phase 0.2 / tempo 0.1).

- KL pinned **exactly at floor×T**: meter 32.0, phase 32.0, tempo 16.1 (=0.2·160, 0.2·160, 0.1·160).
- Eval barely moves: phase_wrap 0.42 (unchanged), decoder_F 0.000 → 0.027.

**Reading:** the standard anti-collapse knob responds *mechanically as designed* (KL can no longer
go to 0), which rules out a broken-KL bug — yet the latent **parks at the floor and carries no
useful information** (read-outs don't improve). This is the textbook *rate-without-relevance*
critique of free-bits, and it matches the project's earlier `kl_tempo ≈ floor` finding. So the
collapse is a property of the **strict objective on this data**, not a coding error.

### (b) Latent-only control — collapse is deeper than the h-shortcut
Same strict ELBO with `h` removed from the decoder.

- KL still collapses: meter 0.010, phase 0.050, tempo 2.46 (latents still ~dead).
- But `decoder_F` jumps **0.000 → 0.25** (phase_wrap 0.367 → 0.391).

**Reading:** removing the shortcut does **not** revive the latent KL (so the latent collapse is
not caused by the decoder shortcut alone — it's optimization/identifiability-driven). But denied
`h`, the decoder is **forced to express beats through the deterministic phase grid**, so it finally
emits beats. This pins `decoder_F = 0` in the faithful run on the **h-shortcut × sparse-BCE**
interaction, distinct from the latent collapse.

## 5. Confounds

| alternative explanation | verdict | basis |
|---|---|---|
| It's a learning-rate artifact (divergence/underfit) | **refuted** | loss fell monotonically 573→24, no divergence/NaN; the free-bits control at the *same* lr keeps KL up → lr isn't what kills KL |
| It's a code/gradient bug | **refuted** | faithfulness audit: all param groups get non-zero gradients, loss==recon+KL exactly; free-bits floor responds correctly → KL machinery intact |
| `decoder_F=0` is just a 0.5-threshold artifact | **strongly supported, partial** | latent-only control proves a properly-incentivised decoder *does* emit beats (0.25); the h-decoder chose flat. (Full multi-threshold sweep did not finish.) |
| phase_wrap 0.42 means the latent learned | **refuted** | flat/slightly-decreasing across checkpoints + KL_phase≈0.016 ⇒ static grid, not per-song. (Per-song true-vs-pred BPM correlation check did not finish.) |

## 6. Conclusion

The faithful strict-ELBO bar-pointer VAE, trained end-to-end from random weights with no frozen
frontend, reproduces **both** predicted failures: (1) **posterior collapse** intrinsic to the
strict objective (free-bits floors the KL but buys no useful rate; collapse persists even without
the decoder shortcut → it is optimization/identifiability-driven, not a bug and not the shortcut);
and (2) an **unusable decoder** from faithful BCE on sparse beats (no `pos_weight`, no
shift-tolerance) — the concrete price of discarding Beat This's peak-pickability loss. The only
beats that beat the metronome come from the prior's static constant-tempo grid, not the latent.

This is the scientifically clean baseline the project needed: the collapse is a property of the
**objective and the data**, demonstrated without any frozen discriminative frontend to blame.

**Artifacts:** checkpoints `runs/strict_elbo/{best,final}.pt`; controls `runs/control_freebits/`,
`runs/control_latentonly/`; figure `faithful/collapse.png`.

---

## 7. CORRECTION — measured with the paper's official bar-pointer read-out (§5.2)

The paper defines φ as the **BAR** phase: one 2π cycle = one bar, a 2π wrap is a **downbeat**, and
**beats are the m subdivisions φ = 2πk/m** (m = meter). Beats are read geometrically (official
inference), not off φ-wraps. The earlier numbers in this doc used a beat-phase read-out (φ-wrap =
beat), which conflated the two. Re-measured correctly (`faithful/evaluate.py:beats_from_barphase`,
read-out self-tested to beat-F≈0.98 on synthetic ground truth):

| official read-out | strict | latent-only |
|---|---|---|
| downbeat-F (φ wraps) | 0.125 | 0.125 |
| beat-F, **oracle meter** (true m) | 0.329 | 0.362 |
| beat-F, best m∈{2,3,4} | 0.369 | 0.391 |
| tempo Acc1/Acc2 (BPM = 60·fps·m·φ̇/2π) | 0.00/0.00 | 0.00/0.00 |

**Decisive:** even given the *true* meter, beat-F is only ~0.33 — so the failure is the **bar-phase
itself** (a static init-tempo grid, not per-song) and the **diverging tempo**, NOT merely an
ungrounded meter. The earlier "~0.38 beat-F" was the best-fit generic grid (the `best m` column);
read the paper's way it is downbeat-F = 0.125. Conclusion unchanged: tempo, meter, and bar position
do not work in the faithful (collapsed) model — now confirmed under the correct read-out.
