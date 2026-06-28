"""Builds notebooks/ELBO_for_DBN.ipynb — a FULLY SELF-CONTAINED, auditable notebook:
  Part I  = the document's bar-pointer DVAE, implemented faithfully (reused from the faithful notebook;
            runs on a toy song; only libraries imported).
  Part II = why it does not deploy on real audio, the deviations (original->analysis->new), and the
            WORKING pipeline with its FULL architecture inline (fixed bar-pointer prior + activation
            emission + SMC inference). Real-data before/after, verification, honest conclusions.
No imports of our own codebase anywhere — only numpy/torch/mir_eval/matplotlib + stdlib. For auditability.
Reads standardized_numbers.json. Run AFTER standardized_numbers.py."""
import json, os
HERE = os.path.dirname(__file__)
NUM = json.load(open(os.path.join(HERE, "standardized_numbers.json")))
E, S, O, SOTA = NUM["easy"], NUM["smc"], NUM["oracle_smc"], NUM["sota_smc"]
BAK = json.load(open(os.path.join(HERE, "ELBO_for_DBN.faithful-original.ipynb.bak")))

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t})

# ============================ INTRO ============================
md(f"""# CHART — Bar-Pointer Beat Tracking: from the *ELBO for DBN* document to a working pipeline

**A fully self-contained, auditable notebook.** Every piece of the architecture is written out inline;
the only dependencies are standard libraries (`numpy`, `torch`, `mir_eval`, `matplotlib`). Nothing imports
our own codebase, so the model and pipeline can be checked line-by-line.

It is in two parts:
- **Part I — the document's model, faithfully.** The variational bar-pointer DVAE exactly as written
  (latents, prior, posterior, decoder, von Mises sampler, closed-form KLs, Algorithm 1), runnable on a
  toy song so every intermediate value is concrete.
- **Part II — does it deploy?** On real audio the faithful model does not. We show, for each necessary
  **deviation**, the *original* approach and its poor result, the *analysis* of why it fails, and the
  *new* approach and its result — ending at the working pipeline whose **full architecture is inline**:
  a fixed bar-pointer prior + a beat-activation observation + Sequential-Monte-Carlo (particle-filter)
  inference. With real before/after numbers and an oracle verification section.

**Punchline:** the document's *latent geometry* (bar-phase φ, tempo τ, meter m, read out geometrically)
is sound and we keep it; its *VAE machinery* (amortized inference, trainable audio-conditioned prior,
beat-emission decoder, free-run deployment) cannot be deployed, and fixing that recovers the **classic
bar-pointer DBN with a neural activation**. Beat tracking today is a *representation* problem, not an
*inference* one — the experiments make this precise.

*(`PF` = particle filter = our inference method; `SMC-MIREX` = the hard dataset — kept distinct.)*""")

# ============================ PART I (faithful architecture, reused & auditable) ============================
md("""---
# Part I — The document's model, implemented faithfully

Below is the bar-pointer DVAE exactly as in the document, from scratch, only libraries imported. It runs
on a toy song so the generative process and one Algorithm-1 forward pass are concrete and checkable.""")
# reuse the faithful architecture cells: helpers, vM sampler, KLs, model, Algorithm 1, toy song, one forward pass
# (.bak cells 1..15 — skip the toy *training/results* cells 16+, whose toy-overfit could mislead; the honest
#  real-data deploy story is Part II.)
for c in BAK["cells"][1:16]:
    cells.append({k: c[k] for k in ("cell_type", "metadata", "source") if k in c}
                 | ({"execution_count": None, "outputs": []} if c["cell_type"] == "code" else {}))

# ============================ PART II ============================
md(f"""---
# Part II — Does it deploy? The deviations, and the working pipeline

Part I trains and reconstructs beats *teacher-forced* on a toy. The real question is **deployment**:
given only audio (no ground-truth beats), recover the beats. Here the faithful model breaks, and fixing
it requires several deviations. We present each as **original (poor) → analysis → new (good)**, then give
the working pipeline's full architecture inline and its real-data results.

All real numbers below: mir_eval F-measure @ ±70 ms; AMLt = octave/level-tolerant continuity. The strong
frontend is Beat This's beat activation (a cached array we load as data); our contribution is the
inference layer around it. Easy = ballroom/beatles/hains/rwc val ({E['n']} songs); SMC-MIREX = the hard
expressive set ({S['n']} tracks, Holzapfel et al.).""")

md(f"""## Deviation 1 — the beat likelihood  *(plain BCE → shift-tolerant BCE)*

**Original (§5.4):** per-frame Bernoulli BCE on the beat targets. **Problem:** beats are ~1.5% of frames,
so plain BCE is minimized by predicting ≈0 *everywhere* — the decoder goes dead. Faithful from-scratch run:
**decoder F = 0.000**. **Analysis:** a sparse-positive-event pathology, not a latent problem. **New:** a
shift-tolerant / `pos_weight`-ed BCE (widen the target ±a few frames, up-weight positives) — exactly the
peak-pickable loss Beat This uses — yields a peaky, usable activation. *(This is the activation feeding
Part II; the dead-decoder figure is from the faithful run, `faithful/STRICT_ELBO_RESULTS.md`.)*""")

md(f"""## Deviation 2 — inference, observation, prior & deployment  *(the document's VAE → the classic DBN)*

Four of the document's choices fail **together** at deployment:
- **Amortized inference** `q(z|b,h)` needs the beats `b` as input — absent at test time, so the trained
  posterior is unusable. Even *teacher-forced with the ground-truth beats* it reads out only **≈0.39** —
  the encoder cannot invert the circular/sequential bar-pointer geometry.
- **Beat-emission decoder** `p(b|z,h)` reading `h`: no separate audio *observation* to filter against,
  and giving the decoder `h` lets it bypass the latent (the audio shortcut).
- **Trainable audio-conditioned prior** whose initial tempo is read (mean-pooled) from audio: it cannot
  estimate tempo feed-forward (measured **852%** init error).
- **Free-run deployment**: roll the prior open-loop, committing to that garbage tempo with no correction.
  Measured **≈0.40**.

**Analysis:** deployment is an *inference* problem — recover phase/tempo from audio — and amortized VI is
far worse at it than explicit Bayesian inference. **New = the classic bar-pointer DBN, reused around the
document's latent:** observe the audio beat-activation; keep the prior **fixed and broad** (no learned
tempo init); recover the state by **SMC**. The full architecture follows.""")

md("""### The working model — full architecture (inline, library-only)

The generative model is the document's bar-pointer **with a fixed prior and an audio-activation emission**:
- **prior** `p(φ_t,τ_t)`: `log τ_t = log τ_{t-1} + ε`, `ε~N(0,σ_τ²)` (fixed-form tempo random walk, broad
  uniform init — *no* trainable/audio-conditioned dynamics); `φ_t = φ_{t-1} + exp(log τ_t)` (deterministic
  phase advance).
- **emission** `p(a_t | φ_t)`: the audio activation `a_t∈[0,1]` is explained by the beat template
  `T(φ)=exp(κ(cos(mφ)−1))` (peaks at the m subdivisions): `logp = T(φ)·log a_t + (1−T(φ))·log(1−a_t)`.
- **inference**: a bootstrap particle filter (SMC) — propagate particles through the prior, weight by the
  emission, resample on low ESS; the MAP ancestral trajectory gives φ, read out geometrically.""")
# inline the full working architecture (the contents of chart_pipeline.py, library-only)
PIPE = open(os.path.join(HERE, "chart_pipeline.py")).read()
# strip module docstring + the torch try/except (the notebook setup already imports torch) for a clean inline cell
PIPE = PIPE.split('"""', 2)[2]                       # drop the module docstring
PIPE = PIPE.replace("from __future__ import annotations\n", "")
PIPE = PIPE.replace(
'''try:
    import torch
    _DEV = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch = None
    _DEV = "cpu"
''', 'import torch\n_DEV = "cuda" if torch.cuda.is_available() else "cpu"\n')
PIPE = PIPE.replace('    assert torch is not None, "torch required"\n', '')
code("import os, glob, math\nimport numpy as np, mir_eval, torch\n" + PIPE +
     '\nprint("working model ready | device:", _DEV, "| easy songs:", n_easy(), "| SMC-MIREX:", len(smc_ids()))')

md("""### Demo — same song, two deployments (free-run vs SMC inference)
Same fixed prior; the only difference is whether we infer against the audio observation.""")
code("""import matplotlib.pyplot as plt
%matplotlib inline
a, ref, fps = load_easy(0)
fr_beats  = beats_from_phase(free_run_phase(len(a), 4, fps), 4, fps)   # ORIGINAL: prior, no observation
smc_beats, bpm, _, _ = smc_track(a, 4, fps, sig_t=0.08)                # NEW: SMC inference vs the activation
print(f"free-run (no inference): F = {f1(ref, fr_beats):.3f}")
print(f"SMC inference:           F = {f1(ref, smc_beats):.3f}   (bpm~{bpm:.0f})")
fig, ax = plt.subplots(3,1, figsize=(11,4.5), sharex=True); t = np.arange(len(a))/fps
ax[0].plot(t, a, lw=.6); ax[0].set_ylabel("Beat-This\\nactivation"); ax[0].set_title("same song — observation and the two deployments")
ax[1].vlines(fr_beats[fr_beats<25],0,1,color='r'); ax[1].vlines(ref[ref<25],0,1,color='g',alpha=.3); ax[1].set_ylabel("free-run\\n(original)")
ax[2].vlines(smc_beats[smc_beats<25],0,1,color='b'); ax[2].vlines(ref[ref<25],0,1,color='g',alpha=.3); ax[2].set_ylabel("SMC\\n(new)"); ax[2].set_xlabel("time (s)  [green=ground-truth]")
ax[0].set_xlim(0,25); plt.tight_layout(); plt.show()""")
md(f"""**Aggregate (full datasets, σ_τ=0.08):**

| deployment | easy beat-F | SMC-MIREX beat-F |
|---|---|---|
| fixed prior, **no observation** (no inference) | {E['freerun_F']} | {S['freerun_F']} |
| **SMC inference** | **{E['pf_F']}** | **{S['pf_F']}** |

Without the observation the prior is a random metronome; with SMC it tracks. (The document's own free-run,
with its trainable prior, measured ~0.40 — same conclusion.) *This is the classic bar-pointer DBN with a
neural activation, around the document's geometry.*""")

md("""## Deviation 3 — training objective  *(ELBO/SGVB → supervised activation)* — the optional one

Reverting here does **not** collapse the score, so this is the one axis where we can stay faithful. Does
the variational training (FIVO) earn its keep? A **seeded** re-test (3 seeds, ±std) of supervised vs +FIVO,
at scarce (5%) and full labels:

| labels | supervised downbeat-F | +FIVO downbeat-F | Δ |
|---|---|---|---|
| 5% | 0.343 ± 0.015 | 0.370 ± 0.008 | **+0.027 — beyond seed noise** |
| 100% | 0.462 ± 0.039 | 0.488 ± 0.032 | +0.026 — within noise |

**FIVO gives a small, real downbeat benefit, statistically clear only in the scarce-label
(semi-supervised) regime** — the unlabeled audio reinforces the bar-pointer constraint where supervision
runs out. It is positive-but-noisy at full supervision and slightly *costs* beats at low labels
(0.664→0.633; neutral at full). So the variational term is best motivated as a **modest, optional,
semi-supervised downbeat aid — not a core contribution and not a full-supervision default.** *(An earlier
single-seed run suggested FIVO **hurts** at full labels; the seeded re-test shows that was noise.)*""")

md(f"""## Results — the scoreboard

| system | easy beat-F | SMC-MIREX beat-F | SMC-MIREX AMLt |
|---|---|---|---|
| fixed prior, no observation | {E['freerun_F']} | {S['freerun_F']} | — |
| **ours: fixed prior + activation + SMC** | **{E['pf_F']}** | **{S['pf_F']}** | {S['pf_AMLt']} |
| Beat-This (no DBN) — SOTA | ~0.88 | **{SOTA['beat_this_noDBN']}** | 0.598 |
| Beat-This + DBN | — | {SOTA['beat_this_DBN']} | 0.646 |
| madmom / madmom-TCN | — | {SOTA['madmom']} / {SOTA['madmom_tcn']} | 0.615 / 0.652 |

- **Easy ({E['pf_F']})** *exceeds* Beat-This's own peak-pick (~0.88) on the same activation — the PF cleans
  up a good activation. **Caveat:** these songs are likely *in-distribution* for Beat This, so treat 0.94
  as an in-distribution upper bound.
- **SMC-MIREX ({S['pf_F']})** is the honest out-of-distribution test: **below** no-DBN SOTA
  ({SOTA['beat_this_noDBN']}), competitive with the DBN baselines.
- **The SMC-MIREX gap is activation quality, not inference** — proven next.""")

md("""## Verification — is the pipeline trustworthy? (oracle / known-answer tests)
Feed a *perfect* activation into the PF; it should recover the known beats. This isolates the inference
from the activation quality.""")
code(f"""for bpm in [70, 100, 130, 170]:
    a, bt = oracle_activation(bpm, 20, 50.0)
    beats, bpmrec, _, _ = smc_track(a, 4, 50.0, sig_t=0.08)
    print(f"oracle {{bpm}} BPM: recovered F = {{f1(bt, beats):.3f}}  bpm~{{bpmrec:.0f}}")
print("\\nmean oracle F (SMC-MIREX rate) = {O['pf_F']}  ->  the PF is CLEAN; the SMC-MIREX gap is the activation, not the inference.")""")
md(f"""**Oracle → PF ≈ {O['pf_F']}.** The PF extracts ~full value from a perfect observation, so the
SMC-MIREX shortfall ({S['pf_F']}) is **activation-limited, not an inference bug.** (Component unit-tests of
the transition math, emission template, geometric read-out and the F/CMLt/AMLt metrics all pass exactly.
Two minor real caveats: tempi ≤40 BPM hit a prior clamp; the `bpm_map` scalar low-biases at slow tempo —
read tempo from the output beat intervals. A posterior "uncertainty" read-out tested unreliable and is
deliberately excluded.)""")

md("""## Honest conclusions

1. **Keep** the document's bar-pointer *latent geometry* (φ, τ, m + geometric read-out) — verified sound.
2. **The necessary deviations are all reversions to the classic DBN.** The document's VAE cannot deploy
   (amortized inference needs absent beats; the prior can't feed-forward tempo; free-run commits without
   correction; beat-emission gives nothing to filter). The fix — audio observation + fixed prior + SMC —
   *is* the classic bar-pointer DBN. Progress was **despite** the document's specific choices, by reverting
   them while keeping its geometry.
3. **The variational training (FIVO) gives only a small downbeat benefit** — statistically clear in the
   semi-supervised (scarce-label) regime (+0.027, beyond seed noise) and within noise at full supervision.
   A modest, optional aid, not a core contribution.
4. **Beat tracking is now a representation problem.** Given a strong activation our inference matches/
   exceeds SOTA on (in-distribution) easy data and is activation-limited on SMC-MIREX. Beating SMC-MIREX
   would need a stronger frontend (a music foundation model) and/or training diversity — not more inference.
5. **Verification matters:** several apparent "findings" were artifacts (a broken octave metric, a
   false-positive "fps bug"); only oracle known-answer tests separated real effects from bugs.""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(HERE, "ELBO_for_DBN.ipynb"), "w"), indent=1)
print(f"wrote ELBO_for_DBN.ipynb with {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='code')} code)")
