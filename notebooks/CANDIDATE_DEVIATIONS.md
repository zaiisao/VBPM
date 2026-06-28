# Candidate deviations from the faithful model — for manual review

The notebook `ELBO_for_DBN.ipynb` is the **strictly faithful** bar-pointer VAE: strict ELBO
(β=1), three latents, closed-form KLs, the von Mises sampler, the bar-pointer prior, a Bernoulli
decoder, trained end-to-end from random weights on a fixed log-mel front-end, **no extra tricks**.

Measured outcome on ballroom (600 steps, corrected sampler), free-run F-measure @70 ms:

| read-out | F |
|---|---|
| metronome (120-BPM floor) | 0.269 |
| downbeat (bar wraps of φ) | 0.154 |
| **beat (m subdivisions of φ)** | **0.000** |
| decoder (Bernoulli peak-pick) | 0.003 |

All three per-latent KLs decay to ~0 (posterior collapse). The latent read-outs lose to a
metronome. Below are the minimal changes that would each address one failure mode. **None are in
the notebook.** Each is classified so you and your professor can decide what (if anything) to admit.

Legend — **C** = correctness fix (no method change; safe to apply); **M** = method deviation
(changes the model/objective/deploy; needs sign-off).

---

## 0. Correctness fix already applied in the notebook (C)
**von Mises Best–Fisher sampler.** The `faithful/` source had a wrong acceptance test, so sampled
concentration was ~constant (E[cos]≈0.8) regardless of κ. The notebook uses the corrected,
scipy-verified sampler. This is a bug fix, not a deviation. **Note:** `faithful/distributions.py`
itself is *not yet patched* — patching it changes the committed package and invalidates
`faithful/STRICT_ELBO_RESULTS.md`, so it is left for your decision.

---

## 1. Posterior collapse → KL floor / annealing / richer posterior (M)
- **Symptom:** KL_meter/phase/tempo all → ~0; the latent stops encoding the song.
- **Minimal fix:** free-bits (a per-latent KL floor, e.g. 0.1–0.2 nats) or β-annealing.
- **Cost to faithfulness:** changes the objective (no longer the exact ELBO with β=1). Known to
  raise the *rate* but not always the *relevance* (latent may park at the floor without becoming
  useful), so it should be reported with a read-out, not just the KL.
- **Stronger alternative:** a more expressive / aggressively-trained posterior (lagging-encoder,
  semi-amortized) — larger change, same review gate.

## 2. Tempo never finds a musical scale → anchored / bounded tempo prior (M)
- **Symptom:** the free-run tempo sits near the uninformed init (≈1 rad/frame ⇒ absurd BPM), so
  the geometric beat grid is meaningless ⇒ beat read-out 0.000.
- **Minimal fix:** initialize/anchor the tempo prior mean to a musical range (≈ log of the phase
  advance for ~60–200 BPM at the bar rate), or add the bounded random walk (a hard [φ̇min, φ̇max]).
- **Cost to faithfulness:** changes the prior. The bounded RW is a documented choice in the
  bar-pointer literature; a musically-biased init is a softer version of the same idea.

## 3. Dead Bernoulli decoder → class-weighted / shift-tolerant loss (M)
- **Symptom:** beats are ~1.5% of frames; plain per-frame BCE is minimized by predicting "no
  beat" ⇒ decoder ≈ silent.
- **Minimal fix:** `pos_weight` on the BCE (≈ inverse beat rate), and/or a shift-tolerant
  (±1–2 frame) target so near-misses are not fully penalized.
- **Cost to faithfulness:** changes the observation likelihood from the exact per-frame Bernoulli.

## 4. (Deploy) amortized free-run is weak → explicit filtering (M, largest change)
- **Symptom:** even with the above, free-running the prior is a weak way to infer beats from
  audio at test time.
- **Change:** replace amortized free-run with explicit Bayesian filtering (particle filter / SMC)
  over the *same* generative latent geometry at deploy time.
- **Cost to faithfulness:** this is no longer a VAE deploy path; it is the classic DBN inference.
  Largest departure — list as a separate track, not a "fix" to the faithful model.

---

## Suggested reading order for the panel
1 (collapse) and 3 (decoder) are the cheapest and most defensible; 2 (tempo scale) is needed for
the geometric read-out to mean anything; 4 is a different paradigm (keep separate). Each can be
A/B'd against the faithful baseline in the notebook's own eval, one knob at a time.
