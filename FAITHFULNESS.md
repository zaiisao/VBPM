# Faithfulness audit: is CHART still a bar-pointer VAE?

**Question (the bar we hold ourselves to).** CHART is meant to be a *variational autoencoder* whose **prior is the madmom bar-pointer DBN** and whose **posterior is refined by beat-annotated training data**. If the model that actually scores well has quietly stopped using that generative latent, then it is no longer the VAE we set out to build — it is a discriminative audio→beat predictor with a decorative latent attached.

This document records where the implementation stands against that bar, grounded in the code as of branch `fix/audio-driven-prior`.

## Verdict

The VAE machinery is **fully intact and faithful**. But a single constructor switch — `decoder_use_h_prior` — decides whether the model *uses* that machinery or *bypasses* it:

- `decoder_use_h_prior=False` (latent-only, run **ou2_init**): a genuine bar-pointer VAE. Reconstruction must flow through the latent; the posterior really refines the bar-pointer prior from beat annotations; the latent does real work. **Held-out phase-wrap F ≈ 0.39.**
- `decoder_use_h_prior=True` (run **ou6_hprior**, the strong model): the decoder reads the audio features (`h_prior`) directly, so reconstruction does **not** have to route through the latent. The latent goes decorative (held-out phase-wrap F = 0.137) even though free bits keep the KL number up. **Held-out decoder F = 0.933 — but this is "WaveBeat features → MLP," not the generative model.**

So we did **not** forego the VAE in the code; both modes exist and the faithful one runs today. Which one we *report* is a scientific-framing decision.

## What is still faithful (none of the dynamics work regressed it)

All references are to [`models/svt_core.py`](models/svt_core.py).

- **Bar-pointer prior** — the generative prior is still the DBN recursion: phase `μ = wrap(φ_{t-1} + φ̇_{t-1})`, tempo log-random-walk, meter transition gated by the predicted-mean bar-boundary indicator (`forward`, ~L601-636).
- **Posterior refined by annotations** — `q(z | b, h)` ingests beat/downbeat-distance features built from the targets (`encode_posterior`, ~L386-400; consumed ~L519-524). This is literally "posterior refined with beat-annotated data."
- **The latent distributions you asked to keep are all present and reparameterized** — von Mises (Best–Fisher) phase, Log-Normal log-space tempo, Gumbel-Softmax meter (`forward` init ~L542-549, transition ~L590-599).
- **ELBO** — `BCE + β·(KL_phase + KL_tempo + KL_meter)` with closed-form KLs (`models/loss.py`).
- **Algorithm 1 rollout** — prior at `t` conditions on the *sampled* `ẑ_{t-1}`, not teacher-forced GT.

Two clarifications on the recent dynamics work, since they looked like departures but are not:
- **von Mises is not abandoned.** The deterministic `phase_mu` chain in `sample_from_prior` is an *inference-time read-out only* (a noise-free mean trajectory for the phase-wrap evaluation). Training still samples and reparameterizes von Mises.
- **The OU mean-reversion and bounded log-tempo are regularizers on the prior's tempo random-walk**, grounded in the bar-pointer DBN lineage. They keep within-bar rubato possible while making free-running divergence low-probability. Still inside the framework.

## Where it bends away from a *pure* madmom prior

1. **The decoder shortcut (the consequential one).** `decoder_use_h_prior=True` concatenates `h_prior` (audio) with the latent before the emission MLP (`_decode`, ~L446-451). When the decoder can see audio, the information path becomes audio→decoder and the latent is vestigial — a *functionally posterior-collapsed* VAE. This is exactly what Gate 4 measured on ou6: F=0.933 with phase-wrap F=0.137, best at low-β/early-epoch where latents are loosest. The faithful switch is `False` (ou2).

2. **The prior is audio-conditioned (the smaller one).** The prior means carry learned correction heads `g_ψ(h_t)` read off `h_prior` (`prior_mean_corrections`, ~L406-421). This makes the prior a *conditional/amortized* prior `p(z|h)` rather than a fixed madmom `p(z)`. Legitimate for a VAE, and we deliberately shrank the correction reach to a nudge (`phase_corr_scale=0.1`) so audio *corrects* rather than *replaces* the dynamics — but it is a disclosed departure from a clean fixed prior.

## Why the WaveBeat baseline lands exactly on this question

ou6's decoder rides the frozen WaveBeat 2-channel activations, so its 0.933 is essentially "WaveBeat features → small MLP." A from-scratch WaveBeat baseline (same 4 datasets, same seed-42 splits, scored on the same ballroom val songs Gate 4 uses) measures precisely that contribution:

- If plain WaveBeat→peak-pick already reaches ~0.90 on ballroom val, then ou6-as-a-VAE adds ~nothing — the bar-pointer generative model is not what is scoring.
- The number that represents *the VAE's own dynamics* is ou2's **≈0.39**.

## Recommendation for reporting

If the contribution is "a bar-pointer VAE":
- Report **ou2 (latent-only)** as THE model — it is the faithful one.
- Use **ou6 / h_prior** only as an ablation / upper bound, explicitly flagged as "decoder may read audio" (not the generative model doing the work).
- Use the **WaveBeat baseline** to contextualize ou6.

## Held-out numbers (ballroom val, deterministic-mean read-out)

| Run | decoder F | phase-wrap F | downbeat F | faithful VAE? |
|-----|-----------|--------------|------------|----------------|
| ou6_hprior (`decoder_use_h_prior=True`, ep004) | 0.933 | 0.137 | 0.831 | No — decoder bypasses latent |
| ou2_init (`decoder_use_h_prior=False`) | 0.000 | 0.388 | — | Yes — latent does the work |
| baseline-120 | 0.294 | — | — | n/a |
| tempo-oracle | 0.503 | — | — | n/a |
| WaveBeat (from-scratch) | _pending_ | n/a | _pending_ | n/a (discriminative) |
