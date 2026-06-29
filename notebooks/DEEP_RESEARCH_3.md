# Deep Research 3 — Differentiable DBN, learned tempo, and the SOTA pipeline

Goal: find the literature framing that reconciles our two empirical facts —
(a) gradient-learning a continuous tempo latent is walled; (b) the differentiable DBN /
per-frame phase correction reaches ~0.96 — and decide whether our negative result is
expected/publishable.

## Q1 — Is there a differentiable / end-to-end DBN that BEATS frozen-activation + post-hoc DBN?

**No published beat tracker does this and wins.** The field's structure is invariant: a
trained neural net emits a per-frame beat/downbeat **activation**, then a **separate,
non-trained** probabilistic model (DBN-as-HMM → Viterbi, or particle filter) infers
period+phase. Böck/madmom, TCN+DBN, Beat-Transformer, BeatNet all share this split.

- Generic differentiable dynamic programming exists (differentiable Viterbi / forward-backward,
  Mensch & Blondel 2018) but is **not** the winning beat-tracking recipe — nobody has shown
  end-to-end-trained structured inference beating the frozen-activation+post-hoc-DBN pipeline.
- So our "pure geometric latent read-out underperforms a per-frame-correcting DBN" is the
  **expected** outcome, consistent with the entire field. Publishable as confirmation, not surprise.

## Q2 — Does modern SOTA still use the DBN at all?

**The frontier moves AWAY from the DBN.** Beat This! (2024), current SOTA and most general,
**drops DBN post-processing entirely**: a strong transformer frontend + shift-tolerant BCE
training + trivial peak-picking. It explicitly handles tempo/meter changes *because* it has no
DBN tempo-continuity prior to fight. This directly corroborates our joint-eval verdict: the
**strong frontend is what carries the performance**; a structured generative latent does not
earn its complexity against it. BeatNet keeps a filter (particle/HMM) for the *online* setting.

## Q3 — Is tempo ever gradient-learned as a free continuous latent?

**No.** Tempo is always either (i) **computed** by signal processing — autocorrelation /
resonating comb-filter banks / tempogram (Böck et al. RNN+comb-filter; Foroughmand & Peeters
"Deep Rhythm"; tempogram+Kalman), or (ii) framed as **classification** over tempo bins. There
is no precedent for a free continuous tempo latent learned purely by recon gradient — exactly
the route we measured as walled (every variant floored/constant, failed the leak test). The
literature's answer to "compute the tempo from what?" is: from onset periodicity (comb/autocorr)
or as a classifier — never as an unsupervised generative latent.

## Decision implication

The three questions converge on one verdict, and it matches our experiments:

1. **Keep the contribution honest.** A differentiable end-to-end DBN is *novel framing* but the
   literature gives no evidence it beats the standard split — and our 0.96 geometric read-out is
   the DBN's per-frame correction doing the work, not a learned continuous latent.
2. **The publishable story** is the diagnosis itself: a faithful bar-pointer DVAE deployed by
   free-running its prior is structurally capped (~0.40 amortized / ~0.82 perfect-constant-tempo
   integration / 0.96 with per-frame DBN correction), tempo cannot be gradient-learned as a free
   latent, and the field independently confirms both (SOTA drops the DBN for a strong frontend;
   tempo is always computed/classified, never free-learned).
3. **If we keep the VAE**: the only faithful path that escaped the wall is the exact-filter route
   (Kalman-VAE filtering / SMC), which uses the audio every frame — i.e. it *is* the per-frame
   correction, reframed inside the generative model. That is the honest reconciliation.

Sources: dida beat-tracking overview; Beat This! (2024); Beat Transformer (arXiv 2209.07140);
WaveBeat (arXiv 2110.01436); BEAST (arXiv 2312.17156); Böck et al. RNN+comb-filter tempo;
Foroughmand & Peeters Deep Rhythm; "AI and Tempo Estimation: A Review" (arXiv 2401.00209);
Mensch & Blondel 2018 differentiable DP.
