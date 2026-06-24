# DEEP_RESEARCH_2: Remedies for Free-Run-Deployed Sequential/SSM VAEs

Scope: a bar-pointer Dynamical VAE (DVAE) for beat/downbeat tracking, deployed by
free-running the prior (open-loop, no posterior). Three measured failure modes:
(1) inference-side posterior collapse, (2) the train->free-run deployment gap,
(3) likelihood misspecification for sparse periodic events. Findings below merge
24 adversarially verified claims.

---

## Executive summary

The literature decomposes the project's problem cleanly. For failure mode (1),
ELBO is provably underdetermined along the rate-distortion curve (Alemi 2018), so
fixing collapse requires changing the *objective or variational family*, not just
the optimizer: delta-VAE's committed-rate constraint, InfoVAE's mutual-information
objective, and SA-VAE's semi-amortized refinement each guarantee or restore
informative rate, with SA-VAE the only one validated on sequence data and
explicitly distinguishing *useful* rate (saliency-verified) from merely nonzero
KL. For failure mode (2), the train->free-run gap is a named, structural
limitation of the one-step KL bound (Hafner 2019); the strongest direct remedies
are multi-step latent-space consistency losses (latent overshooting, TD-VAE) and
architectures that enforce dynamics in a dedicated latent space (DVBF, Kalman-VAE)
or condition the prior on temporal context (VRNN) -- all of which make the prior's
own rollout the training target rather than only teacher-forced reconstruction.
For failure mode (3), the surveyed corpus confirms the diagnosis (powerful/iid
decoders enable collapse) but did NOT surface verified point-process or
shift-tolerant likelihood remedies -- this remains the least-covered axis.
The DVAE survey (Girin 2021) is the unifying reference and explicitly names the
"teacher forcing vs generation mode" dichotomy that is exactly the project's gap.

---

## Findings

### Finding A -- ELBO is underdetermined; collapse lives in the objective, not the optimizer (failure mode 1)
HIGH confidence. The standard ELBO does not distinguish points along the
rate-distortion diagonal -- many qualitatively different models share identical
ELBO, so optimizing ELBO alone cannot pin down whether z is informative
(Alemi 2018, "Fixing a Broken ELBO"). With a powerful stochastic decoder (RNN,
PixelCNN) a VAE can ignore z and still attain high marginal likelihood, and
beta>1 drives rate to ~0 (empirically R=0.0004 at beta=1.10). InfoVAE
independently proves that improving the ELBO can *provably degrade* inference
quality -- the failure is in the training criterion itself. Implication for the
bar-pointer model: raising KL/free-bits cannot manufacture useful rate, and the
collapse must be attacked by changing the objective/variational family.
Merged claims: [2],[3],[4],[5]. Sources: arxiv 1711.00464; arxiv 1706.02262.

### Finding B -- delta-VAE: committed minimum rate via constrained variational family (failure mode 1)
HIGH confidence. delta-VAE prevents collapse by constraining the posterior family
to keep a minimum (delta) divergence to the prior, guaranteeing a committed
minimum information rate WITHOUT weakening the decoder or modifying the ELBO. For
sequential latent models the structured (AR(1)-style) prior resembles slow feature
analysis, biasing latents toward slowly-varying-in-time representations -- directly
apt for a temporal bar-pointer latent. Caveat: it guarantees nonzero rate, not
provably *data-useful* rate. Merged claims: [0],[1]. Source: arxiv 1901.03416.

### Finding C -- InfoVAE / MMD-VAE: mutual-information objective for informative (not merely nonzero) rate (failure mode 1)
HIGH confidence. InfoVAE is a new class of objectives that maximizes
code-input mutual information, keeping latents informative regardless of decoder
flexibility and improving variational-posterior quality. Caveat: demonstrated on
iid image data, not sequential/free-run regimes; sequential-VAE literature notes
collapse can persist there, and the project's collapse is encoder/KL-gradient
driven rather than purely decoder-flexibility driven, so relevance is partial.
Merged claims: [6],[7]. Source: arxiv 1706.02262.

### Finding D -- SA-VAE: semi-amortized refinement closes the inference gap with verified useful rate (failure mode 1)
HIGH confidence -- strongest sequence-validated remedy for mode (1). SA-VAE uses
the amortized network only to INITIALIZE variational parameters, then runs
differentiable SVI steps to refine them, training end-to-end by differentiating
through SVI -- directly targeting the inference-side amortization gap that the
project measured. On Yahoo text a plain LSTM-VAE collapses (KL~0.01, PPL 62.5)
while SA-VAE (K=20) reaches KL=7.19, PPL 60.4, beating the LSTM-LM baseline (61.6).
Critically, the paper explicitly warns high KL alone does not prove the latent is
used (could be bad optimization) and VERIFIES via saliency that SA-VAE's rate is
meaningfully used -- the exact "useful rate vs nonzero rate" distinction the
question asks for. Caveat: validated teacher-forced (reconstruction PPL), not
free-run deployment. Merged claims: [8],[9],[10]. Source: arxiv 1802.02550.

### Finding E -- The train->free-run gap is a named structural limitation of the one-step KL bound (failure mode 2)
HIGH confidence. The standard variational objective trains the latent
transition/prior only via one-step KL regularizers: the gradient never traverses
a chain of multiple transitions, so the prior is never trained over multi-step
rollouts -- exactly the project's measured train->free-run gap (Hafner 2019,
PlaNet). The DVAE survey names the same phenomenon the "teacher forcing against
generation mode" dichotomy (train with ground-truth fed back vs deploy with own
outputs fed back) and notes it is poorly discussed in the literature. DVAEs do
deploy generatively by sampling the prior open-loop, confirming the project's
deployment is a recognized mode of the model class. Merged claims: [12],[17],[18].
Sources: PMLR v97 hafner19a; arxiv 2008.12595.

### Finding F -- Multi-step latent-space consistency losses make the prior rollout itself a good generator (failure mode 2)
HIGH confidence -- most directly composable remedy for mode (2). Latent
overshooting is an auxiliary loss computed purely in latent space (KL between
multi-step priors and corresponding one-step posteriors over all distances 1..D);
it improves long-horizon prior rollouts WITHOUT decoding extra observations and is
"compatible with any latent sequence model" -- i.e., it can be added to the
bar-pointer DVAE directly. TD-VAE complements this: trained on pairs of temporally
separated points via a TD-learning analogue, its prior "can be rolled out directly
without single-step transitions," enabling jumpy multi-step generation and
learning without BPTT through the whole interval. These are the closest match to
"train-as-you-deploy / multi-step rollout consistency" for DVAEs.
Merged claims: [13],[14],[15]. Sources: PMLR v97 hafner19a; arxiv 1806.03107.

### Finding G -- Enforce dynamics in a dedicated latent space: DVBF and Kalman-VAE (failure modes 1 and 2)
HIGH confidence. DVBF enables backpropagation through the transitions, which
"enforces state space assumptions and significantly improves information content
of the latent embedding" (mode 1: useful rate) and "enables realistic long-term
prediction" via free-running the learned dynamics (mode 2). It forces the latent
space to conform to the transitions rather than letting the recognition net
shortcut them. Kalman-VAE factorizes into a VAE pseudo-observation a_t plus a
SEPARATE linear-Gaussian SSM over z_t carrying dynamics; because the dynamics are
linear-Gaussian, the z-posterior is computed EXACTLY by Kalman filter/smoother
rather than amortized -- so it cannot be dragged to the prior by KL gradients,
bearing on mode (1). Caveats: DVBF/KVAE validated on low-dim physics
(pendulum/bouncing balls); KVAE requires the dynamics latent be linear-Gaussian
(the bar-pointer's von Mises/Log-Normal latents are not), and KVAE's open-loop
*superiority* claim was REFUTED in verification (split 1-2). Merged claims:
[19],[20],[21],[22]. Sources: arxiv 1605.06432; arxiv 1710.05741.

### Finding H -- Context-conditioned priors: VRNN (failure mode 2)
HIGH confidence. VRNN uses a time-varying prior p(z_t | h_{t-1}) conditioned on
the previous RNN hidden state rather than a fixed N(0,I), making the free-running
prior itself temporally context-dependent -- a minimal, composable change that
gives the open-loop rollout memory. Claim [23]. Source: arxiv 1506.02216.

### Finding I -- The bar-pointer DVAE is a well-defined, surveyed model class (framing)
HIGH confidence. The DVAE survey (Girin 2021, Foundations and Trends in ML) is the
unifying taxonomy: DVAEs model temporal dependencies in both latent and observed
sequences via RNNs/state-space models, encompassing VRNN, DKF, STORN, SRNN, KVAE,
etc. The bar-pointer model satisfies the general DVAE definition (caveat: it is not
literally one of the seven surveyed instances). This survey is the right reference
hub for composing the above remedies. Merged claims: [11],[16]. Source: arxiv 2008.12595.

---

## Composition guidance

- Modes (1) and (2) are separable and the remedies stack: a mode-(1) inference fix
  (SA-VAE refinement, or delta-VAE committed rate) can be combined with a mode-(2)
  rollout fix (latent overshooting, context-conditioned VRNN prior).
- Most directly droppable into the existing bar-pointer DVAE without rearchitecting:
  latent overshooting (Finding F, "compatible with any latent sequence model") and
  a VRNN-style context-conditioned prior (Finding H).
- delta-VAE's slow-feature/AR-prior structure is the most natural inference-side fix
  given a temporal latent, and unlike beta/free-bits it does not push q away from p.
- SA-VAE gives the only verified useful-vs-nonzero-rate evidence but is heavier
  (differentiate through SVI) and unvalidated in free-run.
