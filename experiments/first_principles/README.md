# First-principles experiments (single-file, synthetic, vendor-verified)

Each script is self-contained (fixed seeds, NO project code, NO cached data, NO checkpoints),
demonstrates ONE claim from the VBPM campaign, runs in minutes, and prints a self-explanatory
verdict. **Every mathematical primitive comes from an established library**; the one component
that must stay hand-written (the differentiable forward recursion in exp3, which needs gradients)
is certified against the library implementation in exp2.

| script | claim | vendor components | key output |
|---|---|---|---|
| exp1_multimodal_posterior.py | gauge posterior is M-modal; any unimodal q is confidently-wrong (keeps 1/M mass) or uninformative (kappa->0) | scipy.stats.vonmises, scipy.special.rel_entr, scipy.signal.find_peaks | both optimal unimodal fits and their failure modes |
| exp2_discretization_vs_elbo.py | discretized model IS a GaussianHMM: exact log p(x) via hmmlearn CONVERGES under refinement; best-possible unimodal ELBO (8 restarts, unbounded kappa, zero amortization gap) still ~2.2 nats/frame below | **hmmlearn.GaussianHMM.score** (every exact value), torch.distributions.VonMises | exact -15.09 (converged by 360 bins) vs best ELBO -79.9; gap 64.9 nats/30 frames = pure approximation gap |
| exp3_elbo_collapse_vs_exact.py | headline: same model + data + decoder; ELBO training corrupts dynamics (omega 0.15->0.013, deploy F=0.103), exact forward training recovers them (omega 0.1501, F=0.841) | torch.distributions.Normal.rsample + kl_divergence (arm A), **librosa.sequence.viterbi** (deployment decode) | deploy F 0.103 vs 0.841 |

Cross-certification: exp3's torch forward recursion is the same recursion pattern whose values
match hmmlearn.score() digit-for-digit in exp2's table.

Correspondence to the real system (docs/overnight_packet_2026-07-12.md): 327-song val, wave-2
corpus, ELBO-trained deploy 0.398 vs exact-forward-trained 0.844.

Known toy simplifications (documented in-file): exp3's tempo is a global learned scalar (aliased
likelihood -> needs in-basin init; the real model searches tempo as part of the latent state);
exp2/3 ELBO arms are given every advantage (multi-restart, unbounded concentration, no
amortization) so the reported gaps are LOWER bounds on the variational penalty.

Run:  python exp{1,2,3}_*.py   (env: chart; needs scipy, torch, librosa, hmmlearn)
