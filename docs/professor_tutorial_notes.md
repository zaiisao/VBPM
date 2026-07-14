# Professor's tutorial (VAEBPM_fin.pdf, 2026-07-10) — NORTH STAR notes

User directive 2026-07-12: keep saved, refer often. Full PDF held by user; these are working notes.

## Structure: θ emission, φ encoder, ψ conditional prior (notation disambiguates Sohn's compressed θ).

## THE DESIGN FORK the tutorial contains (both variants, different sections):
- **§7-8 variant ("encoder-only")**: encoder q(z|x) (NO b), fixed physical prior p(z) (no ψ);
  deployment = encoder + deterministic rule g (phase wraps): z_hat = mu_phi(x), b_hat = g(z_hat).
  "No training-inference gap" BY CONSTRUCTION (trained pipeline == deployed pipeline).
  == the user's intuitive pipeline. Our empirical analog: recognition read-out with silent b /
  GSNN-style paths (~0.25-0.35 F). Gap-free != accurate: the inversion difficulty moves INTO the
  x-only encoder (multimodal posterior, per-frame marginals).
- **§9/§12 variant (Sohn-standard)**: encoder q(z|x,b), learned conditional prior p_psi(z|x), three
  parameter sets; Misconception 6: the training-inference gap IS present. == what we built (matches
  ELBO_for_DBN spec).

## On the deployment gap (§6.8.6-6.8.7) — the tutorial's own structural argument:
x-only inference at best matches the AGGREGATED posterior = mixture over y of per-instance
posteriors, "definitionally broader"; gap is STRUCTURAL, unfixable by fitting psi. (Our
mode-averaging/gauge-multimodality story is this, with numbers: 0.25 amortized vs 0.92 decoded.)

## Remedies named + our measurements:
- Two-step generalized EB (§6.8.5, alternating psi-fit vs theta,phi-fit; psi -> aggregated
  posterior at convergence): implemented (EB arms) -> score-neutral, prior read-out unimproved.
- Sohn hybrid (§6.8.8, GSNN term; trains decoder on prior samples): implemented (hybrid_alpha)
  -> marginal. Tutorial itself: EB "does not address the gap", hybrid "reduces" it.
- §6.8.11 also offers physical-prior anchoring L_reg-EB (lambda_prior KL(p_psi || p_physical)) —
  UNTESTED arm; relates to our fixed_prior_scales (L4).

## Inference stance:
§8.1.6/§10: classical DBN = exact Viterbi on hand-set model; ours = amortized on learned model —
presented as framework difference, accuracy cost NOT adjudicated. Emission at inference:
"training scaffolding; can be discarded or retained as alternative read-out" (§8.1.5, 8.3 lists
alternatives A-D incl. emission-based ones). Grid-Viterbi decode of the LEARNED model = classical
inference x learned model: compatible completion, not contradiction (variational TRAINING is the
thesis; deployment slot open).

## Misc: §9.9 vM phase extension (we use WC — measured heavy-tails deviation); §9.10 joint
beat+downbeat == ours; §9.12 semi-supervised == our meter-only/M2 pathway; §9.13 domain adaptation
ideas (test-time adaptation) relevant to evidence-head transfer findings.
