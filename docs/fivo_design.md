# FIVO for VBPM — design doc (2026-07-12)

**Status: PROPOSAL. Not implemented. Decision pending user sign-off (changes the training objective).**

## Why (the cause, not a symptom)
Every result this week converges on one diagnosis: the ELBO pays the model in reconstruction nats,
and the collapsed model reconstructs *fine* (recon 188 vs healthy 183; phase worth only ~9 nats to
the ELBO, ~0 when collapsed). Collapse is nearly free by the objective's own accounting and only
catastrophic for the particle filter — which the objective knows nothing about. So:

- free-bits (two-way or not) = holds a KL number up, not the latent's usefulness — symptom.
- prior-side remedies (EB, hybrid, anchor, anneal) = all null/negative; they can't reach a
  reconstruction-side disease. (Professor-remedy scorecard: 0/4.)
- early-stop = dodges collapse, can't train longer; v3-full 0.810 < v3-early 0.829 proves longer
  training under this objective HURTS deployment.
- x-only (§7) = removes the amortization gap but yields latent-unused (recog 0.000): the gap was
  not the whole disease.

**The root cause is objective misalignment: training optimizes reconstruction; deployment runs a
particle filter. FIVO makes the training objective BE the filter's own likelihood bound.**

## What FIVO is (Maddison et al. 2017, "Filtering Variational Objectives")
Instead of the per-frame ELBO, maximize log of the particle filter's marginal-likelihood estimate:
  L_FIVO = E[ log ( (1/N) sum_i w_t^i ) ]  summed over t,
where w_t^i are the SAME importance weights our bootstrap filter already computes
(model/particle_filter.py: emission_log_likelihood scored against observations). Key properties:
- It is a valid lower bound on log p(observations) that IMPROVES with N (tighter than ELBO).
- Its gradient trains the proposal + model to make the FILTER's estimate good — i.e. it optimizes
  exactly the deployment computation. No train/deploy gap by construction (the RIGHT kind, unlike
  §7 which removed the gap by removing the useful signal).
- Resampling is the mechanism that kills collapse: a particle set that ignores phase gets a diffuse
  weight distribution -> low marginal-likelihood estimate -> gradient pressure to USE phase.

## How it maps onto our code (reuse, minimal new surface)
Our filter already computes everything FIVO needs; we currently only use it at deployment.
1. `model/particle_filter.py`: add a `differentiable=True` path that
   - keeps `log_weights` in the graph (today it's under @torch.no_grad),
   - accumulates `logsumexp(log_weights) - log(N)` per resample segment into `L_FIVO`,
   - uses the reparameterized proposal samples (wrapped-Cauchy/Laplace rsample -- we already have
     implicit-reparam gradients for these) so the estimator is pathwise-differentiable,
   - detaches ancestor indices at resampling (standard FIVO; the biased-but-workable gradient. If
     variance is bad, add the DReG / stop-gradient-on-weights correction later).
2. `train.py`: behind `objective.fivo_weight > 0`, run the differentiable filter on the SAME batch
   (observations = frontend act2 or own head, available at train time -- no ground truth) and add
   `- fivo_weight * L_FIVO.mean()` to the loss. Start as an AUXILIARY term alongside the ELBO
   (curriculum), then anneal ELBO down so FIVO dominates -- avoids the cold-start where an untrained
   proposal gives a hopeless filter.
3. `config.py ObjectiveConfig`: `fivo_num_particles` (start 16-32 for train speed), `fivo_weight`,
   `fivo_elbo_anneal_steps`.

## Cost / risk
- Compute: N particles through the model per frame per step. N=16-32 in training (vs 800 deploy) is
  the standard compromise; ~Nx the rollout cost. Our model is tiny (759k params) and GPU is
  underused, so wall-clock is affordable.
- Gradient variance: the known FIVO wart (resampling). Mitigations, in order: (a) start as aux term,
  (b) biased detached-ancestor gradient first, (c) DReG estimator if needed.
- Honesty: FIVO is a departure from BOTH the spec (Alg 1 = SGVB) and the tutorial (amortized-only
  §8). It is NOT tutorial-faithful. It IS the deployment-faithful objective. Frame it as such.

## Decision needed
- GO: implement the differentiable filter path + aux-term training; first smoke on MERT (early800
  corpus) since that lane is MIREX-clean and the collapse there is well-characterized.
- Expected read if it works: v3-full (2000-step) deployment >= v3-early (0.829) instead of < it --
  i.e. longer training stops hurting. That single inequality flipping is the cleanest success test.
