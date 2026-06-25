# Deep dive: where *exactly* the faithful bar-pointer VAE breaks down, and why every fix missed

End-to-end analysis of the free-run deployment failure, with a per-experiment retrospective. All
numbers are beat-F1 @ ±70 ms (mir_eval); fr_lat = free-run latent read-out = the deployment metric.

## The precise breakdown (localized to one thing)

Pipeline, stage by stage, with the measured status of each:

| stage | works? | evidence |
|---|---|---|
| data / targets / fps | ✅ sound | earlier data-bug fixed; fps 86.13 verified |
| geometric read-out + mir_eval | ✅ sound | ideal latents → 0.97 (`ideal_readout.py`) |
| decoder / likelihood | ✅ (after widen) | tf_post_dec 0.00→0.99 with widened target |
| posterior inference (teacher-forced) | ✅ excellent | tf_post_lat 0.65, **posterior tempo corr 0.93** |
| **prior: tempo estimate from audio** | ❌ **BROKEN** | **init-tempo error 852%, 0% in right octave** |
| prior: phase dynamics | ✅ fine given tempo | one-step tf_prior_lat 0.48–0.63 |
| deployment (free-run) | ❌ 0.36 | uses the broken prior tempo |

**THE BREAKDOWN IS A SINGLE THING: the prior cannot estimate tempo (global beat rate) from audio alone.**

The decisive decomposition (`tempo_decomp.py`, healthy oulong checkpoint):
- model's `prior_init` tempo is **852% off on average, NEVER within 0.5–2× of GT (0% octave-correct)** — it
  isn't a tempo estimate, it's a near-constant garbage value.
- free-run with the model's own tempo frozen = stochastic free-run = **0.356**; with GT tempo frozen =
  **0.510**. So supplying the right *constant* tempo alone buys +0.15. Drift is secondary (oracle injection
  earlier: periodic phase+tempo resync → 0.67–0.83, so phase drift costs the rest *once tempo is right*).

**The red herring that misled the whole investigation:** the *posterior* tempo is accurate (corr 0.93), so
it looked like "the model can do tempo, it just doesn't deploy it." False. The posterior reads the
**ground-truth beats** (`encode_posterior(h, b)`) — it gets tempo *for free off the beat spacing*, it does
NOT extract it from audio. Deployment has no beats and uses the **prior**, whose audio-only tempo estimate
is garbage. So tempo competence was always an illusion of teacher-forcing.

**Why the prior can't estimate tempo — two concrete architectural causes:**
1. `prior_init_head(pc.mean(1))` — the initial state is read from the **mean-pooled** prior context.
   Mean-pooling destroys the temporal/periodicity structure that tempo estimation requires → the initial
   tempo is structurally un-estimatable (hence ~constant garbage, 0% octave-correct).
2. The per-frame prior tempo **mean is the audio-blind random walk** (`μ_τ = log φ̇_{t-1}`) → it never
   corrects the garbage init from the audio thereafter.

So at deployment: garbage initial tempo + no audio correction = a metronome at the wrong rate, catching
~1/3 of beats by periodicity coincidence (the 0.33–0.40 "floor" was always this).

## Per-experiment retrospective — why each fix did NOT solve deployment

Every experiment falls into one of three buckets, none of which touch "prior estimates tempo from audio":

| experiment | what it actually changed | why it didn't fix deployment |
|---|---|---|
| **widen target** | decoder recon (likelihood) | fixes the decoder; the prior's tempo estimate is untouched |
| **β<1 / free-bits / KL-anneal** | inference-side rate balance | helps the *posterior*; deployment uses the prior |
| **He-2019 aggressive encoder** | trains the *encoder* harder | encoder = posterior = discarded at deploy |
| **latent_only** | forces decoder to use z (teacher-forced) | raises tf_post_lat; prior tempo still garbage |
| **overshoot (PlaNet)** | multi-step prior KL on σ/κ only | phase/tempo *means* are parameter-free → can't move the tempo estimate; collapses |
| **OU tempo prior** | bounds tempo *variance* | the killer is the wrong *mean*, not variance; reverts to a constant, not the song's tempo |
| **survival / renewal likelihood** | makes *posterior* tempo accurate (corr 0→0.5) | posterior accuracy is from beats; deployment uses the prior (audio-only) |
| **Tier B (audio→prior tempo mean)** | the RIGHT target (prior audio→tempo) | but reads per-frame `pc` (no periodicity) → estimate weak; anchor-KL corrupts posterior; ~0.40 |
| **distillation (g_τ ← posterior tempo)** | distill posterior tempo into prior head | target (posterior tempo) is a beat-readout; `g_τ` sees only audio → can't reach it; ~0.40 |
| **free-run reconstruction** | train prior rollout to match beats | deep stochastic-rollout graph collapses; prior_init (mean-pooled) can't carry per-song tempo |
| **scheduled sampling** | expose rollout drift | drift is secondary; collapses latent_only |
| **h-dropout (word-dropout)** | force latent use in decoder | posterior-side; doesn't touch prior tempo |
| **FIVO (filtering ELBO)** | better *training-time* inference | deployment still free-runs from the garbage prior_init; no observation to filter at deploy |
| **particle-filter deploy (non-VAE)** | DOES filter audio at deploy | crude PF (0.35) < raw peak-pick (0.64); and not a VAE |

**Unifying conclusion:** ~all experiments either (a) improved the **posterior / teacher-forced** side
(which deployment discards), (b) attacked **drift/variance** (secondary), or (c) aimed at the prior's
tempo but **could not solve the underlying problem — extracting tempo (long-range periodicity) from audio
alone.** Not one of them made the prior estimate tempo from audio. That is the single unsolved thing, and
it is exactly the one thing deployment needs.

## What this implies
- The faithful VAE's deployment cap (~0.4) is, precisely, **the prior's inability to estimate tempo from
  audio** — compounded by an architecture that mean-pools away periodicity at init and never corrects.
- The posterior "tempo works" result is not transferable: it's beat-readout, not audio-extraction.
- The fix has to be **tempo estimation from audio** — which needs (i) a periodicity-aware encoder (long
  receptive field / dilated attention / autocorrelation), not mean-pooling, and (ii) that estimate placed
  where deployment can use it. This is the capacity problem, not an objective/inference trick — which is
  why no loss or inference change moved it.
