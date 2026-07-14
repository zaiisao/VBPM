# VBPM — project overview for publication planning

*A self-contained briefing (2026-07-11). Written so a reader with ML/MIR background but no access to
this repository can understand what the project is, where it honestly stands, and what the open
publication decisions are. Numbers and caveats are stated as the internal evidence currently supports
them — including the retractions.*

---

## 0. TL;DR

**VBPM (Variational Bar Pointer Model)** is a fully-differentiable, continuous-state neural
realization of the classic "bar-pointer" dynamic Bayesian network (DBN) used for beat/downbeat
tracking (the model behind madmom). It is a **conditional Deep Markov Model** — a structured
sequential VAE — with three interpretable per-frame latents (meter, bar phase, tempo), trained by
ELBO on top of a frozen neural frontend, and deployed by a particle filter (the lineage's own
inference), not by its decoder.

**The strongest result is a mechanism finding, not a leaderboard number.** We diagnosed *why*
structured sequential VAEs of this kind silently fail when you stop teacher-forcing them and actually
run them at deployment — an "emission side-channel" through which the encoder smuggles the answer
past the generative structure — and we found an ELBO-exact modeling fix. That diagnosis is
score-independent and does not depend on any contaminated baseline.

**Honest status on the tracking numbers:** after a rigorous internal audit we **retracted** an
earlier "beats SOTA on SMC" headline (frontend train/test contamination) and confirmed that, in every
*trustworthy same-evidence* measurement, the structured post-processor **has not yet beaten
same-evidence peak-picking** on beat-F. On clean data (GTZAN) it is within ~0.02 of peak-picking and
learning contributes a large, real +0.25 over the untrained architecture. The genuinely open — and
publishable — scientific question is whether structure **helps on weak/imperfect evidence** (rubato,
soft onsets), where peak-picking is not obviously optimal.

---

## 1. Task and setting

**Task.** Beat tracking and downbeat tracking from music audio: emit beat times and downbeat (bar-start)
times; jointly this also implies tempo and meter (time signature / beats-per-bar). Standard evaluation
is mir_eval F-measure with a ±70 ms tolerance window.

**The modern pipeline split.** State-of-the-art beat trackers are a *neural frontend* (produces a
per-frame beat/downbeat activation) followed by *post-processing* that turns the activation into
discrete times. Historically the post-processor was a DBN (madmom: a hand-designed HMM over
tempo × bar-position × meter, decoded by Viterbi). The recent "Beat This!" system (Foscarin et al.,
ISMIR 2024) argued the DBN is unnecessary — a strong transformer frontend plus simple peak-picking is
enough on standard benchmarks. **VBPM sits in the post-processor slot**, but replaces the fixed DBN
with a *learned, differentiable generative state-space model*.

**Our frontend.** We freeze Beat This (`final0` checkpoint) and consume two things from it:
its 512-dim penultimate features (as conditioning `h`) and its beat/downbeat activation probabilities
(as the observation the particle filter scores). The frontend is modular (a MERT self-supervised
variant is scaffolded).

---

## 2. The model (VBPM)

### 2.1 Provenance
VBPM is the implementation of an internal spec, **"ELBO for DBN"** (Ahn, March 2026), which takes the
bar-pointer DBN (Whiteley, Cemgil & Godsill 2006; the model class behind Krebs/Böck/madmom) and,
instead of collapsing tempo/phase/meter into one composite HMM state for exact Viterbi decoding,
**keeps all three as separate latent variables with their own distributional forms and does
variational inference**. That is the structural novelty relative to the classical DBN line.

### 2.2 Latents (per frame `t`)
| latent | distribution | meaning |
|---|---|---|
| meter `m` | Categorical (K classes; class k ⇒ k+1 beats/bar) | time signature / beats per bar |
| bar phase `φ` | wrapped Cauchy on [0, 2π) | pointer position within the bar; a wrap = a downbeat |
| log-tempo `s` | Laplace | log angular advance of the pointer per frame |

The phase is **bar phase**: it sweeps once per bar and wraps at the downbeat. Beats are read off as the
`beats_per_bar`-th harmonic of φ; downbeats as φ's own wraps.

### 2.3 Generative dynamics (the bar-pointer transition)
- **Phase:** `φ_t ~ WC(φ_{t-1} + exp(s_{t-1}), c^p)` — mean is the deterministic pointer advance.
- **Tempo:** `s_t ~ Laplace(s_{t-1}, b^p)` — a heavy-tailed random walk.
- **Meter:** `m_t ~ Categorical(π^p)` with a full K×K transition matrix from a network conditioned on
  `(m_{t-1}, φ_t, φ_{t-1}, h)` — generalizes madmom's fixed switch probability.

Distribution-family choices (wrapped Cauchy for phase, Laplace for tempo) were **adopted for fidelity
to measured heavy-tailed increment/microtiming laws in real annotations, and are score-neutral** — von
Mises / Gaussian are kept only as light-tailed ablation arms. Both have closed-form KLs (the wrapped
Cauchy KL needs no Bessel functions and its sampler is a plain reparameterized inverse-CDF transform).

### 2.4 Emission (the crux — see §3)
The default emission is a **madmom-style parametric cosine bump**: `logit_beat = a + softplus(g)·exp(k(cos(bpb·φ) − 1))`
at the beats-per-bar harmonic, and an analogous bump for downbeats at the fundamental — **five scalar
parameters total**, with `bpb` coming from the *soft meter latent* (so meter is causally
consequential, never a hardcoded 4). MLP emissions are retained only as the diagnosis-ladder ablation
axis.

### 2.5 Training objective
Standard SGVB / negative ELBO: reconstruction (BCE of the emission against beat/downbeat targets) +
the three per-frame KLs (initial + transition). Additions, all of which are **likelihood terms on
observed quantities (emissions), so the objective stays a valid ELBO — nothing annealed, nothing with
detached gradients in the loss value, nothing "auxiliary" in the un-principled sense**:
- **Meter CE**: the song's annotated beats-per-bar is an observed variable with a categorical emission
  `p(M | m_t)` (semi-supervised, Kingma M2 style; frame-summed).
- **A tempo-grounding emission** (tempo-slope, or optionally a "sawtooth" phase target): supervises the
  tempo/phase latent with the annotation-derived ramp so tempo stays physical.
- **Prior-preserving free bits**: a KL floor (anti-collapse) plus a zero-value gradient channel that
  lets the prior networks keep learning toward the posterior below the floor (an ELBO-exact cousin of
  DreamerV2's KL balancing).

**Minimal faithful core (established by an ablation "minimality ladder"):** parametric emission +
meter CE + prior-preserving free bits + **one** tempo-grounding emission. Removing meter CE, free
bits, or *both* grounding terms breaks it; sawtooth, the g-prior, and (for the Bayesian read-out) the
tempo-slope are individually removable.

### 2.6 Deployment inference
At test time **we discard the decoder and infer the latent trajectory** by a bootstrap **particle
filter** on the learned model, scoring the frozen frontend's activations as evidence. Beats/downbeats
come from either the MAP particle's phase wraps or a **Bayesian ensemble read-out** (peak-pick the
particle-weighted wrap-activation); the Bayesian read-out is consistently ~+0.13 better. The filter
uses "DBN-like" settings we fixed on 2026-07-10: sharp proposals (near-deterministic phase advance,
near-constant per-particle tempo), stratified bar-gauge initialization (particles born at all bar
offsets, offsets taken from each particle's own meter latent), and downbeat-channel evidence
up-weighting. These are standard filtering practices and **change no part of the ELBO**.

---

## 3. The central scientific finding — the emission side-channel

This is the contribution that is robust to every measurement caveat below, because it is a mechanism,
diagnosed by controlled probes, not a benchmark ranking.

**The phenomenon.** For weeks, *every* deployment read-out was capped at F ≈ 0.35–0.40 and invariant
to everything — data scale (200→3300 songs), inference dials (proposal noise ×0.01–×50, particles
400→1600, evidence temperature), and objective knobs. Yet **teacher-forced reconstruction was good.**
Train-time metric excellent, deploy-time metric stuck: the signature of a train/deploy gap, not
ordinary posterior collapse.

**Root cause (three probes, one mechanism).** The factorized ELBO prices each latent's KL but does
**not assign semantics** — semantics live in the *structure of the generative functions*. The
transition respects that structure (`φ_t = φ_{t-1} + e^{s}`); a flexible MLP decoder on
`(φ, s, m)` respects nothing. So SGD routed *all* event-timing information through whichever latent was
cheapest to shape — the unbounded, KL-cheap **log-tempo** channel — and the prior tempo scale was
itself a learnable head that **inflated** (σ ≈ 0.57 nats/frame, a ±77%/frame tempo kick) precisely so
those posterior tempo wiggles cost ~zero KL. The encoder **Morse-coded the beat grid into physically
absurd tempo** (posterior log-tempo ≈ 6.1 ⇒ ~400 rad/frame), and the decoder became phase-blind.
At deployment the *prior* produces smooth physical tempo ⇒ no wiggles ⇒ the decoder emits its base
rate ⇒ capped. Three probes nail it:
1. **The likelihood ranks garbage above truth** — the ground-truth trajectory (oracle phase+tempo)
   scores ~330 nats *worse* than a deliberately wrong one (tempo ×1.25, phase +half-bar), on 8/8
   songs. No inference scheme can recover truth from such a likelihood — this alone explains the wall.
2. **Phase is nearly decorative** — a half-bar misphase costs only ~22 nats/1600 frames.
3. **Kill shot** — flatten the tempo channel of the teacher-forced posterior ⇒ P(downbeat) at
   downbeats collapses 0.976 → 0.001; flatten phase ⇒ stays 0.937. Events lived entirely in tempo.

**The fix — a modeling choice, ELBO stays exact.** Specify the emission to depend on pointer
**position** `(φ, m)` only, not tempo — which is *more* faithful to the bar-pointer lineage
(Whiteley/madmom: the observation depends on position; tempo only parameterizes the transition
kernel). The strongest form is the parametric cosine bump (§2.4): five scalars, **structurally
incapable of smuggling events through tempo or meter**. Companion (equally ELBO-exact): freeze the
*prior* transition scales at physical values so the KL actually prices side channels (posterior scales
stay learned). Together these close the channel.

**Why this retro-explains the whole history.** It explains the long-standing "latent unused"
observation, the historic tempo blow-ups (the unbounded random walk was the *symptom*; the decoder's
pull was the *cause*), why an earlier exact-inference Kalman-VAE variant escaped the wall (its
structured Gaussian emission offered no cheap side channel), and why a brute-force "synthesis" config
worked (it forced tempo to be physical, closing the channel by other means).

**Positioning.** This is a concrete, mechanism-level instance of the information-routing / "cheapest
channel wins" problem in VAEs (Chen et al., Variational Lossy Autoencoder; Alemi et al., Fixing a
Broken ELBO), specialized to *structured sequential* VAEs where the pathology is invisible under
teacher-forcing and only appears at free-running deployment. We are not aware of this specific
diagnosis-and-fix being made in the DVAE literature.

---

## 4. The deployment result, and the honest reckoning

### 4.1 The wall broke
With the fixed recipe (parametric emission + κ≈2 phase concentration + a tempo-grounding emission) and
the fixed particle filter, in-domain deployment F jumped from the multi-week 0.35–0.40 cap to
**Bayesian beat F ≈ 0.82–0.84 / downbeat ≈ 0.78–0.80**. A full re-audit of every historical checkpoint
under the fixed pipeline found the old and fixed metrics are **uncorrelated** (Spearman ρ = +0.12,
p = 0.6, n = 18): *weeks of prior F-based A/B decisions had been made on read-out noise.* Consequently
several older internal verdicts (a "~0.4 deployment cap," "the VAE doesn't earn its complexity") are
**invalidated as artifacts of the broken pipeline** — the mechanism findings are not.

### 4.2 Three caveats that must travel with every number
1. **Untrained-model control.** An *untrained* VBPM run through the same fixed filter already scores
   ~0.77/0.64 (in-domain / short-window SMC), because the Bayesian read-out leans on (a) the frozen
   frontend's evidence, (b) the fixed-filter machinery, and (c) the parametric emission's *initialized*
   bump shape. So **every headline must quote the untrained baseline.** Learning's contribution on the
   *Bayesian* read-out is small at short horizons but **large at full length** (+0.20 at 40 s) and large
   on the MAP read-out (0.35 → 0.68) — learning is real, but the architecture-plus-frontend floor is high.
2. **Frontend train/test contamination.** The frozen Beat This `final0` was trained on all datasets
   *except GTZAN* — including SMC and our in-domain val sets. An earlier "**SMC 0.700 zero-shot, +8.7
   over SOTA**" headline was therefore **RETRACTED**: it was "SMC-trained frontend + zero-shot inference
   layer," not comparable to fold-held-out published numbers. Fold-honest re-extraction (each song's
   activation from the checkpoint that held it out) gives filter **0.462** vs same-evidence peak-pick
   **0.599** vs fold-trained published SOTA 0.55–0.61. **GTZAN is the only currently clean cross-system
   evaluation** with this frontend.
3. **Never beaten same-evidence peak-picking (yet).** On clean GTZAN (993 songs, same activations):

   | method | beat F | downbeat F |
   |---|---|---|
   | peak-pick (same activations) | **0.893** | **0.774** |
   | VBPM trained filter (Bayes) | 0.868 | 0.754 |
   | VBPM trained filter (MAP) | 0.721 | — |
   | untrained-architecture control (Bayes) | 0.615 | 0.548 |

   The trained filter is **−0.025 / −0.020 under peak-picking** — the honest pattern is uniform: the
   structured layer has *not* beaten same-evidence peak-picking in any trustworthy measurement. But two
   real facts survive: **learning adds +0.253 / +0.206 over the untrained architecture** on clean data,
   and on *honest weak evidence* (fold-honest SMC) the gap to peak-pick is 5–7× larger and **negative**
   (structure's continuity assumptions *hurt* on weak evidence) — which is itself the interesting signal.

### 4.3 What survives regardless of the baseline war
- **Meter.** VBPM infers beats-per-bar as a trained, consequential latent; the best arm reaches meter
  accuracy 0.97 with genuine non-4/4 (3/4) predictions on real validation. **Peak-picking cannot do
  this at all.** Meter is a capability axis, not a beat-F axis.
- **A single joint generative state** over tempo/phase/meter/beats/downbeats (vs. separate heads).
- **The mechanism paper** (§3) — score-independent.
- **OOD / weak-evidence behavior** — to be measured cleanly (GTZAN clean; fold-honest caches pending).

---

## 5. The mission and the open question

**Standing mission (user, 2026-07-05):** reach **SMC beat F ≥ 0.7** from the VBPM line — SMC_MIREX is
the hard, expressive/weak-onset benchmark where published SOTA sits ~0.62–0.65, so ≥0.7 would be a
headline. This connects directly to the user's **own prior paper, "SMC Blind Spot"** (arXiv 2605.12287),
which showed DBN post-processors have tempo-coverage blind spots; VBPM is positioned as its
**constructive sequel** — a continuous learned bar-pointer filter with no tempo grid and no prior floor.
(Note the untrained-control result sharpens the sequel: the blind-spot *removal* is largely
**architectural**, provable without training.)

**The reframed research question after the reckoning:** *Can a structured, learned post-processor
**help on imperfect/weak evidence**, where same-evidence peak-picking is not optimal?* On strong clean
evidence peak-picking is a very high bar (0.89) and structure roughly ties it while adding meter; the
scientifically live regime is rubato / soft-onset / expressive audio. Known prior art the user has
already established: naïvely "fixing" Beat This's tempo augmentation did **not** help SMC (a
statistics-layer fix that doesn't touch the acoustics layer). Current levers under test: rubato-adaptive
proposal noise (empirical-Bayes arms), expressive training data (ASAP / MAESTRO performances), a
completed tempo-stretch augmentation pool, and evidence calibration.

---

## 6. Candidate publication framings (for discussion)

These are the options to weigh with the web LLM. They are not mutually exclusive (a mechanism paper now,
a systems paper later).

**A. The mechanism / methods paper (lowest-risk, strongest today).**
*"The emission side-channel: why structured sequential VAEs pass teacher-forcing and fail free-running,
and an ELBO-exact fix."* Contribution: the diagnosis (three probes), the general principle (factorized
KL prices magnitude, not semantics; the cheapest unconstrained channel absorbs the signal; invisible
under teacher-forcing), and the fix (constrain the emission's *arguments* to the structurally-meaningful
subset — equivalently, give the emission a fixed physical functional form). Beat tracking is the
illustrative testbed. Venue flavor: a DVAE / probabilistic-ML venue, or an MIR venue with the ML
framing foregrounded. **Pro:** does not depend on any contaminated baseline or on beating peak-picking.
**Con:** needs to be shown to generalize beyond this one model to be maximally compelling (a second toy
or a second modality would strengthen it).

**B. The system paper (higher-risk, needs a clean win).**
*"A differentiable bar-pointer VAE: joint tempo/meter/beat/downbeat tracking as one generative state."*
Contribution: the faithful continuous-state neural DBN, meter as a consequential latent (which
peak-picking can't provide), particle-filter deployment. **Pro:** the "meter + joint state + interpretable
latents" story is real and peak-picking has nothing to say about meter. **Con:** on beat-F it currently
*ties-to-slightly-loses* to same-evidence peak-picking on clean data; a system paper usually needs a
headline win on *some* clean axis. Candidate clean wins to chase: (i) downbeats or meter accuracy under
matched evidence, (ii) the weak-evidence regime (§5), (iii) a self-supervised-frontend (MERT) variant
where "the frontend does all the work" objection is removed because all beat supervision flows through
our ELBO.

**C. The blind-spot sequel (needs the clean OOD measurement).**
*"Removing DBN tempo blind spots architecturally"* — extends arXiv 2605.12287; the untrained-control
result is the hook. **Pro:** directly builds on the user's own paper; the architectural-not-learned
framing is clean. **Con:** the flagship OOD claim currently rests on a contaminated frontend; it needs
the fold-honest caches (re-extraction in progress) or a frontend that never saw SMC.

**Cross-cutting requirement for B and C:** the honest baseline table (untrained control + same-evidence
peak-pick + fold-honest evaluation) must be front-and-center. The internal audit has already done the
painful part of building it; a reviewer who would otherwise catch the contamination will instead see it
disclosed and controlled.

---

## 7. Related work (positioning anchors)

- **Bar-pointer / DBN lineage:** Whiteley, Cemgil & Godsill 2006; Krebs, Böck & Widmer 2013/2015;
  Böck et al. (madmom) 2016. *VBPM = a differentiable, continuous-state, variational realization of this.*
- **Frontends / "no-DBN" position we argue with:** Foscarin, Schlüter & Widmer (Beat This!) 2024;
  Beat Transformer 2022; BeatFM 2025 (same-activation A/B target; SMC 8-fold numbers). MERT 2023
  (self-supervised frontend arm).
- **DVAE / structured VAE foundations:** Krishnan, Shalit & Sontag (Deep Markov Model) 2017; Girin et al.
  (DVAE review) 2021; Fraccaro et al. (KVAE) 2017 (our exact-inference proof-of-paradigm); Sohn et al.
  (CVAE) 2015 (the prediction-vs-recognition deployment framing).
- **Posterior collapse / KL control / information routing (the mechanism's home):** Kingma et al. 2016
  (free bits); Chen et al. 2017 (Variational Lossy AE — cheapest-channel argument); Alemi et al. 2018
  (Fixing a Broken ELBO); Hafner et al. 2021 (DreamerV2 KL balancing).
- **Particle methods (deployment):** Doucet & Johansen 2009; the FIVO/AESMC filtering-objective family
  (discussed, not adopted).
- **Our own prior work:** SMC Blind Spot (arXiv 2605.12287) — VBPM is its constructive sequel.

*(A fuller citation inventory with verification flags lives in `docs/paper_citations.md`.)*

---

## 8. One-paragraph honest bottom line (for the abstract-writing conversation)

VBPM is a faithful differentiable bar-pointer VAE whose most defensible contribution is a mechanism:
we show that structured sequential VAEs trained by ELBO develop an *emission side-channel* — the
encoder smuggles the target through the cheapest unconstrained latent (here, an unphysically-inflated
tempo), which passes teacher-forcing but caps free-running deployment — and that constraining the
emission to the structurally-meaningful state (an ELBO-exact modeling choice, realized as a
madmom-style parametric emission) removes the pathology. Applied to beat/downbeat tracking on a frozen
frontend, the fix breaks a multi-week deployment wall and makes a learned particle-filter post-processor
competitive with strong peak-picking on clean data (0.87 vs 0.89 beat-F on GTZAN) while additionally
inferring meter, which peak-picking cannot. We are explicit that, under rigorous same-evidence and
fold-honest evaluation, the structured post-processor does not yet *beat* peak-picking on clean
beat-F — the open question we frame is whether structure helps where evidence is weak.
