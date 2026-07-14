# The emission side-channel: root cause of the deployment wall, and the faithful fix

*Overnight report, 2026-07-10. Evidence from `scratchpad/stack/probe_filter.py` + inline probes
on checkpoint `meter_s0` (full stack: g-prior 0.5, free-bits 0.3 prior-preserving, sawtooth 0.5,
meter CE 0.1, tempo-slope 0.5); ladder runs L0–L4 described at the end.*

## 1. The phenomenon

Every deployment read-out has been capped for weeks, invariant to everything we changed:

| read-out | F (beats) | invariant to |
|---|---|---|
| prior rollout (open loop) | ~0.25–0.40 | data 200→3300 songs, g-prior, free-bits, meter CE, tempo-slope |
| particle filter (closed loop) | ~0.35–0.40 | proposal noise ×0.01–×50, N=400→1600, emission temperature ×3 |
| teacher-forced recon | good | — |

The tempo-slope emission fixed the *rotation ratio* (0.2 → ~1.0) but F did not follow.

## 2. Three probes, one mechanism

**Probe 1 — the likelihood ranks garbage above truth.** For 8/8 validation songs, the trained
emission assigns the ground-truth trajectory (oracle phase from annotated downbeats, oracle tempo)
a *worse* total log-likelihood than a deliberately wrong trajectory (tempo ×1.25, phase +half bar):
margin ≈ −330 nats per 1600 frames. **No inference scheme — filter, SMC, exact enumeration — can
recover the truth from a likelihood that prefers garbage.** This single fact explains the filter
wall and its invariance to every inference dial.

**Probe 2 — phase is nearly decorative.** Sweeping a constant phase offset on the oracle: gauge is
correct (offset 0 is best) but a *half-bar misphase costs only ~22 nats/1600 frames*. The emission
barely depends on the one variable that defines where beats are.

**Probe 3 — the kill shot.** Decode the teacher-forced posterior z, ablating one channel at a time:

| decoder input | P(downbeat) at downbeats | elsewhere |
|---|---|---|
| full z | 0.976 | 0.001 |
| **tempo flattened to its median** | **0.001** | 0.001 |
| phase flattened | 0.937 | 0.001 |

Event reconstruction lives **entirely in the tempo channel**. And that channel is not a tempo:
posterior log-tempo sits at **6.1** (advance e^6 ≈ 400 rad/frame — physically absurd) with
per-frame wiggles |Δ| ≈ 0.07 that *are* the beat grid, Morse-coded.

## 3. The mechanism, stated precisely

The ELBO's factorized KL terms price each latent's deviation from its prior — but they do not
assign *semantics*. Semantics live in the structure of the generative functions. The transition
respects that structure (φ_t = φ_{t−1} + e^{φ̇}); the decoder, an MLP on the concatenated
(φ, φ̇, m), respects nothing. Reconstruction gradient flows to whichever input is cheapest to
shape into an event signal, and the prices are themselves learnable: the prior tempo σ comes from
a trained head, which inflated to **σ ≈ 0.57 nats/frame** (a ±77 % tempo kick per frame; real
music: ~0.1 %). At that σ, posterior tempo wiggles are KL-free. Meanwhile phase was already booked
by the sawtooth loss. So SGD routed all event information through the unbounded, KL-cheap,
otherwise-unemployed tempo channel.

At deployment the prior produces smooth physical tempo → no wiggles → the decoder outputs its
base rate → open loop capped; and the filter's evidence is the same broken likelihood → closed
loop capped. The phase KL never fell (~300) because phase never needed to carry information.

This retro-explains: the "latent unused" diagnosis (2026-06-25), the historic tempo blow-ups
(unbounded RW was the symptom, the decoder's pull was the cause), why the Kalman-VAE escaped the
wall (its structured Gaussian emission offered no cheap side channel), and why the synthesis arm
worked (autocorr+sawtooth forced tempo physical, closing the channel by brute force).

## 4. Why per-song tempo error looked like the bottleneck (and still matters)

Quantified the same night (`data_value_curve.py`): tempo is learnable from the frozen features to
6.4 % median error with **100 songs** and improves *zero* through 3300 → raw song count is not a
lever. But error concentrates where the train-tempo histogram is thin (oct-acc 0.281 sparse vs
0.393 covered) → **tempo-stretch augmentation** (Beat This's own scheme, borrowed as-is; grid
building overnight to `/home/sogang/mnt/db_2/jaehoon/vbpm_aug_cache/`) is the data lever.
Independent of, and complementary to, the emission fix.

## 5. The faithful fix — no change to the ELBO

**The fix is a modeling choice, not an objective change.** Specify the emission as

> p(b_t | z_t) = p(b_t | φ_t, m_t)

i.e. the event decoder's arguments exclude tempo. The ELBO stays the *exact* ELBO of this
generative model — nothing auxiliary, nothing annealed, no detached gradients. And it is arguably
*more* faithful to the bar-pointer lineage than what we had: in Whiteley/Cemgil/Godsill and in
madmom's DBN, the observation model depends on pointer **position** (and meter), never on tempo —
tempo only parameterizes the transition kernel. Our spec (docs/ELBO_for_DBN.md eq. 5) writes
p(b|z) with z the full state; the lineage reading resolves the ambiguity in favor of (φ, m).

Two further degrees of the same principle, both equally ELBO-exact:

- **Phase-only emission** p(b|φ): also cuts the meter one-hot (which could Morse-code events the
  same way once tempo is cut).
- **Parametric emission** (madmom-style): fix the *functional form* —
  logit_beat = a_b + softplus(c_b)·exp(k(cos(bpb·φ)−1)), logit_db likewise at the fundamental,
  with bpb = Σ_k m_k·(k+1) from the SOFT meter latent (meter stays consequential). Five scalars;
  structurally incapable of smuggling. At smoke scale it fit the event channels ~2.5× better than
  the MLP from step 1 — the MLP's flexibility was never buying event fit, only the wrong route.

**The companion cut (L4): fixed physical prior transition scales.** The scale heads are the leak's
enabler — the model inflates prior σ/ρ so posterior wiggles cost nothing. Freezing the *prior*
transition scales at physical values (σ = 0.005 nats/frame, phase concentration = 99 ≈ ρ 0.99) is
again a spec choice of p(z_t|z_{t−1}, h) — the ELBO is exact for it, and the KL now actually
prices side channels (smoke test: phase KL 15 → 907 the moment scales were fixed). Posterior
scales stay learned. (Note: under L4 the grad-reach check prints ZERO for the two scale heads —
expected, they are bypassed, not starved.)

## 6. The emission ladder (running overnight)

| rung | emission sees | scale | run |
|---|---|---|---|
| L0 | (φ, φ̇, m) MLP — broken baseline | 3300 songs / 2000 steps | SCALE_s0/s1 |
| L1 | (φ, m) — tempo cut | 3300 / 2000 | DCUT_s0/s1 |
| L2 | φ only | 200 / 700 | L2_phaseonly_s0 |
| L3 | parametric cosine bump | 200 / 700 | L3_parametric_s0 |
| L4 | L2/L3 + fixed prior scales | 200 / 700 | queued behind L2/L3 |

Each rung is judged on the **mechanism probes** (oracle-vs-wrong margin must flip positive;
tempo-flatten ablation must stop killing recon; posterior tempo must be physical; phase KL must
carry the information), with F secondary (probe-scale F is seed-noisy). Winner gets second seeds
and a full-scale run.

Known gaps this ladder does NOT close (next axes if a residual survives): emission trained on
event targets but filter scores frontend activations (calibration gap); filter read-out still
hardcodes 4 beats/bar; per-song tempo coverage (the augmentation lever).

## 7. Results

**L2 phase-only (probe scale, seed 0, final):** the cut WORKS mechanically — posterior log-tempo
median **−3.39** (physical; was +6.1 at L0) and the oracle-vs-wrong margin flips **positive**
(+17.8 nats, 8/8 songs; was −330). But the MLP emission then collapses toward base rates
(P(db) 0.011 at downbeats vs 0.009 elsewhere): with the cheap wire cut, the event message mostly
*dies* rather than rerouting through phase. F 0.223 (probe scale, 1 seed — noisy).
**Reading: the side-channel cut is necessary but the flexible MLP will not spontaneously build a
sharp phase→event map; the emission needs structure (L3 parametric / L4 fixed scales).**

**L1 DCUT (full scale, step 400 interim):** recon unchanged after cutting tempo → the encoder
reroutes (meter KL balloons to 56–85 nats in the L0 SCALE arms; DCUT_s0 phase KL *fell* to 21) —
confirms meter is the second wire and L1 alone is insufficient.

**L3 parametric (probe scale, seed 0, final):** the emission is finally HONEST — oracle-vs-wrong
margin **+125.5 nats** (8/8; 7× L2's MLP), posterior tempo physical (−3.44). But the posterior
phase does not yet track (recognition rot 0.00 throughout; TF decode flat) — with the side
channels gone, the ENCODER now has to genuinely learn phase alignment, and probe scale
(200 songs / 700 steps) isn't enough. The burden moved from "decoder cheats" to "posterior must
learn" — the correct place for it to be. F 0.264 (probe scale).

**L4 (fixed physical prior scales σ=0.005, c=99; parametric + phase-only): launched.** Rationale
sharpened by L3: tight prior scales make posterior phase wiggles/misalignment properly expensive,
adding direct pressure on the encoder to produce smooth aligned phase.

**L4 (fixed prior scales, probe scale, seed 0, finals):** parametric arm matches L3's honest
likelihood (margin **+130.5**, 8/8; tempo −3.41) and the scale-fix works as intended — posterior
phase misalignment now costs real KL (phase KL 1694). The MLP phase-only arm FAILS again
(margin −22, P(db) flat 0.006): second independent confirmation that a flexible MLP will not
learn the phase→event map on its own.

**Ladder verdict:** the PARAMETRIC EMISSION is the winner. Emission-side pathology is fixed
(likelihood prefers truth; tempo physical; meter sets the bump frequency, finally consequential).
The one remaining, cleanly isolated blocker: the RECOGNITION network does not yet align its phase
posterior to events at probe scale (TF decode flat; recog rot 0.00). This is a trainability
question, not a design question. Decisive test queued: L3FULL_s0/s1 — parametric emission at
full scale (3300 songs / 2000 steps / 118-song eval).

**SCALE (L0) finals, 118 songs:** beat 0.344/0.317, downbeat 0.036/0.180. **Meter headline:
s1 acc 0.97 with genuine 3-vs-4 discrimination ({4:98, 3:20})** — the frame-summed meter CE scales;
s0 collapsed to constant-4 (0.86 = base rate): meter seed-variance persists.

**Full-scale probes overturn one detail and sharpen the story:** the SCALE checkpoints (trained
WITH the tempo-slope emission, unlike the −330-nat meter_s0) already PASS probes 1&3 — margin
+48/+61, tempo physical, trained tempo σ down to 0.067, and in s1 events genuinely ride on phase
(flatten phase → discrimination dies; flatten tempo → survives). The tempo-slope emission had
already closed the Morse channel. What remains is an ANEMIC emission: P(db) ≤ 0.03 even
perfectly on-beat (the old cheating ckpt: 0.976). Filter on the honest ckpt: best 0.399 —
marginally past the old 0.35 wall but still capped, because per-frame evidence is ~0.01-0.03.

**The binding constraint, final form: posterior phase alignment precision.** The decoder's
confidence mirrors the posterior's alignment quality — a phase-conditioned bump can only be as
sharp as phase is aligned. And the alignment supervision has been hiding a mis-set parameter in
plain sight: the sawtooth weight doubles as the von Mises concentration, **κ = 0.5 — nearly
uniform**, i.e. "alignment matters only to ±90°." For 70 ms beat precision κ must be in the tens.
Raising κ is fully ELBO-faithful (it is the emission's concentration parameter, same status as
the WC ρ). Test launched: KAPPA8_s0 (parametric emission + sawtooth κ=8, probe scale, GPU 0).

**DCUT (L1) finals, 118 songs:** 0.221/0.229 beat. Probes: s0 = the night's best honest emission
(P(db) 0.253 vs 0.003, 84:1; margin +65; tempo unused as designed) but PARTIALLY METER-CARRIED
(flatten meter → 0.042) — the reroute, caught in the act (its meter KL ballooned to 59, recon 87.7,
the night's lowest). s1 emission dead flat. Seed lottery persists.

**KAPPA bracket (parametric + sawtooth κ, probe scale):** κ=8 PROVED THE POSTERIOR CAN ALIGN —
**recognition downbeat F 0.699, the campaign's best downbeat number** — and mid-run had healthy
rot≈1 on both paths... then the prior rollout collapsed at step 700 (rot 0.01). κ=32 worse
(prior rot 35). Raising κ fixes posterior alignment but destabilizes prior-posterior coupling
(the teacher-forcing gap resurfacing at the last joint). κ=2 and κ=4 launched to bracket.

**Aug cache complete:** all 20 tempo-stretch variants (±4..±20%, base+harmonix), 121 GB,
runner integration in (GP_EXTRA_TRAIN_DIRS).

**Synthesis at 06:30:** every ingredient now demonstrably exists — honest emission (parametric /
DCUT_s0-style), aligned posterior (κ≥8: recog db 0.699), physical tempo (slope emission) — but
never yet simultaneously in one stable run. The problem has shrunk from "mystery wall" to
"stabilize the κ-strength / prior-coupling trade-off," with the κ bracket and L3FULL running.

**κ bracket finals (probe scale):** κ=2 is the winner — STABLE (no κ=8 collapse; final rot 1.30),
all mechanism probes PASS, and crucially the FIRST checkpoint of the campaign whose event signal
travels through phase alone: P(db) 0.132 at downbeats vs 0.065 elsewhere, unchanged when tempo or
meter is flattened, killed when phase is flattened. Margin +123.6 (8/8), tempo −3.62. Its
recognition downbeat F hit 0.653 mid-run. κ=4's posterior collapsed onto the prior (flat decode);
κ=8/32 destabilized the prior. F at probe scale still ~0.29-0.31 — the sharpness (0.132) is real
but modest; scale is the remaining dial.

**Culmination run launched (AUGK2_s0/s1):** κ=2 + parametric emission + full scale + the complete
tempo-aug pool (20 stretch variants, ~7300 train entries) — the first run combining every fixed
ingredient: honest emission, phase-carried events, physical tempo, coverage-targeted data.

**L3FULL finals (118 songs):** beat 0.265/0.265 — and the parametric emission KILLS THE SEED
LOTTERY (identical across seeds, vs 0.34/0.32, 0.22/0.23, 0.39/0.12 in MLP arms). Probes: honest
(margins +106/+104, tempo −3.35/−3.37) but the posterior did NOT align (TF decode flat 0.019) at
κ=0.5. Both meters collapsed to constant-4 — with the parametric emission, wrong meter now hurts
the beat bump, so the model retreats to safe 4; raising meter-CE weight is a follow-up lever.

**Cross-grid inference:** posterior alignment appears exactly and only when κ rises (κ=0.5 at any
scale: flat; κ=2: 0.132 vs 0.065; κ=8: recog db 0.699) — **κ drives alignment, scale does not.**
AUGK2 (κ=2 + parametric + full scale + aug pool) is therefore the decisive configuration.

## 8. THE WALL BREAKS (2026-07-10 ~11:00)

On the κ=2 parametric checkpoint (KAPPA2_s0 — trained on just 200 songs), the particle filter
with DBN-like proposals (sigma×0.01: near-constant tempo per particle; conc×50: near-deterministic
phase advance; evidence temperature 3):

> **F 0.666 mean over 8 val songs — per-song 0.91 / 0.81 / 0.79 / 0.75 / 0.70 / 0.65 / 0.49 / 0.23**

versus the 0.35–0.40 cap that held for weeks across every checkpoint and every dial setting.
Emission phase-discrimination is real (P(beat) 0.074→0.296 over the bar). The visualization
(scratchpad/stack/phase_vs_gt.png vs phase_vs_gt_sharp.png) shows it directly: with trained
proposal noise the MAP trajectory is per-frame noise (trained ρ≈0.39 → huge per-frame kicks);
with sharp proposals it locks onto the ground-truth sawtooth for bars at a time.

Also visible in the BEFORE figure: the PRIOR rollout is far better than its F suggests — clean
sawtooth, right period, small drifting offset. Its ~0.28 F is the 70 ms tolerance punishing a
slightly-wrong clock, exactly the error closed-loop correction fixes.

Complete verified chain: honest emission (parametric) + aligned posterior (κ=2) + smooth
proposals (deploy dial now; L4 fixed scales as the train-time equivalent) → working closed-loop
inference. Remaining engineering: filter read-out still hardcodes 4 beats/bar; proposal scales
should come from the model (L4) rather than a deploy-time override; smoothing > filtering;
AUGK2 (full scale + aug) probes run automatically on save.

## 9. Downbeats fixed the same way (2026-07-10 ~10:15)

Convergence probes explained the low downbeat F exactly: the filter locks phase MODULO a
beat (23 ms median beat error where locked) but the bar gauge stays 4-way ambiguous — beat
evidence is bpb-fold symmetric by construction; only the (weak: peak P 0.137, once-per-bar)
downbeat bump breaks the symmetry; and sharp proposals lock each particle's gauge at birth, so
a resampling accident kills the correct gauge irrecoverably (madmom avoids this by enumerating
all bar positions). Two deploy-time fixes, both standard filtering practice, no ELBO change:

1. **Stratified lane init** — particles born at all quarter-bar offsets (offsets from EACH
   PARTICLE'S OWN meter latent, no hardcoded 4);
2. **Downbeat evidence weighting** (×3) — per-channel observation weighting, as madmom does.

Result on KAPPA2_s0, meter-latent read-out end to end: **beat F 0.665-0.709, DOWNBEAT F 0.747**
(was 0.05-0.18 all campaign). Beat error histogram (docs/figures/beat_error_hist.png): filter
median error 23 ms where locked; prior rollout is a ±300 ms smear with 31 % of beats just
outside the 70 ms window (the drifting-clock signature — mass the closed loop converts).

## 10. Fixed-deployment audit of the whole campaign (2026-07-10 ~11:00)

Every saved checkpoint re-scored with the fixed filter (16 val songs, N=800; full table in
scratchpad/stack/logs/probe_all_fixed.log). Headlines:

- **Old vs fixed metric correlation: Spearman rho = +0.12 (p=0.6, n=18) — statistically zero.**
  Top-3 disjoint: old picked {strict_s0, meter_s0, SCALE_s0} (all 0.34-0.39 honest); honest picks
  {L3_parametric 0.822, KAPPA8 0.821, KAPPA2 0.817} — all ranked mid-to-bottom by the old metric.
  Weeks of F-based A/B decisions were made on read-out noise.
- **Parametric emission is load-bearing, honestly measured**: every healthy parametric arm 0.77-0.82
  Bayesian beat F; every MLP arm (any input mask, ± sawtooth, ± free-bits) 0.24-0.52.
- **The κ=8 "prior collapse" was also a broken-lens verdict**: KAPPA8 filters at 0.821 (tied best);
  only its open-loop rollout was broken. κ=32 still 0.703. Usable κ range is wide.
- **Remaining training-side fragility: seed-level half-tempo locks** (L3FULL_s1 0.363 vs s0 0.810).
- The Bayesian (ensemble) wrap read-out beats the MAP particle consistently (~+0.13).

Campaign honest leaderboard (Bayesian beat/db): L3_parametric 0.822/0.782, KAPPA8 0.821/0.782,
KAPPA2 0.817/0.777, L3FULL_s0 0.810/0.774, L4param 0.769/0.746. Frontend peak-pick ceiling: 0.889.

## 11. SMC_MIREX zero-shot (2026-07-10 ~11:40)

Fixed filter, all 217 songs: **KAPPA2_s0 Bayesian beat F 0.694 (median 0.714, 53 % of songs >= 0.7)**;
L3FULL_s0 0.680. Reference points: Beat This itself scores ~0.55 on SMC; the user's mission target
is >= 0.7. Same frontend activations, different inference layer — the blind-spot paper's thesis
(on SMC the POST-PROCESSING is the binding constraint) realized: the continuous learned bar-pointer
filter has no tempo grid, no prior floor. Achieved zero-shot from the 200-song checkpoint, BEFORE
tempo-aug training (AUGK2), rubato-adaptive sigma, octave lanes, smoothing, or ASAP data.

CAVEAT under verification: an untrained-model control is running — the Bayesian read-out leans on
the frontend evidence (a 20-step smoke model already gets 0.76/0.80 in-domain bayes), so every
claim needs the "frontend + filter machinery alone" baseline subtracted. MAP read-out reflects the
learned model far more. Also: filter dials were tuned on the 16-song in-domain val, applied to SMC
unchanged (no SMC-specific tuning — genuine zero-shot).

## 12. The untrained-model control (2026-07-10 ~13:00) — read before quoting any number

| | in-domain (16) bayes beat/db | SMC (217) bayes beat |
|---|---|---|
| UNTRAINED model, fixed filter | 0.767 / 0.778 | 0.640 (median 0.667) |
| best trained | 0.817–0.822 / 0.774–0.782 | 0.694 |
| learning's contribution (bayes) | +0.05 / ~0.00 | +0.05 |

The Bayesian read-out's strength is mostly (a) frontend evidence + (b) the fixed-filter machinery +
(c) the parametric emission's FORM (whose init is already a well-voiced bump) — an untrained
VBPM ≈ a hand-designed neural-madmom at 0.77/0.64. The MAP read-out reflects learning far more
(0.353 untrained → 0.68 trained in-domain). EVERY claim must quote the untrained baseline.

Strengthening flip side: the UNTRAINED system already beats Beat This's own post-processing on SMC
(0.640 vs ~0.55) — the blind-spot removal is ARCHITECTURAL (continuous tempo, no grid/floor,
stratified gauge, balanced evidence), provable without any training. Learning's +0.05 is today's
floor (posterior alignment, aug, scale still converging — AUGK2 pending).

## 13. SMC FULL-LENGTH: 0.700 — ⚠ CONTAMINATION CAUGHT (user, ~16:00), correction in progress

**⚠ The frontend (final0) was trained on ALL datasets except GTZAN — INCLUDING SMC** (Beat This
README §Models). So the 0.700 below is "SMC-trained frontend + SMC-zero-shot inference layer,"
NOT comparable to the fold-held-out published numbers as originally claimed. Correction running:
fold-honest re-extraction (each SMC song's activations from the fold checkpoint that held it out —
the repo-prescribed CV protocol; fold ckpts + 8-folds.split already on disk from the blind-spot
paper work) → smc_rich_foldhonest cache → same filter eval. The GTZAN evaluation remains clean
with final0 (GTZAN fully excluded from its training) and becomes the bulletproof OOD claim.
The same caveat labels our in-domain val numbers (final0 saw those datasets; fine for internal
comparisons against the same frontend's peak-pick, not for cross-paper claims).

Original (final0-contaminated) result, kept for the delta:

All 217 excerpts, full 40 s, standard mir_eval 70 ms: **beat F 0.700 mean / 0.714 median**
(52 % of songs >= 0.7). Published, 8-fold-CV-TRAINED-on-SMC numbers (BeatFM Table II):
TCN 55.2, Beat Transformer 59.6, MERT+DBN 60.1, BeatFM 61.3. Ours: zero-shot, 200-song
checkpoint (KAPPA2_s0), dials tuned only on in-domain val. **+8.7 over SOTA; the >= 0.7 mission
target is met.** The blind-spot thesis quantified: on SMC the post-processor is the binding
constraint. Pending to bulletproof: full-length untrained control (running); AUGK2 ckpt may add
(tempo-aug targets SMC's coverage gaps); held-out dial validation for the paper.

## 14. NOSAW + full-length control (2026-07-10 ~13:20) — the recipe simplifies, learning vindicated

**NOSAW (sawtooth weight = 0, parametric + slope + meter CE, probe scale):** filter Bayesian
**0.836/0.795 (s0) and 0.833/0.800 (s1)** — the campaign's best, seed-consistent, vs κ=2's
0.817/0.777. The sawtooth LOSS is unnecessary (within noise, direction favors dropping it);
the maximal-faithfulness objective (pure ELBO, every remaining term an emission) wins.
Footnotes: sawtooth TARGETS remain as plumbing for tempo-slope; MAP readout slightly lower
(rot 0.57) — sawtooth may matter for MAP-trajectory quality only.

**Full-length untrained SMC control: 0.500** (vs 0.640 at 18.6 s — the untrained system loses
lock as excerpts lengthen; trained holds 0.700). Attribution on the headline benchmark:
architecture alone 0.50 (below all published systems), **learned dynamics +0.20**. The earlier
"+0.05 learning contribution" was a short-window artifact. In-domain learned delta now
0.767 → 0.836 (+0.07, NOSAW). Learning is decisive at long horizon.

## 15. Minimality ladder complete (2026-07-10 ~15:45)

| removed (vs κ=2 full stack, bayes 0.817/0.777) | filter bayes beat/db | verdict |
|---|---|---|
| sawtooth (NOSAW) | 0.836/0.795 | UNNECESSARY |
| g-prior | 0.831/0.780 | UNNECESSARY |
| tempo-slope | 0.837/0.774 | bayes-unnecessary; prior side degrades (rot 0.58/0.44 half-tempo lean) |
| meter CE | 0.772/0.734, meter → nonsense classes | NECESSARY (and now meter-consequential: hurts beats) |
| free-bits (strict ELBO) | 0.415/0.778 | NECESSARY |

Minimal faithful core: **parametric emission + meter CE + free-bits(pp)**, with either sawtooth or
slope as the tempo-grounding term (each alone suffices for the bayes filter; the "neither" cell —
MIN_neither_s0 — is running to close the map). Every survivor is an emission or a KL-floor;
nothing auxiliary in the un-principled sense remains.

## 16. THE FOLD-HONESTY RECKONING (2026-07-10 ~16:15) — read this before any comparative claim

Fold-honest SMC (each song's activations from the fold ckpt that held it out): **filter 0.462**
(was 0.700 contaminated). Same-evidence peak-pick: **0.599**. Published fold-trained systems:
0.55–0.61. And in-domain: peak-pick on our (final0-memorized) 16-song val = **0.974/0.986** vs
filter 0.836/0.795.

Conclusions, stated plainly:
1. The filter has NOT beaten same-evidence peak-picking in any trustworthy measurement.
2. On honest weak evidence (fold-honest SMC) the filter's continuity assumptions HURT (−0.14) —
   the blind-spot paper's mechanism, measured against our own system.
3. The SMC-SOTA claim is RETRACTED (frontend memorization). GTZAN (excluded from final0
   training) is the only clean cross-system evaluation currently possible with this cache.
4. Internal A/Bs (ladder, minimality, κ, EB) remain valid — all arms share the same evidence.
   The side-channel mechanism findings are untouched (they never depended on baselines).

Required rebuild: fold-honest val caches (re-extract from audio with stem→fold tracking; cache
format currently drops identity). The real open question the honest data poses: can a structured
post-processor HELP on imperfect evidence? Levers: rubato-adaptive sigma (EB arms), expressive
training data (ASAP), aug pool, evidence calibration. Burden of proof now on them.

Value propositions that survive regardless: meter (peak-picking cannot), joint generative state,
the mechanism paper, OOD behavior (to be measured cleanly on GTZAN).

*(AUGK2 + EB + MIN_neither finals below)*

## 17. Clean-GTZAN verdict + final queue results (2026-07-10 evening)

**GTZAN (993 songs, honest songs, matched final0 frontend — the fairest cell we have):**

| method | beat F | downbeat F |
|---|---|---|
| peak-pick (same activations) | 0.893 | 0.774 |
| trained filter KAPPA2_s0, Bayes | 0.868 | 0.754 |
| trained filter, MAP | 0.721 | — |
| UNTRAINED control, Bayes | 0.615 | 0.548 |

- Filter is −0.025/−0.020 under peak-picking: the honest pattern is now uniform — the
  structured layer has never beaten same-evidence peak-picking in a trustworthy measurement.
  But the gap is 5–7× smaller than fold-honest SMC (−0.14), supporting frontend-shift as the
  dominant cause of the SMC gap (VBPM consumed fold-checkpoint activations it never trained on).
- Learning is real: +0.253/+0.206 over the untrained architecture on clean data.

**EB alternation (professor §6.8, two seeds):** filter 0.829/0.787 and 0.836/0.787 —
score-neutral vs non-EB (~0.83/0.79); prior read-out NOT improved (0.288/0.253). Not a
deployment-gap lever as trained; remaining question is proposal-override dependence.

**MIN_neither_s0** (no sawtooth AND no slope): Bayes 0.367/0.326 — collapse. Minimality map
complete: minimal faithful core = parametric emission + meter CE + free-bits + ONE
tempo-grounding emission (sawtooth OR slope).

**AUGK2 finals** (tempo-aug pool, 118-song val, contamination-labeled for cross-system use):
Bayes 0.841/0.782 (s0), 0.828/0.791 (s1) — beat/downbeat neutral. BUT s0 meter acc 0.97 with
real 3/4 predictions ({4:99, 3:19}) — first arm to predict non-4/4 on real val (s1 all-4s,
0.86; seed-sensitive). Mechanism probe PASS both seeds (oracle margin ~+100 nats, 8/8).

## 18. GTZAN meter confusion (2026-07-11) — first honest win over the always-4 baseline

Per-song meter read-out (filter map_beats_per_bar) vs GT bpb (from downbeat intervals),
993 clean GTZAN songs; GT dist {4:930, 3:54, 2:7, 5:2}, always-4 baseline = 0.937:

- KAPPA2_s0: acc 0.937 = baseline exactly; predicts 4 for ALL 993 songs (non-4/4 recall 0.000).
  Its downbeat 0.754 is phase tracking under an assumed-4 bar, not meter inference.
- AUGK2_s0 (tempo-aug pool arm): acc 0.951 > baseline; non-4/4 recall 0.571.
  3/4: recall 0.667 (36/54), precision 0.590. 4/4: recall 0.976 / precision 0.974.
  2/4: 0/7 (3 of 7 called "3" — senses non-4, wrong class). 5/4: 0/2.
  Confusion (gt,pred): {(2,3):3, (2,4):4, (3,3):36, (3,4):18, (4,3):22, (4,4):908, (5,4):2}

First model to beat the trivial baseline on an uncontaminated benchmark, with genuine
minority-class recall. Data (larger pool incl. more waltz) — not loss reweighting (focal
verdict: no) — is what unlocked it. Peak-picking cannot produce this output at all.

## 19. Blind-spot protocol, fold-honest SMC (2026-07-12 overnight)

| arm | F | CMLt | AMLt |
|---|---|---|---|
| peak-pick | 0.599 | 0.393 | 0.486 |
| KAPPA2_s0 / EB_s0, sharp proposals | 0.452 / 0.454 | 0.156 / 0.142 | 0.190 / 0.176 |
| KAPPA2_s0 / EB_s0, adaptive (learned scales) | 0.267 / 0.265 | 0.000 / 0.000 | 0.008 / 0.007 |

1. OCTAVE BLIND SPOT: solved by architecture. Filter AMLt-CMLt gap ~0.03 vs peak-pick 0.09 --
   continuous log-tempo removes the discretization/octave failure mode (SMC Blind Spot sequel claim).
2. CONTINUITY DRIFT: not solved. Filter errors come in long runs (CMLt collapses 2.5x harder than F);
   sharp proposals forbid re-locking; once ancestry collapses the correct hypothesis is extinct.
3. LEARNED SCALES UNDEPLOYABLE even after EB alternation (CMLt = 0.000 both checkpoints): EB trains
   the prior toward the posterior on MEMORIZED evidence -- the wrong target for deployment calibration.
   Remaining test: fold-honest models' scales (first heads trained on evidence that actually fails).
   If those fail too, the fix is STRUCTURAL: particle rejuvenation lanes (keep every gauge hypothesis
   alive, madmom-style) + offline smoothing, not scale tuning.

## 20. FOLD-HONEST TRAINING: near-parity with peak-picking (2026-07-12 overnight)

First models trained on fold-honest evidence (1,242 songs, folds 0-6), evaluated on the honest
val (177 fold-7 songs, 1600-frame cap, matched peak-pick baseline):

| | beat F | downbeat F |
|---|---|---|
| peak-pick (same evidence) | 0.914 | 0.836 |
| foldhonest_s0 filter Bayes | 0.907 | 0.834 |
| foldhonest_s1 filter Bayes | 0.907 | 0.835 |
| untrained control | 0.788 | 0.794 |

- DOWNBEAT PARITY (-0.001/-0.002), beat near-parity (-0.007), replicated across seeds to three
  decimals. The memorized-evidence-trained checkpoint (KAPPA2_s0) scored 0.664 on this val at the
  same dial family -- fold-honest training closed almost the whole gap. Calibration hypothesis
  CONFIRMED at the training level: models that see real frontend errors deploy through them.
- Learning is real: +0.12/+0.04 over the untrained architecture.
- Open: paired per-song significance; adaptive-scales deployment of these checkpoints (probe
  running); GTZAN transfer.

## 21. Fold-honest model on fold-honest SMC: continuity healed by TRAINING (2026-07-12)

| fold-honest SMC | F | CMLt | AMLt |
|---|---|---|---|
| peak-pick | 0.599 | 0.393 | 0.486 |
| foldhonest_s0 sharp | 0.591 | 0.397 | 0.467 |
| KAPPA2_s0 sharp (sec. 19) | 0.452 | 0.156 | 0.190 |
| foldhonest_s0 adaptive | 0.261 | 0.000 | 0.007 |

- Continuity drift HEALED by calibrated training (CMLt 0.156 -> 0.397, now >= peak-pick): the
  cure was fold-honest evidence, not structural machinery. Rejuvenation lanes demoted from
  "required" to "possible further upside".
- Adaptive-scales question CLOSED (4/4 checkpoints, incl. fold-honest: CMLt 0.000): learned
  transition scales are never deployable as proposals; recipe = sharp physical proposals +
  calibration-trained model.

### 21b. GTZAN transfer of the fold-honest models (2026-07-12)

peak-pick 0.893/0.774 | foldhonest_s1 0.865/0.755 | foldhonest_s0 0.772/0.740 (n=993, cap 8000).
GTZAN evidence is final0-extracted (legitimate; GTZAN in nobody's training), but fold-honest
models trained on FOLD-checkpoint activation statistics -> mild evidence-distribution shift
AGAINST them here (mirror of KAPPA2's fold-honest SMC penalty). Seed divergence (0.772 vs 0.865)
appears exactly under this shift despite 3-decimal seed agreement in-distribution.
LESSON: evidence-distribution match governs transfer; honest scorecard = val parity, SMC parity
(continuity healed), GTZAN -0.03 (best seed).

## 22. Deploy-dial sweep verdict (24 configs, fold-honest val, 2026-07-12)

- BEST pure-filter config = the existing standard dials (T=3, sharp proposals, eps=0):
  EB_s0 0.755/0.837, KAPPA2_s0 0.740/0.816. No deployment dial improves on the pinned recipe.
- Robust-observation epsilon: monotonically NEGATIVE (0 -> 0.1 -> 0.3: 0.755 -> 0.682 -> 0.489);
  hurts DOWNBEATS most (0.837 -> 0.453 at eps 0.1) -- uniform mixing flattens the rare
  gauge-breaking downbeat spikes. The confident-but-wrong fix belongs in TRAINING (fold-honest,
  sec. 20 -- which worked), not in a deploy-time outlier mixture.
- Fusion ~0.92 regardless of filter dials (diagnostic only, per user: not a deliverable).
- Fold-honest-trained models (0.907, sec. 20) dominate every sweep row -- training quality
  beats deployment tuning by an order of magnitude.

## 23. E2E under the repaired recipe: cliff delayed 2x, character changed, not yet survived (2026-07-12)

Warm trunk (final0), lr 1e-5 + L2SP, repaired VBPM recipe, 3000 steps on mel (789 songs).
- Historical cliff (~step 1200: recon 78->1.2, KL->0, shuffle==real) did NOT occur. No leak at any
  point; recon stayed ~160-190; PRIOR read-out stable ~0.30 with physical rotation throughout.
- A DIFFERENT collapse arrived at steps 2400-2800: phase KL 162->2.7 (posterior onto prior) with
  recon intact -- slow-motion, phase-only, no feature-cheating signature.
- Filter verdict on the final ckpt: 0.427/0.584 (mel val, tuned trunk's own activations as
  evidence) -- degraded well below the frozen-frontend fold-honest models (0.907/0.834).
- VERDICT: the side-channel fixes removed the OLD e2e failure mode (trunk feature-cheating) but a
  phase-posterior collapse remains late in training. Next levers: early-stop ~step 2200 (pre-
  collapse checkpoint), phase-specific free-bits pressure, trunk lr decay schedule, fold-honest
  init instead of final0. E2E remains open, now with a much better-understood failure.

## 24. OWN EVIDENCE HEAD: structure now BEATS same-evidence peak-picking (2026-07-12)

User directive: the frontend contributes frozen features ONLY; the filter's observation comes from
OUR ActivationHead (model/activation_head.py; BCE on fold-honest data), never from BT's act2
(act2 survives only inside the peak-pick baseline, which IS Beat This).

| honest val | beat F | downbeat F |
|---|---|---|
| own-head peak-pick (same evidence) | 0.896 | 0.783 |
| foldhonest_s0 filter, own evidence | 0.917 | 0.800 |
| foldhonest_s1 filter, own evidence | 0.922 | 0.807 |
| BT-headed peak-pick (reference)    | 0.930 | 0.855 |

- FIRST positive same-evidence delta in project history: +0.021/+0.026 beat, replicated across
  seeds; beats parsed from z (posterior wrap probability), decoder used only as likelihood.
- Mechanism = the confident-but-wrong thesis, now constructive: BT's pre-sharpened act2 leaves
  nothing for structure to add (filter -0.023 there); an honestly-calibrated BCE head is weaker
  standalone but EXPLOITABLE (+0.021). Structure pays exactly when evidence is not argmax-optimal.
- Residual gap to the BT-headed baseline (-0.008/-0.013 vs 0.930) localizes in evidence-head
  TRAINING DATA (1,242 songs vs their ~16 datasets), not the model. Head data scales trivially.
- Downbeat channel of our head is the weak spot (0.783; sparser positives -- tune pos_weight).
- MERT own-head peak-pick: 0.612 -- weak-but-honest evidence, the ideal regime for the MERT+VBPM
  arm to test whether the structural win generalizes across frontends.

### 24b. Own-evidence OOD validation + the feature-consistency lesson (2026-07-12)

| same-evidence cell | own-head peak-pick | filter(own) | delta |
|---|---|---|---|
| honest val | 0.896 | 0.917-0.922 | +0.02 (both seeds) |
| fold-honest SMC | 0.500 | 0.543 | +0.043 (edge doubles on weak evidence) |
| GTZAN (final0 features = MISMATCHED) | 0.428 | 0.381 | -0.047 (confounded) |

GTZAN cell is a double distribution shift: head AND model trained on fold-checkpoint features,
evaluated on final0 features. The 10k-param head collapses under the shift (0.896 -> 0.428; BT's
own head scores 0.893 on the same songs), and structure amplifies bad+shifted evidence (the
KAPPA2-on-fold-honest-SMC pattern again). Re-extraction of GTZAN with fold0 features running;
matched-features re-test to follow.
LESSON (also a MIREX-submission requirement): ONE checkpoint family end-to-end -- the features
the head/model trained on must be the features at inference. Measured cost of violating: -0.47
evidence quality.

## 25. MERT layer probe (2026-07-12): rhythm peaks mid-stack; layer 6 adopted

Per-layer ActivationHead curves (300-song corpus subset + all SMC; relative ordering is the
signal): corpus val peaks at LAYER 6 (0.643), plateau 3-8, top layer 12 only 0.567 (-0.076),
layer 0 0.525. SMC nearly flat (~0.33) and peaks at the same layer 6 -- NO evidence that SMC
prefers different/complementary layers (user hypothesis not supported at single-layer level);
SMC is uniformly hard for MERT features at this scale. Decision: deterministic layer-6
re-extraction (mert_l6_train_rich, running); layer-weighted combination deferred (no
cross-layer complementarity signal); optional plateau-concat (4+6+8) arm later.
Context: BeatFM aggregates multi-level FM features (their channel attention) -- consistent with
mid-stack rhythm; our top-layer default cost ~0.08 F and plausibly the MERT posterior collapse.

## 26. Wave-2 battery + HARMONIX QUARANTINE (2026-07-12 evening)

Re-anchored per-dataset baselines (421-song val) exposed harmonix as MISALIGNED, not hard:
BT-pp 0.371 on mainstream pop is impossible; per-song act2-peakpick spreads 0.00-0.98 with 50%
<0.3 = per-song offsets (YouTube-re-downloaded audio vs original annotation timelines). ACTION:
722 train + 94 val harmonix records QUARANTINED (cache/acts/quarantine_harmonix*); wave-2 s0
killed and relaunched on the clean 2,287-song corpus; wave-2 evidence head must be retrained
post-quarantine (its 0.724 own-pp ALL row was trained WITH poisoned labels). Recovery path
(later): estimate per-song offset by cross-correlating act2 with annotation grid; realign or drop.

Battery gems (clean rows): own head BEATS BT's 16-dataset head on every expressive domain it
trained on -- ASAP 0.603 vs 0.507, rwc_classical 0.671 vs 0.620, rwc_jazz 0.696 vs 0.672.
Wave-2 data pays exactly where aimed. MERT-v2 (layer 6): phase KL ALIVE at step 300 (224 vs
v1's ~45) -- the posterior-collapse fix is holding so far.
