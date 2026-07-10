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
