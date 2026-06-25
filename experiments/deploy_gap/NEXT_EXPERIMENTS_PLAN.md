# Next-Experiments Plan — Bar-Pointer DVAE (multi-agent ideation, 2026-06-25)

Source: 6-lens ideation workflow (30 ideas) → adversarial critique (14 survived / 16 killed) → synthesis.
Binding problem (established): latent is identifiable one-step (tf_prior_lat=0.48) but the multi-step
open-loop rollout DESYNCS because the phase prior mean is parameter-free & audio-blind (fr_lat=0.018).
Two open questions gate everything: (Q1) pure phase-drift vs also tempo/octave drift? (Q2) can phase-lock
be recovered WITHOUT crossing the audio-blind-prior faithfulness line, or is Tier B necessary?

## TIER 0 — decisive gate (FAITHFUL, eval-only, <1 day) — RUN FIRST
- **0.1 Oracle-state injection.** During free-run, every K seconds overwrite phase (arm A) or phase+tempo
  (arm B) from GT, then keep free-running. Guards: K=inf must reproduce 0.018; K=every-frame ~0.48; plus
  one coarse K≈4s, A vs B. Discriminates phase-drift (A recovers) vs tempo/octave-drift (needs B) vs
  read-out fault (neither). The A−B gap PRICES the entire Tier-B effort. Report downbeat-F + meter-acc + P/R.
- **0.2 Free PF/SMC measurement** on the latent_only+widen checkpoint (deploy-time audio-only refinement
  ceiling). Expected ≈ open-loop → confirms emission signal too weak to localize phase open-loop → motivates
  a PRIOR-MEAN fix over a deploy-time inference fix.

## TIER 1 — cheapest attempts to actually FIX the rollout
- **1.1 Damped-resonator tempo prior** (MAJOR-DEV): phidot relaxes to a learned per-song tau_ref (limit
  cycle), bounds Var, NO audio in the mean. a=0 known-answer must reproduce fr_lat~0.018. Likely outcome:
  bounds tempo but fr_lat stays ~0.02 → that itself PROVES phase (not tempo) is the bottleneck.
- **1.2 Autoregressive event decoder** (MINOR-DEV): p(y_t|z, y_{t-1:t-L}) closes the loop via the OBSERVATION
  channel (keeps audio-blind prior). L=0 reproduces non-AR. Likely degenerates to a self-reinforced
  metronome → headline = oracle-past−self-past gap + IOI-autocorr metronome test, NOT raw fr_lat.
- **1.3 Renewal/survival IOI likelihood** (MINOR-DEV) — STRONGEST "latent earns its keep": add a survival
  hazard (scale=1/phidot) to the widened occupancy BCE (ADD, ramp weight — don't replace). A per-frame
  discriminative frontend structurally CANNOT emit a calibrated inter-event-interval density. Controls:
  oracle-tempo IOI slope≈1; shuffle-tempo → held-out NLL worsens. Scope: revives TEMPO only, not phase.

## TIER 2 — phase-lock fix that CROSSES faithfulness (USER MUST CHOOSE; run after 0.1)
- **2.1 Bounded innovation prior**: small g_phi(h)=eps*tanh(.) correction to mu_phi (eps≈0.3 rad ≪ beat).
  eps=0 reproduces 0.018. Copy-detector eps-sweep: load-bearing = PLATEAU at small eps; relay = RAMP.
  Mandatory disambiguators: (i) warm-up ablation (zero g_phi after warm-up — dynamics should hold lock);
  (ii) circ-corr + integrated |Σ Δφ|. MUST re-run the copy-detector on the Beat-This [T,512] frontend.
- **2.2 FIVO / scheduled-sampling 2×2** {g_psi on/off}×{ELBO/FIVO}; run scheduled-sampling first (near-free),
  escalate to FIVO only if it plateaus low. N=1 reduces to ELBO; report ESS.

## TIER 3 — does the latent earn STRUCTURE the frontend can't? (read one-step/TF-posterior, not dead free-run)
- **3.1 Coarse/song-global tempo-phase latent**: one state per ~1–2s → O(T/window) drift steps. Then z-only
  downbeat head vs h-only vs Beat-This peak-pick. Win = within-model z≫h contrast (pre-registered).
- **3.2 Octave-stability audit** (FAITHFUL) on the LOAD-BEARING latent_only model (one-step/TF tempo, not the
  dead Acc1=0.009 free-run) vs peak-pick & madmom-DBN on octave-error subset.
- **3.3 GT-tempo observer** as a known-answer control: is tempo dead because UNOBSERVED (fixable) or
  NON-IDENTIFIABLE (stays dead with GT)? Drop the frontend-derived arm unless it BEATS the frontend.

## TIER 4 — orthogonal (uncertainty): 4.1 calibration of free-run κ/σ vs frontend confidence baseline. Low priority (likely finding-#8 balloon null).

## RECOMMENDED ORDER
1. 0.1 oracle-state injection + 0.2 PF — cheap, faithful, prices everything (<1 day).
2. 1.3 renewal/survival IOI — cleanest "latent earns what the frontend can't"; revives tempo (not phase).
3. Branch on 0.1: if phase-only recovery → 1.2 AR decoder (faithful) before paying for Tier 2; if AR fails
   the metronome test (likely) → 2.1 bounded-innovation prior with the Beat-This copy-detector + warm-up
   ablation, as the deliberate user-approved faithfulness crossing.

## FALSE-START FLAGS (stated up front)
1.2 AR likely degenerates to a metronome; 1.1 resonator likely bounds tempo but leaves fr_lat~0.02 (useful
proof phase dominates); 2.1's small-eps plateau may be a slow integrated copy on Beat-This (hence warm-up
ablation); 4.1 likely uninformative; all Tier-3 likely lose to peak-pick on clean data — that is expected,
the falsifiable claim is the within-model z≫h contrast.

## NOTE: agents found EXISTING infra in the (old) repo: models/svt_core.py already has scheduled_sampling +
an audio-conditioned mean (g_psi); pf_eval_smc.py exists. These are the OLD codebase, not faithful/ — but
0.2 and 2.2 can reuse them rather than reimplement.
