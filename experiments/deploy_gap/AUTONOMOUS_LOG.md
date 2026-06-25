
# ============================================================
# AUTONOMOUS RESUME — 2026-06-25 04:45 KST
# ============================================================

## [04:45] Task 1 — Results already on disk

### Overshoot D=4 (scratchpad/runs/os/d4)
| metric | strict baseline | overshoot D=4 | verdict |
|---|---|---|---|
| beat_F_oracle | 0.331 | **0.400** | modest +0.07 |
| downbeat_F | 0.104 | 0.119 | ~flat |
| tempo_Acc1 | 0.000 | 0.000 | unchanged (dead) |
| bpm_mean | ~ | 747.8 | still absurd |
| bpm_std | 147.7 | **45.9** | tightened 3.2x |
| dec_dz | 0.035 | 0.043 | decoder still ignores z |
| kl_p / kl_t | 0.028/1.88 | 0.014/22.4 | phase KL→0 |

INTERPRETATION (matches pre-registered prediction): overshoot trains the prior
sigma/kappa heads only (means are parameter-free), so it TIGHTENED the tempo spread
(bpm_std 147.7->45.9) and nudged beat-F up, but did NOT fix the cause — tempo still
dead (Acc1=0), bpm_mean still 747 (absurd), decoder still ignores z (dec_dz 0.043).
Symptom relief, not a cure. Confirms the two-axis story.

### He-2019 (runs/he) — CONFIRMED NEGATIVE
strict:     posterior_beatF 0.046 (still collapsed), freerun 0.316, sigma 0.273(UP), kappa 0.046
latentonly: posterior_beatF 0.000,                  freerun 0.354, sigma 0.114,     kappa 0.106
Aggressive encoder did NOT un-collapse the posterior nor move free-run. Axis-1 fix
does not help an axis-2 (deployment) problem. As predicted.

## [04:5x] Task 0 — AUTHORIZED BUG FIX (faithful/evaluate.py) — DONE & COMMITTED
- evaluate() now: beat_phase = beats_from_barphase (m subdivisions) vs BEAT ref;
  downbeat_phase = 2pi wraps vs DOWNBEAT ref; meter m via _estimate_meter (median bpb).
- Old metric scored bar-wraps vs beats => capped at ~1/m. Verified fix with
  scratchpad/verify_eval_fix.py on ideal latents: beat_phase 0.976, downbeat_phase 0.959 (PASS).
- Committed 22a7945 on faithful/strict-elbo.

## [~05:2x] Task 4 — FOUR-WAY F (strict_elbo/final.pt, 16 songs) — DECISIVE
  tf_post_dec  = 0.000   (teacher-forced posterior, DECODER read-out = what ELBO recon optimizes)
  tf_post_lat  = 0.024   (teacher-forced posterior, latent subdivision read-out)
  tf_prior_lat = 0.054   (prior one-step mean seeded by posterior mean)
  fr_dec       = 0.000   (free-run decoder)
  fr_lat       = 0.328   (free-run latent subdivision)

KEY: free-run latent (0.33) > teacher-forced posterior (0.024/0.000). The posterior is WORSE
than the prior's free rollout. So the "0.33" is NOT inference tracking beats -- it's the
bar-pointer free-running at ~the right average rate (a learned-tempo metronome) catching ~1/3
of beats by periodicity. The DECODER is dead both TF and free-run (0.000). This is the fork the
blind agent named: tf_post_dec=0.000 means the model NEVER FIT THE DATA -- "posterior collapse"
is a misnomer; the decoder/likelihood failed. Caveat: beats_from_activation uses thr=0.5 and the
decoder maxes ~0.48 (below threshold) -> the overfit-one-song test (decmax trend) checks whether
the decoder CAN fire at all.

## [~05:2x] Task 5 — DECODER 2x2 (free-run + sensitivities, 400-step ablate)
  full       : beat_F 0.331  dec_dz 0.035  dec_dh 0.059  post_dctx 0.12
  latent_only: beat_F 0.407  dec_dz 0.415  dec_dh 0.000  post_dctx 0.104
Full decoder barely uses z (0.035) OR h (0.059) -- both tiny: the decoder is DEAD, not taking an
audio shortcut. Forcing latent_only raises dec_dz to 0.415 and nudges free-run to 0.407.
Posterior barely reads audio (post_dctx ~0.1) in both -> inference-side collapse confirmed.
(Teacher-forced decoder F for the 2x2 comes from Task 2/4: it is ~0.000.)

## [~05:5x] Task 4 — overshoot D=4 four-way (same pattern)
  tf_post_dec=0.000 tf_post_lat=0.000 tf_prior_lat=0.000 fr_dec=0.000 fr_lat=0.404
  Identical story: only the free-run latent is nonzero (periodicity). Overshoot improved the
  free-running metronome (0.40) but inference is still 0. Confirms it doesn't fix the cause.

## [~05:5x] Task 6 — TEMPO VARIANCE-GROWTH (task6_tempo_var.png)
  strict:    sigmabar=0.244  emp_Var[log tau]@T=49.6  RW_pred(cumsum sig^2)=47.7  ratio=1.04
  overshoot: sigmabar=6.69   emp_Var@T=28766          RW_pred=35820              ratio=0.80
  CONFIRMED: empirical tempo variance == the random-walk prior's own prediction (ratio ~1.0).
  The 1e9-BPM blowup is the UNBOUNDED log-RW prior behaving EXACTLY as defined (Var grows ~t*sig^2);
  not an optimization bug -> argues for an OU/mean-reverting tempo prior.
  CORRECTION to my earlier prediction: overshoot pushed prior tempo sigma UP (0.244->6.69), NOT
  down. Reason: KL(stop_grad q || multi-step prior) with a parameter-free RW mean can only "cover"
  the d-step-ahead posterior by INFLATING sigma. So overshoot makes the stochastic tempo prior MORE
  diffuse. The ablation's bpm_std "improvement" (147->46) was from the deterministic MEAN chain,
  which is blind to sigma -- exactly the frozen-mean-metronome caveat biting.

## [~05:5x] Task 2 — OVERFIT ONE SONG (interim, lr 1e-3) — EXPRESSIVITY CONFIRMED
  s1:   recon 167.9  TF-beatF dec 0.000  decmax 0.48
  s100: recon 12.4   TF-beatF dec 0.000  decmax 0.44
  s200: recon 0.4    TF-beatF dec 1.000  TF-beatF lat 0.286  decmax 0.98
  => The architecture CAN express the answer: teacher-forced decoder beat-F -> 1.000 and recon->0
  on a single song. So full-data failure (tf_post_dec=0.000, Task 4) is NOT an expressivity limit;
  it is optimization/data-scale (decoder collapses to majority class across diverse data).
  NOTE latent read-out (0.286) lags decoder (1.000): decoder memorizes beats via h, the PHASE
  latent is not what carries it.

## [~05:5x] Task 3 — SYNTHETIC-TRUTH (interim, s200) — TRAIN/DEPLOY GAP IS FUNDAMENTAL
  s200: recon 0.2  TF-beatF=0.841  freerun-beatF=0.000  phase-circcorr=0.172
  On CLEAN click-track audio with KNOWN tempo/meter: teacher-forced recovers beats (0.84) BUT
  free-run = 0.000 and recovered phase does NOT match planted phase (circcorr 0.17). Even with zero
  audio ambiguity, the generative/prior rollout fails and the latent does not learn the true phase.
  The train->free-run gap is real and fundamental, not a real-data artifact.

## [~06:1x] CORRECTION/nuance on Task 2 & 3 finals
- Overfit lr=1e-3 FINAL: TF-dec-F=1.000, TF-LATENT-F=0.000 (n_ref=6). Decoder fits PERFECTLY;
  the phase LATENT carries nothing (0.000) even when overfitting one song. (latent was 0.286 at
  s200 then decayed to 0 as the decoder took over via h.) Expressivity = decoder-only.
- Synth: the TF-beatF there is the LATENT subdivision read-out (not decoder). It is UNSTABLE
  (0.000->0.841@s200->0.050@s400) while phase circular-corr to the PLANTED phase stays low
  (0.17->0.23). So the transient 0.84 was a subdivision-wrap coincidence, NOT true phase recovery.
  Stable synth signals: free-run=0.000 throughout; planted phase NOT recovered (circcorr ~0.2).
  Conclusion holds and is stronger: the latent never learns the true generative phase, even on
  clean known-truth data; only the decoder (teacher-forced, reading h) can fit.

## [~06:3x] FINAL overfit + synth
Overfit lr=1e-3: TF-dec-F=1.000, TF-lat-F=0.000.
Overfit lr=1e-2: dec-F hit 1.000 @ s100-300 (decmax 0.55->0.64) then DIVERGED @ s400 (klt
  exploded 13->374->1210, recon back up, dec-F->0.000). lr=1e-2 unstable via unbounded tempo-KL.
  -> expressivity confirmed at BOTH lrs (decoder reaches F=1.0); but the PHASE LATENT is 0.000 in
     EVERY eval at both lrs even when overfitting ONE song. The decoder does it ALL via audio h.
Synth FINAL: free-run=0.000 at every step; phase circ-corr to planted stays ~0.17-0.24 (even
  -0.18 once) -> latent NEVER recovers the planted phase. recon->0.0 (decoder fits clicks via h).
  TF-lat-F bounces (0/.84/.05/0/.24) = noise, not recovery.

SHARPENED HEADLINE: it is not merely "decoder collapses on full data". The deeper fact is the
STRUCTURED LATENT (phase/tempo) is never used — the decoder reconstructs from audio h alone. On
one song the h-decoder memorizes (F=1); on full data it fails to generalize (Task4 tf_post_dec=0).
The bar-pointer latent earns nothing. Bonus: unbounded RW tempo is also an OPTIMIZATION hazard
(lr=1e-2 divergence via tempo-KL explosion), not just a deployment one.

## [~06:5x] NEXT-FIX GRID launched in tmux session `nextfix` (scratchpad/nextfix.py)
Tests the two diagnosed fixes factorially. (A) widen beat target +/-W frames (shift-tolerant);
(B) OU/mean-reverting tempo prior log tau ~ N((1-theta)lt_prev + theta*C, sig), C=-3.3 (~120bpm).
PRIMARY metric tf_post_dec (TF posterior decoder beat-F). Cells (GPU: wave1 -> wave2):
  g0: base(w0,ou0)      -> widen5(w5,ou0)
  g1: widen3(w3,ou0)    -> ou10(w0,ou.1)
  g2: ou05(w0,ou.05)    -> both_w5(w5,ou.05)
  g3: both(w3,ou.05)    -> both_strong(w5,ou.1)
Results -> runs/nf/<cell>/result.json (tf_post_dec, tf_post_lat, fr_lat). ~40min/wave.

## [~09:1x] NEXT-FIX grid VERDICT + fr_dec deployment check
tf_post_dec (baseline 0.000): base 0.000, ou05 0.000, ou10 0.000 | widen3 0.482, widen5 0.607,
  both 0.504, both_w5 0.614, both_strong 0.622. -> WIDENING THE TARGET (fix A) is the lever;
  OU alone does nothing for the decoder (expected: OU is a deployment fix).
Eval tolerance = mir_eval default +/-70ms = 6.03 frames (1 frame=11.61ms). widen3=+/-35ms,
  widen5=+/-58ms (both within tolerance); added widen6=+/-70ms (= eval tolerance, principled).
DEPLOYMENT (task4_fourway on widen5): tf_post_dec=0.619, fr_dec=0.617, tf_post_lat=0.000, fr_lat=0.335.
  KEY: fr_dec(0.617) == tf_post_dec(0.619) -> the DECODER ignores the latent and reads beats from
  audio h; teacher-forcing vs free-running the latent makes no difference. tf_post_lat=0.000 -> the
  bar-pointer latent is INERT. Widening produced a working DISCRIMINATIVE FRONTEND detector (~0.62,
  weak vs BeatThis ~0.88), NOT a working generative VAE. Reconfirms joint-eval verdict mechanistically.
NEW BASELINE: widen target to +/-70ms (W=6) is the necessary TRAINING recipe (nothing trains
  without it). It is NOT a solution to the thesis: latent still earns nothing (fr_dec=tf_post_dec).
  Open problem B unchanged: force the latent into the deployment path.

## [~09:5x] TIER-1 latent_only on new (widen) baseline — PROBLEM NARROWED TO MULTI-STEP ROLLOUT
All nextfix tf_post_dec (baseline 0): widen3 .482 widen5 .607 widen6 .587 both .504 both_w5 .614
  both_w6 .590 both_strong .622 | ou05/ou10 .000 | w6_latent .665 w6_latent_ou .669.
KEY latent_only effect (removes audio shortcut, forces decoder through z):
  tf_post_lat: widen5 0.000 / widen6 0.068 -> w6_latent 0.399 / w6_latent_ou 0.407  (latent LOAD-BEARING)
Full four-way (latent_only):
  w6_latent    : tf_post_dec .654  tf_post_lat .399  tf_prior_lat .476  fr_dec .276  fr_lat .018
  w6_latent_ou : tf_post_dec .668  tf_post_lat .407  tf_prior_lat .380  fr_dec .331  fr_lat .330
READING: tf_prior_lat (one-step prior from posterior state) = 0.48 GOOD; fr_lat (multi-step free
  rollout) = 0.018 BAD. One-step prior works, multi-step rollout compounds errors -> the binding
  problem is now PRECISELY multi-step prior consistency (Hafner/PlaNet gap). Latent overshooting now
  has a WORKING latent to roll (previously inert). fr_dec(0.28) << tf_post_dec(0.65): removing the h
  shortcut makes the deployment gap VISIBLE (no longer masked by fr_dec=tf_post_dec). OU keeps free-run
  at periodicity floor (.33) vs collapse (.018) but doesn't track.
PROGRESS: (A decoder collapse) FIXED by widen; (latent inert) FIXED by latent_only; (B multi-step
  deployment) REMAINS = the clean, isolated, literature-addressed next target.

## [~10:40] TIER A COMPLETE — deployment gap proven fundamental; P/R splits added
All on widen6+latent_only baseline (w6_latent: tf_post_dec .654 tf_post_lat .399 fr_lat .018):
  w6lat_os4   .000/.000/.332   w6lat_os4fn .000/.000/.317   w6lat_os8 .000/.000/.331
  w6lat_os4ou .000/.000/.000   w6lat_ou20  .000/.131/.372   w6lat_ou10 .556/.329/.344
  w6lat_oulong (OU0.05, 1200 steps) = tf_post_dec .993  tf_post_lat .607  fr_lat .356  <-- KEY
DECISIVE: with 3x training the teacher-forced decoder->0.99 and latent->0.61 (BEST latent ever),
  yet fr_lat stays .356 (periodicity floor). The train->free-run gap is NOT training-amount, NOT
  decoder, NOT latent-capacity. Tier A (faithful audio-blind levers: overshoot, OU, longer) CANNOT
  close deployment. Overshoot @ w=1.0 always collapses (starves recon; free-nats doesn't save it);
  strong OU (0.20) collapses; mild OU (0.05/0.10) safe but no deployment lift. Confound: oulong's tf
  gains are mostly the extra steps (ou10@400 ~= baseline), but the fr_lat conclusion is robust.
P/R (+/-70ms) failure modes: collapsed cells fire NOTHING (tf_dec R=0); free-run latent either
  DESYNCS (w6_latent fr_lat R=.06) or OVER-FIRES at periodicity (R~1.0 P~.2). Never tracking.
  w6_latent: tf_dec P.59/R.81/F.665, tf_lat P.32/R.56/F.395, fr_dec P.26/R.46/F.327, fr_lat P.18/R.06/F.019.
IMPLICATION: Tier B (audio-conditioned prior MEAN, VRNN-style; small head g_phi/g_tau on frontend
  features -> Delta_phi/Delta_logtau, distilled from posterior via KL) is now evidence-justified as the
  ONLY remaining lever for deployment-time feedback. Crosses the paper's audio-blind-prior faithfulness line.
TODO next session: re-eval needs the gentler-but-longer PLAIN baseline (no OU, 1200 steps) to fully
  decouple steps from OU; then implement Tier B.

## [~11:xx] EXP 0.1 ORACLE INJECTION — DECISIVE: deployment failure is TEMPO drift, not phase drift
w6_latent (beat F1): Kinf 0.019 | Kevery(ceiling) 0.936 | K4s armA(phase) 0.031 | K4s armB(phase+tempo) 0.674 |
  K2s armA 0.079 | K2s armB 0.833.  w6lat_oulong: Kinf .356 armA .358 (no help) armB(4s) .559 armB(2s) .714.
=> Phase-only resync does NOTHING (0.019->0.031); phase+tempo resync recovers hugely (->0.67@4s, ->0.83@2s).
   The binding deployment failure is TEMPO DRIFT (wrong frozen tempo re-desyncs phase in <1s). Matches task6
   (tempo Var ~ t*sigma^2). IMPLICATION: Tier B must condition the prior TEMPO mean (g_tau) on audio, NOT just
   phase. A phase-only innovation prior would fail. Ceiling check (Kevery=0.936) reconfirms read-out sound.

## EXP 1.3 RENEWAL/SURVIVAL launched (inhomogeneous-Poisson beat-rate NLL ties tempo x meter to event rate).

## [~11:1x] Tier-A collection (cron) — four-way on best cell + verdict written to FINDINGS
w6lat_oulong four-way: tf_post_dec .994  tf_post_lat .615  tf_prior_lat .626  fr_dec .359  fr_lat .354.
One-step prior 0.63 vs multi-step free-run 0.35 = the deployment gap, UNTOUCHED by all Tier-A levers.
VERDICT: faithful audio-blind prior fixes (overshoot/OU/longer) cannot close multi-step deployment.
0.1 shows the cause is TEMPO drift -> Tier B must condition the prior TEMPO mean on audio. AUTONOMOUS_FINDINGS.md updated.

## [~12:05] EXP 1.3 (fixed, h-reading) + TIER B v1 (delta form) — results
1.3 survival, h-reading (robust decoder did NOT collapse): tf_tempo_corr CONTROL(widen6, no surv)=-0.034
  -> survh02(w.2)=0.499, survh05(w.5)=0.458. fr_lat ~0.39-0.41 (floor). dec 0.55-0.60.
  VERDICT: 1.3 WORKS for tempo identifiability (corr 0->0.5; the per-frame decoder carries ZERO tempo
  info, control=-0.03). Cleanest "latent earns what frontend can't". Scope: tempo correlation only, NOT
  absolute (tAcc1=0, m_eff offset) and NOT deployment (fr_lat=floor).
TIER B v1 (audio-conditioned tempo mean, DELTA form mu_tau=lp+g_tau(h)), latent_only 1200 steps:
  dec 1.000, lat 0.177, fr_lat 0.200 (WORSE than 0.33 floor), tf_tempo_corr 0.784 (highest yet).
  The audio head READS tempo well (corr 0.78) but DEPLOYMENT got worse: the delta ACCUMULATES at free-run
  (lt=sum g_tau) -> audio-driven drift, no restoring force. Wrong functional form.
  FIX: audio-conditioned OU (restoring force toward an audio target): mu_tau = lp + a*(g_target(h) - lp).
  Unifies OU (restoring, but constant target) + Tier-B-delta (audio, no restoring). = 0.1 arm-B limit (a=1,GT).
NEXT: implement Tier B v2 (audio-OU anchor) and test fr_lat.

## [~12:45] TIER B v2 (audio-OU anchor) sweep — MODEST lift, gap NOT closed
mu_tau = lp + a*(g_target(pc) - lp). latent_only, 800 steps. floor~0.33, one-step ceiling~0.5-0.63.
  a=0.1: dec .998 lat .337 fr_lat .406 tcorr .041
  a=0.3: dec .962 lat .642 fr_lat .401 tcorr .103   <- best latent (0.64) AND fr_lat
  a=0.5: dec .788 lat .305 fr_lat .338 tcorr .183
VERDICT: audio-OU anchor lifts deployment 0.33->0.41 (+0.07) but does NOT reach the one-step ceiling.
Two diagnosed causes: (1) audio tempo TARGET is weak on LogMel (tcorr 0.04-0.18) -> needs Beat-This
features for an accurate target; (2) only tempo corrected, but 0.1 arm-B showed phase ALSO needs
anchoring. => Tier B v3 = phase+tempo anchoring on Beat-This [T,512] features is the evidence-indicated
next config. Cumulative: NOTHING this session closed the deployment gap to ceiling; best fr_lat ~0.41.
Clean scientific WIN remains 1.3 (tempo identifiability, corr 0->0.5, frontend can't).

## [~13:0x] BLIND PANEL (7 indep agents) + GT-constant-tempo diagnostic — REFINED VERDICT
Blind panel (wnxnaruy2): CONVERGED independently on "tempo is the broken deploy state; no FAITHFUL fix
for competitive tracking; minimal departure = audio-conditioned tempo MEAN (mu_tau += g(h))". Corrected:
(a) frozen-tempo read-out elbo.py:144 is cosmetic not the limiter (stochastic chain diverges too);
(b) t=1 tempo anchor IS learnable via KL to GT-informed posterior ("no gradient to tempo" too strong).
NEW diagnostic (tempo_const_test.py on oulong) — best possible CONSTANT-tempo metronome (perfect global
tempo + best phase, audio-blind):
  model open-loop 0.356 | GT-const 0.510 overall | 0.645 near-constant-tempo (11/16) | 0.214 varying (5/16)
=> FAITHFUL CEILING ~= 0.51 (deployed prior IS a constant-tempo metronome). Model (0.356) underperforms
   its OWN faithful ceiling by mis-estimating ONE learnable global-tempo scalar -> faithful headroom
   0.36->~0.51 exists. Above ~0.51 = tempo VARIATION -> requires audio-conditioned tempo mean (unfaithful).
PRECISE ANSWER to "why can't faithful do better": constant-tempo metronome by construction, hard-capped
~0.51; currently below cap due to a botched learnable tempo scalar; >0.51 needs unfaithful audio->tempo.

## [~14:16] TIER B + 1.3 COMBINED (never-done) — does NOT break through
latent_only, end-to-end, 800 steps. floor~0.33.
  combo_a3s05 (a.3,s.05): dec .821 lat .362 fr_lat .382 tcorr .082
  combo_a3s1  (a.3,s.1) : dec .999 lat .651 fr_lat .388 tcorr .059
  combo_a3s2  (a.3,s.2) : dec .987 lat .643 fr_lat .363 tcorr .093
VERDICT: fr_lat stuck at floor (0.36-0.39); tcorr stuck 0.06-0.09 (<< survival-alone 0.50). Two causes:
(1) the Tier-B anchor's KL drags posterior tempo toward the weak prior target, UNDOING survival's tempo
supervision (mechanisms conflict); (2) more fundamentally, our small from-scratch GRU encoder cannot
extract usable tempo from h in the deploy config (tcorr ~0.08). Matches Beat-Transformer (arXiv 2209.07140,
Zhao 2022 = demixed dilated-self-attention beat/downbeat TRACKER, NOT a tempo estimator): reading tempo
from audio needs a LONG receptive field our GRU lacks. The audio HAS tempo; our encoder can't pull it out.
REMAINING LEVER (end-to-end): upgrade encoder to dilated-conv/attention (Beat-Transformer-style, FROM
SCRATCH). Big build, uncertain payoff, re-raises "is it the transformer doing the work". Science (diagnosis)
is complete: faithful capped ~0.51; model sits 0.36 (botched tempo scalar); >0.51 needs audio->tempo;
our encoder can't extract it; bigger encoder is the only untested path.

## [~16:30] FREE-RUN RECON + SCHEDULED SAMPLING + h-DROPOUT batch — floor-bounce + collapse
floor .33 | prev best .41 | faithful ceiling ~.51:
  frec_faith .242 dec0 | frec_faith2 .000 dec0 | frec_tierb .378 dec0 | frec_tierb2 .264 dec0  (ALL free-run COLLAPSED)
  hdrop05 fr_lat .291 lat .322 dec .591 tcorr .509  (HEALTHY, latent load-bearing + tempo-corr up, but deploy=floor)
  ss05 .019 dec.56 | ss08 .018 dec0 | ss_surv fr_lat.403 dec0 tcorr -.588 (collapsed/artifact)
VERDICT: nothing cleared 0.41 with a healthy decoder. Free-run reconstruction (backprop thru 128-step
stochastic rollout) COLLAPSES (starves recon) — confirmed risk. Scheduled sampling on latent_only ALSO
collapses/degrades (my "SS is stabler" was WRONG). NEW finding: the latent_only decoder is
OPTIMIZATION-FRAGILE — any aggressive auxiliary rollout term collapses it; only h-reading (hdrop05) survived.
hdrop05 = word-dropout (Bowman) WORKS for latent-use (lat .07->.32) + tempo-corr (.51) but deployment still floor.
CONCLUSION: every training/objective-side attack on the deploy gap (overshoot/OU/survival/TierB/free-run/
sched-samp/dropout) either collapses the model or leaves free-run at floor. Deployment wall does NOT yield to
loss tricks at this scale. Remaining lever = CAPACITY (tempo-capable dilated-attention encoder), a real build.

## [~17:xx] WARM-START clean test (from oulong, no collapse confound) — DECISIVE + CONSTRUCTIVE
Fine-tune healthy oulong (lr 3e-4, 300 steps) with method, + no-method control. floor .33, ceiling ~.51:
  ws_control  fr_lat .355  tf_post_lat .263  dec .956  tcorr .886
  ws_ss03     fr_lat .369  tf_post_lat .527  dec .994  tcorr .824
  ws_ss05     fr_lat .370  tf_post_lat .557  dec .991  tcorr .711
  ws_frec05   fr_lat .332  tf_post_lat .610  dec .993  tcorr .928
VERDICT: fr_lat FLAT (0.33-0.37 ~ control) -> scheduled sampling / free-run do NOT lift deployment even
from a healthy model (the earlier collapses WERE a from-scratch confound, now removed; still no lift).
CONSTRUCTIVE KEY: the methods drove tf_post_lat -> 0.61 and tcorr -> 0.93 -> the POSTERIOR TEMPO IS
ACCURATE (capacity is NOT the blocker given enough training). Deployment fails ONLY because the prior
mean is audio-blind/parameter-free -> the accurate posterior tempo has NO parameter to carry it into
free-run. => SS/free-run improve the posterior side but it can't reach deployment without an
audio-conditioned prior mean (Tier B). NEXT (evidence-backed): warm-start this healthy model (tcorr .93)
+ Tier B g_tau trained to DISTILL the accurate posterior tempo (STOP-GRAD on posterior) into the prior
mean -> the one config that connects the accurate tempo to the deployment path. Not yet run.

## [~18:40] ROUTE 2: FIVO (filtering variational objective, a REAL VAE) — floor, clean negative
K=1 FIVO == ELBO verified; bound tightens with K (153->142->130 for K=4/8/16). Training:
  fivo_k4 fr_lat .383 lat .000 dec0 | fivo_k8 .369 lat .039 dec0 | fivo_k16 .380 lat .057 dec0 |
  fivo_ws_k8 (warm-start) fr_lat .371 lat .000 dec .537
VERDICT: FIVO does NOT lift free-run deployment (all ~0.37 = floor). From-scratch FIVO collapses the
decoder (dec0); warm-start keeps decoder but latent unused. The filtering objective improves TRAINING-time
inference (tighter bound) but deployment is OPEN-LOOP free-run with NO observation to filter against, so it
doesn't transfer. DEEP REASON (now very robust): at deployment the only signal is audio; the faithful
generative model's latent dynamics are audio-blind, so NO inference method (FIVO, He-2019, SMC) can help the
open-loop rollout. To exceed ~0.4 you must EITHER leave the VAE (discriminative frontend peak-pick = 0.64,
not a VAE) OR leave faithfulness (audio-conditioned prior mean Tier B = ~0.40 and corrupts tempo / departs
from the DBN). Within the faithful VAE/ELBO_for_DBN paradigm, free-run deployment is structurally capped ~0.4.
Also non-VAE route-2 MVP (pf_deploy.py: BCE activation + particle filter) = 0.351 (my crude PF) vs raw
peak-pick 0.642 -> a plain frontend beats our crude DBN; that route is the [NN+DBN] pipeline CHART replaces.

## [~19:xx] SIMPLE TEMPO ESTIMATOR (autocorr, no learning) — validates deep-dive + sharpens it
Classic onset-envelope autocorrelation tempo estimator (tempo_estimator.py):
  estimator Acc2 (octave-tolerant)=0.75, Acc1(+-4%)=0.375  vs VAE prior_init tempo 0% octave / 852% err.
  (2) metronome @ estimated tempo, BEST-of-8 phase = F1 0.663
  (3) oulong VAE free-run, tempo FROZEN @ estimate = F1 0.401  (model 0.356, GT-frozen 0.510)
CONFIRMS deep-dive: a TRIVIAL autocorr estimator gets tempo 75% octave-right; the VAE prior gets 0% ->
the breakdown really is prior audio->tempo (mean-pooled periodicity). SHARPENS it: tempo alone in free-run
=0.40 (phase becomes the limiter); metronome+phase-search=0.66. So the missing ingredient is INFERENCE
(search over BOTH tempo AND phase, audio eliminates wrong ones = the DBN), not just an estimator. Free-run
searches neither -> commits to one feed-forward guess each. Pure classic [tempo est + phase search]=0.66
(no VAE). The VAE's value must come from elsewhere if a 10-line classic method already gets 0.66.

## [~20:xx] ROUTE 1 WORKS: corrected SMC (AESMC-referenced) + MAP readout BEATS free-run
Deep analysis refuted my 3 PF hypotheses (readout/degeneracy/tempo-found) and revealed ESS=0.999 ->
the filter was NEVER reweighting (per-frame-softmax bug, no accumulation). Cloned official AESMC
(third_party/aesmc, tuananhle7) -> correct recursion: per-step weight=emission logp (bootstrap),
ACCUMULATE across frames, adaptive resample on ESS<K/2, RESET after. Corrected SMC (pf_analyze2.py):
  ESS/K 0.999->0.780 (12 resamples/song) -> filter now actually filters
  F1 MAP-trajectory=0.475 >> circular-mean=0.319 -> H1 (readout matters) CONFIRMED once filter works
  DEPLOY-by-inference (MAP)=0.475 BEATS free-run 0.40 (first real-VAE deployment off the ~0.40 wall!)
  toward classic 0.66 (not there: weak emission vs trained activation, 800 steps, K=800, m fixed = knobs)
Weighted-mean BPM still 91% off = multimodal-average artifact; the MAP PARTICLE carries the good traj.
LESSON: "feeling good" was premature (filter silently broken, ESS=0.999); deep analysis + official code
+ verify-not-declare is what worked. Route #1 (trainable bar-pointer SSM-VAE deployed by SMC) VALIDATED.
