# Overnight packet — 2026-07-12 (autonomous run)

Deployment numbers are FILTER/recog read-outs, not training-log read-outs. Baselines quoted.

## 1. Wave-2 diverse-corpus collapse — fully characterized
- **Data acquitted**: vanilla-BT per-song label audit unimodal on every wave-2 dataset
  (ASAP 0.61, Groove 0.89, GuitarSet 0.88, RWC 0.67, TapCorrect 0.65). No 2nd Harmonix.
- **Deployment-fatal AND seed-reproducible**: w2_s0 filter 0.398/0.539, w2_s1 filter 0.397/0.487
  on own-evidence 0.830 (327 val). Both seeds → same ~0.40 floor. Healthy wave-1 = 0.91.
- **Root cause (probe)**: reconstruction pays only ~9 nats for phase on the HEALTHY model, ~0 on
  collapsed. Free bits is the sole thing holding phase up, and its clamp is a one-way door
  (zero gradient below floor). ELBO barely values what the filter needs → objective misalignment.
- **Tutorial anti-collapse arms, all null/negative** (same 0.785 MERT evidence):
  early-stop 0.829 | anchor(0.1) 0.819 | anneal -> 0.778. Prior-side medicine can't reach a
  reconstruction-side disease. Professor-remedy scorecard now: EB neutral, hybrid marginal,
  anchor neutral, anneal negative.

## 2. MERT lane — UNBLOCKED via early-stop
- early800 x GRU head: filter 0.829 (evidence-pp 0.785) — +0.044 over peak-pick on same evidence.
  MLP head: 0.758 (evidence 0.676) — filter edge grows as evidence weakens (+0.082).
- v3 full 2000-step (survived danger window): 0.810 — BELOW its own 800-step ckpt. Longer training
  under the current objective makes deployment WORSE. Early-stop is optimal, not a collapse-dodge.
- Head A/B: biGRU (0.785) > transformer (0.525) > temporal-conv (0.729) > MLP (0.676) at our scale.
- MIREX-compliant lane (never touches SMC/GTZAN).

## 3. Three-metric table (F/CMLt/AMLt, beat+downbeat) — flagship foldhonest_s0, 327-val, act2 evidence
| arm | beat F | bCMLt | bAMLt | db F | dCMLt | dAMLt |
|---|---|---|---|---|---|---|
| vanilla peak-pick | 0.833 | 0.670 | 0.752 | 0.723 | 0.541 | 0.638 |
| filter default    | 0.828 | 0.664 | 0.738 | 0.726 | 0.546 | 0.646 |
| filter winner-readout | 0.830 | 0.664 | 0.745 | 0.732 | 0.555 | 0.659 |
- **Beats: filter TIES vanilla** on act2 evidence (0.830 vs 0.833) — the +0.02 own-evidence win was
  on our weaker head; on BT's sharpened act2 there is nothing to add.
- **Downbeats + downbeat continuity: filter WINS across all three metrics** (+0.009 F, +0.014 CMLt,
  +0.021 AMLt). The structured model's advantage is at the BAR level, as designed.
- **Latent sweep +0.044 was a 15-song artifact**: +0.002 beat F on 327 (noise). Kept only the mild
  downbeat gain. Read-out threshold retune does NOT beat vanilla on beats.

## 4. BeatFM repro (external yardstick, MERT+MSAM+DBN, separate from VBPM)
- Fold 0: GTZAN beat 0.872 (paper 0.895), Ballroom 0.925, Hainsworth 0.912, SMC 0.505 (paper 0.613).
- Near-paper on all but SMC (single-fold undersells the 8-fold CV aggregate). Folds 1-7 + full
  8-fold eval running -> paper-comparable table by mid-morning.

## 5. Harmonix repair — the FAITHFUL fix (per user steer to the official method)
- Official approach = Harmonix Set DTW mel-spec alignment; BT ships the ALREADY-aligned specs on
  Zenodo (rec 13922116, CC-BY-4.0). My constant-offset probe recovered 51% (365/722); Zenodo specs
  recover ALL 819.
- **DONE (smoke PASSED 0.859 vs poisoned 0.35)**: 45G download + fold-honest re-extraction from the
  Zenodo specs into a FRESH dir (cache/acts/harmonix_zenodo_rich; training cache untouched). Only
  fix needed was key format (<stem>/track) + spec orientation ([T,128], no transpose). 819/819
  songs matched. Full extraction running. Merge into training = morning decision.

## 6. x-only (§7 encoder-only fork) — NEGATIVE, informative
- Full 2000-step run: recon strong (163) but recog read-out 0.000 THROUGHOUT. Latent-unused.
- §7 removes the amortization gap but the encoder puts no usable structure in the phase latent
  (decoder fits events from features directly). The gap was NOT the whole disease -> back to FIVO.

## 7. FIVO — the cause-level fix (docs/fivo_design.md, PROPOSAL, not implemented)
- Root cause is objective misalignment: ELBO pays reconstruction; deployment runs the filter;
  collapse is nearly free to the ELBO. FIVO makes the training objective the filter's own
  marginal-likelihood bound -> optimizes the deployment computation directly; resampling penalizes
  a phase-ignoring particle set. Reuses model/particle_filter.py (add differentiable path).
- Success test: v3-full deployment >= v3-early 0.829 (longer training stops hurting).
- DECISION PENDING: it departs from spec (Alg 1 SGVB) and tutorial (amortized §8); deployment-
  faithful, not tutorial-faithful.

## 8. FIVO — IMPLEMENTED + LAUNCHED (user GO 2026-07-12)
- model/particle_filter.py: `fivo_bound()` — batched differentiable bootstrap-filter marginal
  likelihood (resample-every-step, detached ancestors, reparam phase/tempo, gumbel meter).
- config.py: fivo_weight / fivo_num_particles / fivo_elbo_anneal_steps / fivo_elbo_floor.
- train.py: FIVO aux term with ELBO anneal; data/dataset.py: return_obs crops.
- Smoke: grad reaches ALL prior-side heads strongly (phase_conc 3967, tempo_std 5178,
  correction 1985) — the collapse-relevant nets the ELBO starves. configs/fivo_w2.yaml =
  default wave-2 (collapses both seeds to 0.40) + FIVO. Running GPU 3 -> chained filter eval.
- SUCCESS TEST: fivo_w2_s0 filter deploy > 0.40 (clears the collapse floor).

## 9. v3 seed 1 — survival was SEED-DEPENDENT, not corpus
- v3_s1 phase KL 2.25 at step 900 = collapsing, while v3_s0 held ~200. So the matched-corpus
  "fix" for MERT was luck, not robust. Weakens the corpus-vs-luck story (as cautioned). Still
  finishing for the full trace.

## 10. BeatFM — all 8 folds trained; full 8-fold eval running (paper-comparable table pending)

## Decisions waiting for user
1. FIVO: build it? (root-cause fix for collapse + objective misalignment)
2. Merge repaired Harmonix into training corpus?
3. Freeze commit of the night's code (activation_head, x_only_posterior, three-metric harness,
   configs) — pending manual review.

## 11. Grid-Viterbi exact decode — BUILT + working (2026-07-12, user GO)
- model/grid_decode.py: offline exact Viterbi over the discretized (phase,tempo) bar-pointer state,
  using OUR model's LEARNED transition/emission -- the apples-to-apples counterpart to madmom's DBN.
  Fixes the online/offline mismatch (particle filter is causal; our setting + encoders are offline).
- 8-song smoke on foldhonest_s0: beat F 0.826 (0.30->0.51->0.826 as tempo-convention + deployment
  sharpening fixed). Most songs near-perfect (1.0/0.99/0.98); one octave outlier (0.10) drags mean.
- v1 approximations: fixed 4/4 emission, time-invariant transition scales, no g-correction.
  Next: octave-error handling, full-val universality matrix (peak/filter/grid/DBN x 3 metrics).

## 12. Beat Transformer (vendored /tmp -> external) — DBN-native frontend for the thesis
- 8 pretrained fold ckpts ship in repo (no training needed). Spleeter-demixed input.
- TRAINS ON SMC+GTZAN (8-fold) -> DBN-thesis vehicle, NOT MIREX-clean.
- FOLD CAVEAT (user): Beat Transformer's 8-fold split != Beat This's. When extracting its
  activations, honor ITS fold assignment (fold-k ckpt only on its held-out fold k) or contamination
  returns. Map its audio_lists/ ordering before any extraction.

## 13. Universality matrix preview (20 songs, foldhonest_s0, act2 evidence, CORRECTED DBN config)
CAUTION: DBN config bug caught by verification -- DBNDownBeat[3,4] gave 0.161/0.804 on beats;
correct DBNBeatTracker gives 0.941. Third artifact caught before it became a claim.
| decoder      | beatF | bCMLt | bAMLt |  dbF | dCMLt | dAMLt |
| peakpick     | 0.892 | 0.796 | 0.819 | 0.752| 0.563 | 0.623 |
| filter       | 0.894 | 0.787 | 0.804 | 0.757| 0.586 | 0.638 |
| grid_viterbi | 0.866 | 0.772 | 0.823 | 0.763| 0.676 | 0.735 |
| madmom_dbn   | 0.941 | 0.887 | 0.936 | 0.682| 0.511 | 0.727 |
- BEATS: DBN wins (0.941 on easy subset; ~0.843 full-val, +0.013 over peak-pick). grid-Viterbi
  BEHIND peak-pick on beats (octave errors).
- DOWNBEATS: grid-Viterbi BEATS the DBN clearly (0.763 vs 0.682; dCMLt 0.676 vs 0.511) -- explicit
  bar-pointer+meter wins bar-level structure, as designed.
- "Universal improvement over DBN" HALF-earned: win downbeats, lose beats. Beat gap = octave errors.
- Levers to close beat gap (both things the DBN HAS, we lack): (1) tempo prior on grid-Viterbi
  (penalize extreme tempos -> kills octave errors); (2) meter in the grid state (un-hardcode 4/4
  -> meter+downbeat bar-constraint disambiguates metrical level). Neither costs the downbeat win.

## 14. Tempo prior FIXES grid-Viterbi octave errors (user's hypothesis confirmed)
- grid-Viterbi beat F (20 songs): strength 0.0 -> 0.861 (3/20 octave-fails); 0.5 -> 0.899 (1/20);
  1.5 -> 0.893; 3.0 -> 0.889. Optimal = 0.5.
- +0.038 beat F, octave-fails 3->1. grid-Viterbi now TIES peak-pick (0.892)/filter (0.894) on beats
  AND keeps its downbeat-continuity win. The DBN's tempo prior was the missing piece.

## 15. Meter census (corrects the "~90% 4/4" assumption)
- VAL 327: non-4/4 = 72 (22%) [3:42, 2:22, 8:6]. TRAIN 2287: non-4/4 = 487 (21%) [3:289, 2:181].
- Reality ~79% 4/4, not 90%. downbeat-by-meter test IS well-powered (72 non-4/4 val songs).
  Rebalancing has real data (487 non-4/4 train + Lakh meter-change). Caveat: inferred "2" likely
  partly mis-inferred fast-4/4.

## 16. FIVO alive THROUGH the collapse window (preliminary positive)
- phase KL held 156/137/148/143/89 across steps 1050-1300 (both ELBO-only seeds crashed to ~2 here).
  Dipped to ~33 @ step 800 then RECOVERED -> filter objective actively resisting collapse (free-bits
  can't; one-way door). Deploy eval (does filter clear 0.40) = the real verdict, ~15:15.

## 17. Downbeat-by-meter — REFUTES the "beat DBN on non-4/4" thesis (honest walk-back)
| stratum | n | control | meterA | DBN |
| 4/4     | 255 | 0.751 | 0.750 | 0.699 |
| non-4/4 |  72 | 0.627 | 0.627 | 0.689 |
- meter co-training (meterA) does NOTHING for downbeats (identical to control). meterA is not the fix.
- We WIN on 4/4 (0.751 vs DBN 0.699) but LOSE on non-4/4 (0.627 vs 0.689) -- OPPOSITE of the thesis.
- Mechanism I missed: DBN supports [3,4]; 42/72 non-4/4 are 3/4 = IN its set, NOT its blind spot.
  Its explicit [3,4] search beats our weak meter latent (KL~0.2) on 3/4.
- The DBN's TRUE blind spot is meters OUTSIDE [3,4] (5,7) + meter CHANGES -- our val barely has these,
  so this test never probed it. Original architectural intuition still valid, just mis-targeted here.
- Corrected: (a) meter latent is too weak even for 3/4 -> rebalancing MORE needed but meterA insufficient;
  (b) real test needs 5/7/meter-change songs (Lakh), not 3/4.

## 18. Particle-FIVO FAILS to fix collapse (both doses) -> pivot to exact grid forward-algorithm
- FIVO-strong (w0.5, 16 particles) deployment trajectory: step300=0.444, step600=0.440 -- FLAT at the
  ~0.40 collapse floor (a working FIVO would climb). Weak w0.1 run delayed collapse but spoiled endgame.
- Verdict (step900 pending confirm): particle-FIVO does NOT lift deployment above the collapse cap.
- Likely cause = particle-estimator gradient variance (Rainforth 2018 "tighter bounds not necessarily
  better") -- too noisy to train good dynamics.
- SUCCESSOR: exact grid forward-algorithm training (log-sum-exp over the discretized state grid) --
  the VARIANCE-FREE version of FIVO. Same formulation gives parallel-in-time training (assoc. scan;
  Sarkka 2021) and connects to CRF beat trackers (Fuentes [22]). model/grid_decode.py already has
  the grid machinery; needs the differentiable forward pass + training loop.

## 19. GRID-FORWARD (exact forward-algorithm) SOLVES THE COLLAPSE -- the day's headline
- Deployment trajectory on the collapsing wave-2 corpus (150 val, w2clean evidence, 800p filter):
  step300=0.831, step600=0.826, step900=0.841, step1200=0.841 -- FLAT ~0.83, NO collapse.
- Same protocol, same corpus: ELBO-only=0.398 | particle-FIVO=~0.44 (flat) | grid-forward=~0.83.
- CONFIRMS the variance hypothesis: exact (variance-free) vs particle-estimated gradient was the
  difference between failure and success. Cause-level fix works; collapse PREVENTED (not delayed --
  step 1200 is well past where ELBO+FIVO both died).
- Identity: this is exact-ML training of the discrete bar-pointer model = neural-madmom/CRF, NOT a
  VAE (encoder now vestigial). VAE tested to exhaustion first (ELBO/free-bits/§7/anchor/anneal/FIVO
  all fail) -> exact discrete inference over the SAME generative model succeeds.
- TODO: full 327-val confirm + 2nd seed; grid-Viterbi decode (vs filter); supervised-path variant;
  parallel scan for speed (currently ~12s/step sequential).

## 20. Grid-forward CONFIRMED + ablated (the campaign result)
- Full 327-val, filter: beat 0.844 db 0.691 (vs same-evidence DBN 0.843 / peak-pick 0.830 on BEATS
  -> grid-forward MATCHES DBN on beats, beats peak-pick; downbeat 0.691 is the weak channel, no win).
- SEED 1 step900 = 0.845 (== seed0 0.844) -> ROBUST, not seed luck.
- NO-Q (elbo_floor 0.0: drop encoder q + fade out beat labels) step900 = 0.835 -> needs NEITHER q
  NOR beat labels; near-self-supervised. Identity confirmed: NOT a VAE (learned bar-pointer HMM by
  exact ML). Caveat: ELBO annealed over first 400 steps so emission had fading label grounding;
  fully-label-free (anneal off) = the airtight version, untested.
- Ladder (same corpus, same protocol): ELBO 0.398 (collapse) | particle-FIVO 0.44 | grid-forward 0.84.
- OPEN: downbeats (0.691, need same-evidence DBN cmp + meter-in-grid); grid-Viterbi full-val
  (backtrack arrays ~1GB/song on 8000-frame songs -> needs frame-cap/streaming rewrite).

## 21. DOWNBEAT BOTTLENECK BATTERY (9-agent workflow, 327 songs, all hypotheses measured)
RANKED (by measured ceiling): 1) SPURIOUS 2x downbeat doubling — 117/327 songs ALL over-predict,
66 at ~2x (half-bar peaks); 47% of lost F; fix = half-bar hypothesis test in decode (FREE, +0.03-0.06).
2) Own head downbeat evidence weak (0.611 vs act2 0.736 on 4/4; +0.02-0.04, training).
3) Bar-constrained decode +0.018 (DBN-on-head 0.703; all 4/4, NEGATIVE non-4/4).
REFUTED: rotation/gauge (+0.016 oracle, 20 songs); read-out tuning (default = grid optimum;
only sigma=2 smoothing +0.008 real); evidence weight (w=3 optimal, more HURTS p=0.007);
meter-oracle as downbeat lever (+0.004 — phase tracked under always-4 bar, meter knowledge
can't rescue post-hoc).
DISCOVERIES: (a) meter latent FULLY collapsed at deploy: MAP=4 on 327/327 (not weak — constant).
(b) "we lose 3/4" story WRONG: 3/4 is our BEST db stratum (0.782); real deficits = 2/4
(double-tempo beat breakage, db 0.474) and meter-8 (outside class range, 0.516).
(c) [3,4] Viterbi search: +0.077 true-3 BEATS (adopt for beats), +0.002 downbeats (useless there);
meter choice by path score = 42% on non-4/4 (worse than chance).
Same-evidence DBN downbeat (the missing number): 0.703 vs our 0.685-0.693 — DBN +0.01, wins 4/4,
collapses non-4/4 (0.589). Post-fix ceiling ~0.73-0.75 (would clearly beat DBN).

## 22. Endgame stability (step-2000 finals) — one honest nuance
- seed1 hybrid (15% ELBO): 0.845@900 -> 0.837@2000 — STABLE to the end, no early-stop needed.
- no-q/no-labels: 0.835@900 -> 0.765@2000 — DRIFTS late. The residual ELBO/label share does real
  endgame stabilization (confound: ablation removed encoder AND labels together; likely the labels
  grounding the emission, untested). Claims stand: exact inference REACHES ~0.84 without q/labels
  (step-900); production recipe = hybrid (grid-forward + elbo_floor 0.15), stable, 0.844.

## 23. Half-bar DECODE fix: REFUTED by direct measurement (2026-07-13)
- Alternation-contrast halving: 0.667 (HURTS vs 0.685); on the 2x band itself 0.559->0.537.
- Likelihood-ratio halving: 0.49 (catastrophic) — down_act's half-bar peaks are as high-probability
  as the true ones; the 1x/2x information is NOT in the activation. No decode-time rule can recover it.
- Verified decode-side total: sigma=2 smoothing +0.008 (0.685->0.693). That is ALL.
- Fix must be UPSTREAM (evidence): per-frame MLP head is structurally half-bar-blind (no temporal
  context). Testing biGRU head on BT features (worked on MERT: 0.676->0.785 beats). RUNNING.

## 24. biGRU evidence head -> NEW PROJECT BESTS on both channels (2026-07-13)
- BT biGRU head (temporal context; the upstream fix after decode-fix refutation): evidence
  beat 0.825 / db 0.634 (MLP: 0.830/0.613; act2 db 0.710).
- DEPLOYMENT (grid-forward + filter + GRU evidence, 327-val): beat 0.844->0.864, db 0.691->0.707,
  db+smooth(s2) 0.722. Beats madmom DBN on BOTH channels (0.843 beats / 0.703 db, MLP-evidence refs).
- Filter extracts +0.039 over its own evidence peak-pick (0.825->0.864) — own-evidence thesis,
  strongest form: context-rich evidence + structured dynamics compound.
- Pending: strict same-evidence DBN (DBN on GRU evidence) to seal the claim.

## 25. STRICT same-evidence DBN on GRU evidence — corrects §24's "beat DBN on both channels"
- madmom DBN on GRU evidence: beat 0.865 / db 0.758. Ours: 0.864 / 0.722.
- VERDICT: beats = dead tie (0.864 vs 0.865); downbeats = DBN WINS +0.036. The §24 claim was
  evidence-mismatched (DBN ref was on weaker MLP evidence). 4th same-evidence correction.
- REVISES "decode tapped out": DBN extracts +0.124 over peak-pick on GRU evidence (our filter +0.088)
  via JOINT bar-constrained Viterbi (downbeat forced onto every m-th beat DURING decode) — different
  mechanism from the refuted post-hoc halving. Decode headroom exists via joint bar-constrained
  decoding (meter-in-grid grid-Viterbi) on context-rich evidence.
- Standing: absolute bests 0.864/0.722; learned pipeline = DBN-parity on beats, -0.036 downbeats.

## 26. First-principles verification suite (experiments/first_principles/, 2026-07-13)
Professor-verifiable single-file experiments: NO project code, NO caches, synthetic data, fixed
seeds, one claim per script, every equation commented at its implementing line.
- exp1: gauge posterior is M-modal; unimodal q = confidently-wrong (keeps 1/M mass) OR uninformative.
- exp2: grid forward = EXACT log p(x) (converges to machine precision by 360 bins); best-possible
  unimodal ELBO (zero amortization gap) still ~2.2 nats/frame below = pure approximation gap.
- exp3: headline in miniature — same model/data/decoder, ELBO-trained deploys F=0.103 (omega
  mis-trained 0.15->0.01), exact-trained F=0.844 (omega recovered 0.1501). Mirrors real 0.398->0.844.
- Toy caveat (documented in-file): global scalar tempo needs in-basin init (aliased likelihood);
  real model avoids this because tempo is IN the latent state.

## 27. Verification suite REFACTORED onto vendor code (trust upgrade, 2026-07-13)
- exp1: scipy.stats.vonmises + scipy.special.rel_entr + scipy.signal.find_peaks (zero hand math).
- exp2: EVERY exact log p(x) value now from hmmlearn.GaussianHMM.score (third party) — matches the
  old hand-rolled recursion digit-for-digit (certifies it); ELBO arm via torch.distributions.VonMises,
  8 restarts + unbounded kappa (every advantage to the ELBO): gap 64.9 nats/30 frames stands.
- exp3: arm A uses torch.distributions.Normal.rsample + kl_divergence; deployment decode for BOTH
  arms is librosa.sequence.viterbi. Result unchanged: ELBO F=0.103 vs exact F=0.841.
- Process note: the refactor itself caught a would-be dishonesty (my kappa<=50 clamp handicapped
  the ELBO to -1423; unclamping restored its true best ~-80) — the suite now provably favors the
  variational side wherever a choice existed.
