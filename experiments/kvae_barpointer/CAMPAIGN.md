# KVAE bar-pointer — overnight campaign (2026-06-29)

Autonomous run while you sleep. Goal: resolve as many open questions as possible so you wake up knowing
what's left. This file is updated as results land — check the **STATUS** and **RESULTS** tables.

---
## ★★★ DIAGNOSTICS (verified vs asserted) — added after the "are we sure?" check ★★★
Ran controlled experiments to confirm-or-refute each claimed "lingering problem":
- **Read-out broken? REFUTED.** Clamp test (feed GT φ): beat **0.975** → the geometric read-out path is sound.
- **Numbers are noise? REFUTED.** Seed variance (3 seeds): beat **0.836 ± 0.009** → stable.
- **Geometric continuous-φ can't lock? CONFIRMED.** `phi_capacity` (pure φ-supervision, NO ELBO): loss
  pinned at ~1.0 (uncorrelated) for 1200 steps, beat stuck 0.375. Even when matching GT is the ONLY
  objective, the filtered-latent→scalar-tempo→cumsum φ won't track GT. Across ~4 variants (atan2-z,
  integ-tempo, +GT-sup w20, +pure-sup) → continuous-latent geometric read-out is a DEAD END. The discrete
  tempo-phase DBN is the only working geometric route. (Scope: no continuous param we tried, not a proof.)
- **Free-run collapses? CONFIRMED.** KVAE `predict_future` open-loop: beat **0.289** vs filter **0.830**
  (ratio 0.35). The prior is not a standalone generator; the model needs filtering (audio every frame).
- **VAE capped < frontend in-domain? leaning CONFIRMED.** Scaled M1 (a/z=16,K=8,800 songs) hovers
  0.82–0.87; the ~0.055 gap to frontend 0.929 does NOT close with scale. (act2-augmented config finishing.)

**Net:** the real problems are (1) geometric needs the discrete DBN, (2) free-run collapses → needs
filtering, (3) VAE learned-head trails the frontend in-domain by ~0.05. The "read-out broken" and
"numbers unreliable" worries are refuted. These are now TESTED, not pattern-matched.
---
## ★★ BOTTOM LINE (read this first) ★★
**1. The wall is broken and reproduced.** Exact differentiable filter (Kalman-VAE) + learned head:
   beat **0.878** (M1), scaled run peaked **0.888** / final 0.848 (mild late fluctuation), leak collapses
   (0.85→0.27→0.00). The amortized-collapse blocker is gone.
**2. The GEOMETRIC bar-pointer WORKS — via the differentiable DBN, not the KVAE filter's free phase.**
   Learned emission trained through the tempo-phase HMM: beat **0.961** (mir), near peak-pick (0.929–0.98),
   far above the KVAE-head geometric (dead, diagnosed as a per-frame-loss degeneracy).
**3. Your dynamic-λ idea works at the GLOBAL level** (+0.013, λ learned 100→57). PER-SONG λ did NOT help
   (collapsed to a constant; needs regularization / tempo-varying data). Over-long λ-training over-softens.
**4. OOD (SMC):** the structured DBN inference on a strong frontend is **competitive and improves
   continuity** (beat-F 0.589 vs peak 0.620; **AMLt 0.656 vs 0.605; beats madmom-DBN 0.575**). The
   from-scratch e2e frontend FAILS OOD (0.135) — the frontend is the OOD bottleneck, not the inference.
**5. End-to-end works** (TCN+filter jointly, leak collapses) but is frontend-limited in-domain (0.473).

**What this means:** we now have TWO working routes — (a) filter + learned head (end-to-end, ~0.88),
(b) **geometric DBN (~0.96, interpretable, dynamic-λ, continuity-improving OOD)** — plus a clean negative
scaffold (amortized wall, KVAE-head geometric, He-2019) that motivates them. The geometric bar-pointer,
the thing that makes this *our* model, is alive. **Biggest open item:** a valid learned-emission DBN OOD
(needs SMC re-extracted with the training extractor) — see Morning to-do.
---


## The result that started this
After the amortized-encoder wall (`experiments/diagram_arch/RESULTS.md`: 6 fixes incl He-2019 all
failed — deploy stuck at ~0.40, input-independent), we pivoted to **Kalman-VAE**: the latent posterior
`q(z|a)` is computed EXACTLY by a Kalman filter, so it can't be dragged to the prior by KL gradients.
Reused `third_party/kalman-vae` (PyTorch, cross-verified vs official TF `third_party/kvae`).

**M1 (certified): WALL BROKEN.** Deploy = Kalman filter on audio → learned head on filtered `z`.
real beat **0.878** / db 0.833; shuffled 0.293/0.126; zero 0.000/0.000 → leak COLLAPSES = audio-driven.

## Reference numbers (same 40-song in-domain val, frozen frontend's own `act2` output)
| | beat-F | downbeat-F |
|---|---:|---:|
| frozen frontend, no DBN (peak-pick) | 0.929 | 0.879 |
| frozen frontend, + madmom DBN | 0.922 | 0.904 |
| amortized wall (diagram_arch) | ~0.40 (input-independent) | — |
| **M1 KVAE (learned head on filtered z)** | **0.878** | 0.833 |

In-domain, the frozen frontend is near-ceiling; M1's win is escaping the wall, not beating peak-pick.
The winnable lines are OOD (SMC-MIREX, where peak-pick is only 0.626), the geometric read-out, and e2e.

---

## STATUS (live)
| job | GPU | what it answers | log | state |
|---|---|---|---|---|
| **M2a** geometric (recon_w=0.3) | 0 | does the GEOMETRIC φ-wrap read-out work with the exact filter? | logs/m2a_recon.log | running |
| **M2b** geometric (recon_w=0) | 2 | same, with z driven purely by phase+kalman | logs/m2b_norecon.log | running |
| **M1-scaled** (2000 songs, 3000 steps, a/z=16, K=8) | 1 | how far does scale close in-domain 0.88→0.92? | logs/m1_scaled.log | running |
| **e2e** from-scratch TCN+KVAE (4 datasets) | 3 | does end-to-end (no frozen frontend) work? | logs/m_e2e.log | running |

## RESULTS (filled as jobs finish)
| job | deploy beat | deploy db | leak (shuf/zero beat) | verdict |
|---|---|---|---|---|
| M1 (done) | 0.878 | 0.833 | 0.293 / 0.000 | WALL BROKEN ✓ (learned head) |
| M2a geometric | 0.35 | 0.000 | — | **FAIL: φ does not rotate** (phi-revs 0); filter PINS the phase |
| M2b geometric (no-recon) | 0.34 | 0.000 | — | FAIL same (killed; redundant w/ M2a) |
| M2-CV (integrate tempo from filter) | running | | | tests forced-rotation + audio-lock |
| M1-scaled | running | | | |
| e2e | running | | | |

### Update (wave 1 nearly done + wave 2 started)
- **M1-scaled**: beat **0.868** / db 0.825 (step ~800, climbing) — reproduces M1 at scale; scale helps modestly.
- **M2a vanilla geometric FINAL**: filter beat 0.345 / **db 0.000 / phi-revs 0**; shuf 0.128 / zero 0.000.
  The 0.345 is φ *jitter* (weak audio corr → partial leak collapse), NOT a rotating pointer. Dead.
- **M2-CV**: beat 0.31 / db 0.09 at a STUCK ~230 BPM, same revs across songs = input-independent grid.
  → geometric-via-KVAE-head fails 3 ways; the failure is a **per-frame-loss degeneracy** (constant-tempo
  grid is a strong local optimum), NOT an inference problem.
- **e2e from-scratch**: beat 0.432 / db 0.308 (step 1200, climbing from 0.254) — viable but frontend-limited.
- **WAVE 2 started — geometric via explicit DBN**: `tests/train_dbn.py` trains a learned emission head
  THROUGH the differentiable `BarPointerDBN` (tempo-phase HMM forward-backward + Viterbi). This structured
  read-out CANNOT collapse to a constant grid (proper observation model rewards phase=0 at audio beats).
  Reports LEARNED-DBN beatF/dbF vs fixed-madmom-DBN vs peak-pick. Running on GPU0 (16756 states).
  Next: add learnable_lambda + per-song dynamic-λ (the user's idea) if the baseline works.

### Key finding (M2 vanilla)
Reading φ = atan2 of a freely-filtered rotational z does NOT rotate (phi-revs ≈ 0): the Kalman
observation-update keeps correcting z back to a near-static estimate, and the per-frame geometric BCE
is satisfied by φ *oscillating* not *advancing*. So **M1's filter win is for the learned-head read-out
(needs an audio-driven latent, not rotation); the geometric read-out needs φ structurally forced to
advance.** Fix = **M2-CV**: φ = integral of a positive tempo read from the filtered latent (integration
guarantees rotation; the exact filter should make the tempo audio-driven; geometric BCE locks wraps to
beats). This merges the two halves that each failed alone (diagram integ = rotated but input-independent;
vanilla M2 = audio-aware but didn't rotate).

---

## ★ HEADLINE REFRAME (the night's key result)
**The geometric bar-pointer WORKS — via the explicit differentiable DBN, not via reading φ off a filter.**
- geometric-via-KVAE-head (M2a, M2-CV): **dead** (φ won't rotate, or rotates as a constant-tempo grid).
  Root cause = a per-frame-loss degeneracy: any "read φ then BCE" objective admits the constant grid.
- geometric-via-DBN (`tests/train_dbn.py` + `dbn_geom2.py`): a learned emission head trained THROUGH the
  tempo-phase HMM forward-backward, deployed by Viterbi. The HMM's observation model rewards "phase=0 at
  the audio's beat frames", so it **structurally cannot** collapse to a constant grid.
  - `train_dbn` (score-metric): LEARNED-DBN beat **0.971** / db 0.837 | fixed-madmom-DBN 0.975 | peak 0.984.
  - `dbn_geom2` (mir_eval, == M1 metric): smoke already ~**0.96** beat (full 30-val number landing now).
  So on the comparable metric the geometric DBN is **near peak-pick and far above M1's 0.878 filter-head**.

This means: the working geometric read-out is the structured DBN inference (which we already had at 0.72-0.91
via SMC); tonight's contribution-relevant news is it's **trainable end-to-end through the differentiable
HMM** and the user's **dynamic-λ** is a clean add (running). The KVAE filter remains the best result for the
*learned-head* read-out and for the end-to-end-from-audio story.

### e2e (from-scratch TCN + exact filter) FINAL — end-to-end claim holds
real 0.473 / shuffled 0.191 / zero 0.000 → leak collapses → genuinely audio-driven, jointly trained whole
stack. Modest number = weak from-scratch frontend, not a broken method.

## Consolidated results (mir_eval metric == M1's, unless noted)
| read-out / model | beat-F | db-F | note |
|---|---:|---:|---|
| frozen frontend peak-pick (ceiling, in-domain) | 0.929 | 0.879 | mir; the bar to clear |
| **M1 filter + learned head** | 0.878 | 0.833 | wall broken; reproduced at scale (~0.87) |
| geometric via KVAE-head (M2a/M2-CV) | ~0.13–0.38 | ~0.0 | DEAD (loss-degeneracy; M2-CV real<zero) |
| **geometric via DBN, fixed-λ** | 0.948 | 0.822 | near peak-pick; the geometric pointer works |
| **geometric via DBN, global learnable-λ** | **0.961** | 0.835 | +0.013; λ learned 100→57 (user's idea ✓) |
| geometric via DBN, per-song λ | 0.71–0.86 | ~0.7 | NEGATIVE: λ-head collapsed to constant 8.2 (clamp floor, std 0), worse than global λ. Pooled-feature→λ signal too weak on steady-tempo data; over-flexes. Global λ is the win; per-song needs reg / tempo-varying data |
| geometric via DBN, scaled (2000 songs, learn-λ) | 0.86 (peaked 0.954) | 0.77 | DEGRADED late: λ drifted 100→42 (over-soft) → needs λ-reg/early-stop; best stays the 400-song 0.961 |
| e2e from-scratch TCN+filter | 0.473 | 0.302 | leak collapses ✓; frontend-limited |

## OOD (SMC-MIREX) status — RESULTS IN
| OOD test (n=217) | beat-F | AMLt | read |
|---|---:|---:|---|
| e2e from-scratch model | **0.135** | 0.061 | FAILS OOD — from-scratch TCN frontend doesn't generalize |
| Beat-This peak-pick (ref check) | 0.620 | 0.605 | ✓ matches published 0.626 → harness valid |
| **Beat-This + our DBN inference** | 0.589 | **0.656** | ~par on F, **beats madmom-DBN (0.575)**, **+continuity (AMLt)** |
- Takeaway: the structured DBN inference is **OOD-competitive on a strong frontend and improves continuity**;
  the from-scratch e2e frontend is the weak link OOD. The DBN test above uses SMC's OWN Beat-This `act2`
  (valid, no transfer). The **learned-emission** DBN OOD still needs matched re-extraction (below).
- **learned-emission DBN OOD: still a morning task** — `smc_rich_heldout` is **Beat-This @ 50 fps**
  (`tests/extract_smc_rich.py`); training (`bt_*_rich`) is **WaveBeat @ 86 fps**; head transfer corr ≈ 0.
  Re-extract SMC with the matching extractor to run the saved learned heads (`dbn_best.pt`).

## Morning to-do (what's left)
1. **Valid geometric-DBN OOD**: re-extract SMC with the training extractor (WaveBeat 86fps) → run the saved
   DBN heads (`dbn_best.pt` etc.). This is THE defensible win-line test.
2. **Geometric-DBN end-to-end**: TCN frontend + DBN-through-training (both halves proven; combine).
3. Decide the paper's spine: filter-route (M1, end-to-end, OOD) vs geometric-DBN-route (0.96, interpretable,
   dynamic-λ). They're complementary — likely both, with the negative results (amortized wall, KVAE-head
   geometric) as the motivating scaffold.
4. Per-song λ vs global λ verdict (running) → is amortizing the DBN hyperparameter worth it.

## Planned wave 2 (after wave 1; chosen by results)
- **SMC-MIREX OOD eval** — cleanest on the **e2e** model (our own log-mel TCN → no frozen-feature fps
  mismatch); load SMC audio → log-mel(86fps) → tcn → filter → head; GT from SMC annotations. This is the
  line we can defend (peak-pick OOD = 0.626). M1's frozen-feature OOD needs 86fps re-extraction (harder).
- **Gears** (`[[project_geared_beat_bar_pointer]]`) — only if M2's geometric read-out shows beats lagging
  downbeats. Implement as a block-rotational `A` (fast beat gear + slow bar gear, ratio m).
- **M2 fixes** if geometric underperforms (rotation/observation balance: tune Q/R init, geom_w, z_dim).
- **M1 scaling follow-up** / feeding the frontend's final `act2` as an extra observation.

## How to read the logs
Each prints periodic `step N | ... deploy: beat X downbeat Y` and a FINAL block with leak controls.
A read-out is REAL only if real >> shuffled ≈ zero. Geometric runs also print `phi-revs` (did φ rotate).
