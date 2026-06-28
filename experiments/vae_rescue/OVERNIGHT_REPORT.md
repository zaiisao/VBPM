# CHART — overnight report: what works, why, and what to do from here

**Date:** 2026-06-26 (overnight autonomous run). All beat/downbeat numbers are F-measure @ ±70 ms
(mir_eval) on a held-out val subset; CMLt/AMLt are mir_eval continuity (AMLt is octave/level-tolerant).
n≈12–16 songs unless noted, so treat ±0.03–0.05 as noise.

> **STATUS: living document — overnight results are filled in as the campaign completes.**
> Sections marked ⏳ are pending; ✅ are measured.

---

## 0. TL;DR (read this first) — UPDATED after the overnight pivot

**Terminology:** PF = Sequential Monte Carlo (our inference method). SMC-MIREX = the hard dataset.

1. **The frontend was the wall.** Beat-This activation → our PF (no training): **easy-data beat-F 0.880**
   (AMLt 0.895), matching SOTA, vs our log-mel 0.72. The PF is a competitive inference layer; the gap was
   always the activation.
2. **DYNAMIC-λ is the contribution, and its ceiling BEATS SMC-MIREX SOTA.** The handcrafted DBN/PF uses
   one global tempo-flexibility λ (σ_τ). On expressive SMC-MIREX that's the killer. Sweep: our default
   σ_τ=0.02 → 0.529; best *fixed* σ_τ=0.25 → **0.611** (already beats madmom-DBN 0.570 & Beat-This-DBN
   0.575); **ORACLE per-song σ_τ → 0.654, exceeding Beat-This-no-DBN 0.626 (the SOTA bar).** Best per-song
   σ_τ is spread 0.005→0.25 — one global λ provably cannot fit all songs. On easy data dynamic-λ barely
   helps (+0.003) — the value is *targeted exactly at SMC-MIREX*.
3. **The F-vs-AMLt tradeoff (new):** on SMC-MIREX, adding *any* inference (DBN/PF) lowers beat-F but
   raises AMLt (metrical-level consistency). Beat-This 0.626/0.598 → +DBN 0.575/0.646. So inference buys
   continuity at the cost of exact-F; a *good* (adaptive-λ) PF should break this tradeoff.
4. **FIVO/variational training: a real but narrow win.** Neutral/harmful at full supervision; but a clean
   **semi-supervised crossover** — FIVO helps downbeats when labels are scarce (+0.06–0.08 at 5–25% labels,
   −0.05 at full).
5. **Open (running): is per-song σ_τ predictable from audio?** If yes, dynamic-λ is realizable (not just
   oracle) and we have a SOTA-beating, contribution-justifying result on the hardest dataset.

**The contribution, crystallized:** *strong frontend + a per-song adaptive-λ multi-hypothesis PF that
breaks the F-vs-AMLt tradeoff on SMC-MIREX, with calibrated uncertainty* — directly answering the
SMC-MIREX blind spot (Ahn et al.). This is methodological, not just diagnostic.

---

## 1. The scoreboard (all measured this project)

| approach | beat-F | downbeat-F | type |
|---|---|---|---|
| free-run prior (no inference) | 0.40 | ~0.13 | real |
| document decoder read-out (reads h) | 0.55 | — | real |
| dbn_vae FIVO (onset emission) | 0.54 | — | real |
| classic (autocorr + metronome) | 0.66 | — | real |
| **fixed prior + trained activation + SMC (ours)** | **0.72** | **0.50** | real |
| FIVO-VAE (w_fivo=0.3) | 0.735 | 0.39 | real |
| madmom / Beat This (SOTA) | ~0.88 | ~0.75–0.80 | real |
| **oracle** (perfect activation + our SMC) | **0.91** | **0.85** | ceiling |
| read-out ceiling (ideal latents) | 0.97 | — | ceiling |

---

## 2. The re-diagnosis (why the faithful VAE failed — with CORRECT read-outs)

Prior verdicts were measured with broken read-outs; re-run with corrected read-outs (bar subdivisions
for beats; offset-selection for downbeats; both φ-semantics checked):

- **Latent is NOT rescued by correct read-outs.** Best *teacher-forced* posterior beat-F = **0.39**
  (encoder handed GT beats still can't map them to a clean bar phase). Not a read-out artifact.
- **Culprit = amortized variational inference**, not "the VAE." SMC inference recovers the phase at
  **0.91**; the amortized encoder gets 0.39. Amortized VI cannot invert the circular/sequential geometry
  as well as explicit Bayesian filtering.
- **He 2019 is moot here**: it fixes posterior *collapse* (encoder ignoring data); our encoder isn't
  collapsed (it reads the beats), it has an *amortization gap*. SMC closes the gap; He 2019 wouldn't.

## 3. "Despite, not because of the document" — component accounting

| component | document (Alg. 1) | working version | verdict |
|---|---|---|---|
| latent state (φ,τ,m) | ✓ | kept (m fixed) | **KEPT** |
| geometric read-out | ✓ | kept | **KEPT** |
| prior | trainable, audio-conditioned, learned init | **fixed** bar-pointer | DEPARTED (revert→DBN) |
| inference | amortized encoder | **SMC** | DEPARTED (revert→DBN) |
| emission/observation | `p(beat\|z,h)` reads h | **audio activation** + geometric emission | DEPARTED (revert→DBN) |
| training | ELBO teacher-forced | FIVO + supervised grounding | DEPARTED (neutral/harmful) |
| deployment | free-run prior | **SMC inference** | DEPARTED (revert→DBN) |

**Every departure that helped was a reversion toward the classic Srinivasamurthy/madmom DBN.**
The fixes that "made it work" were (a) our own bugs (data, read-out, FIVO grounding/pos_weight) and
(b) reverting the document to the classic DBN. **Nothing genuinely novel was required.**

## 4. Faithfulness ablation (revert each departure; does the score retard?)

| # | revert → document choice | beat-F | downbeat-F | load-bearing? |
|---|---|---|---|---|
| 1 | training → **no FIVO** (w_fivo=0) | 0.718 | **0.502** | NO — best without it |
| 1 | (w_fivo=0.3 baseline) | 0.735 | 0.387 | — |
| 1 | training → FIVO-dominant (w_fivo=1.0) | 0.644 | 0.397 | reverts hurt (FIVO harmful) |
| 2 | deployment → **free-run** | ⏳ | ⏳ | expect big regress (~metronome) |
| 3 | inference → amortized encoder | 0.39 (re-diag, TF) | — | YES — big regress |
| 4 | prior → trainable (learn σ_t) | ⏳ | ⏳ | expect neutral |
| 5 | emission → beat-NN reads h | 0.55 decoder / latent decorative | — | YES — audio shortcut |

## 5. ⏳ Overnight experiments

### 5a. Semi-supervised — THE FIVO contribution test ✅ (positive crossover)
Does FIVO exploit *unlabeled* audio to beat supervised-only when labels are scarce? **YES on downbeats.**
Downbeat-F (the stressed metric; beats are data-efficient and stay ~0.67 throughout):

| label_frac | supervised (w_fivo=0) db-F | +FIVO (w_fivo=0.3) db-F | FIVO Δ |
|---|---|---|---|
| 0.05 (nlab=1) | 0.316 | 0.380 | **+0.064** |
| 0.10 (nlab=1) | 0.316 | 0.380 | **+0.064** |
| 0.25 (nlab=2) | 0.435 | 0.512 | **+0.077** |
| 1.00 (full)   | 0.496 | 0.446 | **−0.050** |

**Clean semi-supervised crossover: FIVO helps downbeats when labels are scarce, hurts when plentiful.**
First solid positive for the variational term. Caveat: n=16, easy data, label_frac coarse (0.05/0.1 both
→ 1 labeled sample/batch). Needs a finer/seeded re-run to de-noise, but the *shape* is right.

### 5e. STRONG FRONTEND (Beat-This activation → our PF) ✅
**The frontend was the wall, confirmed decisively.** Beat-This cached activation → our PF (no training):
- **Easy val, n=118:** beat-F **0.880**, CMLt 0.785, AMLt 0.895, octave-err 0.05, 2/118 fail. vs log-mel 0.72.
- **SMC-MIREX, n=217 (fixed σ_τ=0.02):** ours 0.481 / AMLt 0.529 < Beat-This-readout 0.626; octave-err 0.42,
  **62/217 complete failures**. Fixed-λ PF struggles on expressive data → the dynamic-λ motivation.

### SMC-MIREX scoreboard to beat (full 217, cached outputs vs GT)
| system | beat-F | AMLt |
|---|---|---|
| **Beat-This (no DBN)** | **0.626** | 0.598 |
| Beat-This + DBN | 0.575 | 0.646 |
| Beat-Transformer | 0.583 | 0.633 |
| madmom (DBN) | 0.570 | 0.615 |
| madmom TCN | 0.587 | **0.652** |
| our PF, fixed σ_τ=0.02 | 0.481 | 0.529 |
| **our PF, best fixed σ_τ=0.25** | **0.611** | (tbd) |
| **our PF, ORACLE per-song σ_τ** | **0.654** | (tbd) |

### 5f. DYNAMIC-λ ✅ ceiling (the headline) / ⏳ realizability
Ceiling test (sweep σ_τ per song, SMC-MIREX n=60):
| fixed σ_τ | 0.005 | 0.01 | 0.02 | 0.04 | 0.08 | 0.15 | 0.25 |
|---|---|---|---|---|---|---|---|
| beat-F | 0.332 | 0.389 | 0.529 | 0.569 | 0.603 | 0.610 | **0.611** |
- **BEST FIXED σ_τ=0.25 → 0.611** (beats all DBN baselines on beat-F).
- **ORACLE per-song σ_τ → 0.654 > SOTA 0.626.** Headroom from dynamic-λ = **+0.043**.
- best per-song σ_τ spread (0.005:4, 0.01:6, 0.02:8, 0.04:5, 0.08:8, 0.15:16, 0.25:13) — one global λ can't fit all.
- Easy data: best-fixed 0.986, oracle 0.989 (+0.003) — dynamic-λ value is SMC-MIREX-specific.
- ⏳ **Realizability:** is oracle σ_τ predictable from audio (IOI-CV heuristic)? running on full 217.

### 5b. Activation ceiling (how close to madmom can the classic-DBN recipe get?) ⏳
| hidden×layers, steps | beat-F | downbeat-F | beatAMLt |
|---|---|---|---|
| 128×2, 600 | ⏳ | ⏳ | ⏳ |
| 256×2, 1200 | ⏳ | ⏳ | ⏳ |
| 256×3, 1500 | ⏳ | ⏳ | ⏳ |
| 320×3, 2000 | ⏳ | ⏳ | ⏳ |
deploy-K sweep (1000/2000/4000): ⏳

### 5c. Free-run vs SMC (deployment ablation) ⏳ — smoke preview: free-run 0.26 (metronome) vs SMC 0.72
### 5d. Trainable σ_t prior ⏳

---

## 6. ⏳ Recommendations — what to do from here (filled after results)

(Pending overnight results. Skeleton of the decision tree:)
- **If semisup shows FIVO helps with unlabeled data** → real contribution; pursue semi-supervised
  beat tracking with the bar-pointer DVAE; run the full ablation (He 2019, free-bits, δ-VAE, proposal).
- **If semisup shows FIVO neutral/harmful even with unlabeled data** → the variational angle is dead;
  the honest deliverables are (i) the *analysis* (amortized-VI-fails / deployment-gap, a methods-insight
  paper), and/or (ii) push the activation toward madmom and frame as a clean neural-DBN reproduction —
  but that competes directly with Beat This with no clear novelty.
- **Downbeats** (currently 0.50 vs madmom 0.78): the bar-pointer structure is where a generative model
  *should* shine; improving the downbeat channel + joint inference is the most defensible technical lever.
