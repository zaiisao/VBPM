# BeatFM Reproduction — Mathematical Faithfulness Audit

**Target:** `external/beatfm_repro/` — a from-scratch reimplementation of
**BeatFM: Improving Beat Tracking with Pre-trained Music Foundation Model** (Ru et al.,
arXiv **2508.09790v2**). No official code exists; the repro is built from the paper text.

**Scope (as revised mid-task):** this audit covers the **BeatFM reproduction vs its paper only**.
The VBPM-core re-derivations that were originally in the brief (wrapped-Cauchy KL closed form,
the rollout's phase-KL conditioning, the free-bits gradient behaviour) were **removed from scope**
at the user's request and are **not** covered here. Nothing under `model/`, `losses.py`,
`config.py`, or `data/targets.py` is assessed below except `data/feature_extractor.py`, which is
referenced only as the trusted baseline for the MERT pos_conv remap.

**Method.** Every file in `external/beatfm_repro/` was read directly. Findings were then
cross-checked by five independent subagents (adversarial re-derivation of MSAM, of the
data/loss/decode path, and of the frontend remap; a verbatim re-fetch of the paper for exact
wording; and a reasonableness pass over the README ledger). The paper's exact spec sentences were
re-extracted from the arXiv HTML and are quoted where they matter. All GPU work was avoided (all
four GPUs are running live training); this was a read-only source audit.

**Bottom line.** The reproduction is **faithful**. No mathematical errors were found — the MSAM,
the label broadening, the loss, the DBN decode, the metrics, and the MERT frontend load all match
the paper (and, for the frontend, the trusted VBPM reference) once the documented ambiguities are
granted. The residual risk to *matching the paper's numbers* is dominated by one **documented**
data deviation (Harmonix excluded), not by any coding fault. Only one substantive item is a
genuine deviation **not** already in the README ledger (the beat-decode path), and it is standard
practice.

---

## 1. Confirmed-faithful items

Each row was verified against the paper's exact wording (quoted where decisive) and, for the
frontend, against the trusted VBPM reference.

| # | Item | Location | Verdict / paper anchor |
|---|------|----------|------------------------|
| F1 | **MSAM three-branch structure** (temporal, frequency, channel) combined `Attn = Attn_t·Attn_f·Attn_c`, applied residually `h̃ = h + Attn⊙h` | `external/beatfm_repro/beatfm/msam.py:77-94` | Matches paper Eqs. (9)–(10). Broadcasting verified: `a_t→[b,n,1,t]`, `a_f→[b,n,f,1]`, `a_c→[b,n,1,1]` broadcast to `[b,n,f,t]`; pooling axes (f for temporal, t for frequency, both for channel) are correct. |
| F2 | **MS-Conv**: M=4 parallel dilated 1-D convs, dilations `[1,2,4,8]`, concat → MLP → sigmoid; branches unshared | `external/beatfm_repro/beatfm/msam.py:25-44,73-74` | Matches paper Eqs. (3)–(5) and Sec. IV-B ("M=4 parallel 1D convolutions, with dilation rates set to [1,2,4,8]"). `padding=d·(k−1)//2` preserves length; concat → `[b,4n,L]` → 1×1-conv MLP `4n→2n→n`. Kernel/width unstated by paper (see ledger #2). |
| F3 | **Channel self-attention** `Attn_c = σ(Conv(Softmax(QK^T/√·)V))` with Q,K,V from convs | `external/beatfm_repro/beatfm/msam.py:47-66` | Structure matches paper Eqs. (6)–(8). Scale constant is a documented invented dim — see §2.2. |
| F4 | **Per-frame FC classifier** flatten `(n,f)` → 512 → 2 logits (beat, downbeat) | `external/beatfm_repro/beatfm/msam.py:97-113` | `permute(0,3,1,2)→[b,t,n,f]→reshape[b,t,n·f]` matches `Linear(n·f,512)`; output `[b,t,2]`. Hidden width unstated by paper (ledger #5). |
| F5 | **Label broadening** ±1 frame→0.5, ±2→0.25, center→1.0, as **soft BCE targets**, for **both** beats and downbeats | `external/beatfm_repro/beatfm/data.py:34-45` | Paper IV-B: *"…frames adjacent to the annotated beat frames (±2 frames) are also labeled as beats but with reduced weights of 0.5 and 0.25, respectively"* and *"…for both beat and downbeat annotations."* `np.maximum` makes overlap order-independent; the `{off,-off}` set correctly dedups `{0,−0}`. |
| F6 | **Clip windowing**: 15-s clips, 5-s overlap (hop 10 s), `nf = round(15·75) = 1125` | `external/beatfm_repro/beatfm/data.py:20-22,77` | Paper IV-B: *"…15-second clips with a 5-second overlap."* 15 s·24000/320 = 1125 exactly; frame-count slack absorbed by the `min()` trim in `masked_bce`. |
| F7 | **Training protocol**: Adam, lr 3e-4, batch 16, song-exclusive train/val split, early stopping patience 20 on val loss | `external/beatfm_repro/train.py:63-64,95,102-119` | Paper IV-B verbatim: *"Adam optimizer with a learning rate of 3×10⁻⁴ and a batch size of 16 … Early stopping … when the validation loss does not decrease for 20 epochs."* |
| F8 | **Multi-task masked BCE** — equal-weight beat + masked-downbeat, both per-frame means (scale-matched); SMC downbeat masked | `external/beatfm_repro/train.py:27-35` | Paper: BCE, multi-task. Beat term = mean over all `b·t`; downbeat term = masked mean over `(#db-songs·t)`; `.clamp(min=1)` guards zero-db batches. Sound. |
| F9 | **madmom DBN decode** at `fps=75`; beat DBN + joint downbeat DBN(`beats_per_bar=[3,4]`); `min/max_bpm 55/215`, `transition_lambda 100` are madmom defaults | `external/beatfm_repro/beatfm/decode.py:13-36` | Paper III-C: probabilities *"refined through a Dynamic Bayesian Network [24]"* — **no DBN params or fps stated** (ledger #1/#11). Only real override is `fps=75` (native MERT rate; madmom default is 100). |
| F10 | **Downbeat activation conversion** to madmom's `(beat-not-downbeat, downbeat)` 2-column format with row-sum < 1 | `external/beatfm_repro/beatfm/decode.py:29-32` | Correct RNNDownBeatProcessor input convention; `col0 = beat−db` is valid because the beat head is trained on *all* beats incl. downbeats. Negative clip → 1e-8, rescale rows > 0.99. This is the only sensible conversion, not a free choice (ledger #9). |
| F11 | **Metrics**: 5-s trim, 70 ms F-measure, CMLt/AMLt from `mir_eval.continuity` | `external/beatfm_repro/beatfm/metrics.py:9-13` | `trim_beats` default `min=5.0`; `f_measure_threshold=0.07`; matches paper (70 ms tolerance; Tables I/II CMLt/AMLt). |
| F12 | **MERT frontend pos_conv weight-norm remap** — byte-identical to the trusted VBPM reference | `external/beatfm_repro/beatfm/mert_frontend.py:26-35` vs `data/feature_extractor.py:117-125` | Same two `(old,new)` pairs, same order, same `if old in state` guard. **This load is load-bearing** — without it `pos_conv` is random and all features are garbage. |
| F13 | **pos_conv remap orientation is correct** (`weight_g→original0`, `weight_v→original1`) | `external/beatfm_repro/beatfm/mert_frontend.py:27-30` | Independently verified in the env's torch 2.6.0 source: `_WeightNorm.right_inverse` returns `(g, v)` → `original0=g` (magnitude), `original1=v` (direction); torch's own `_weight_norm_compat_hook` performs the identical `{name}_g→original0`, `{name}_v→original1`. The feared g/v swap is **not** present. |
| F14 | **Remap fails loudly** on a target-side naming mismatch (`assert not unexpected`) | `external/beatfm_repro/beatfm/mert_frontend.py:34-35` | If the model used old naming, the injected `original0/1` keys land in `unexpected` and the assert fires — cannot silently reinitialize. A correct load cannot be silently corrupted either. |
| F15 | **Frontend forward** stacks all 13 hidden states → `[b, n=13, f=768, t]`; `num_layers = num_hidden_layers+1` | `external/beatfm_repro/beatfm/mert_frontend.py:39,50-52` | `output_hidden_states` → 13 tensors `[b,t,f]` (embed + 12 layers); `stack(dim=1)`+`transpose(2,3)` → `[b,13,768,t]`. Matches paper's `h ∈ ℝ^{b×n×f×t}`. |
| F16 | **Per-clip waveform normalization** (zero-mean/unit-var), applied consistently in train and inference | `external/beatfm_repro/beatfm/mert_frontend.py:55-59`; `train.py:42`; `inference.py:40` | Dictated by MERT's preprocessor (`do_normalize=true`); train/eval consistent (raw cache is `/32768` only). See §2.3 for the biased-vs-unbiased variance nit. |
| F17 | **Full-song inference** by 15-s / 50%-overlap sliding window, averaging frame **probabilities** | `external/beatfm_repro/beatfm/inference.py:14-47` | Paper evaluates on complete pieces; MERT's quadratic attention forces windowing (ledger #6). Per-window normalization matches training; overlap-averaging smooths window seams. |
| F18 | **Dataset roles** — train-only {Beatles, RWC-Popular}; 8-fold CV {Ballroom, Hainsworth, SMC}; test-only GTZAN; SMC downbeat masked | `external/beatfm_repro/beatfm/index.py:17-19,82` | Matches paper IV-A **except Harmonix** (a paper train-only set) — see §3, R1. |
| F19 | **Audio cache build** — soundfile + `scipy.resample_poly` to 24 kHz mono int16 (soxr avoided; anti-aliased polyphase) | `external/beatfm_repro/scripts/build_cache.py:21-36` | Correct resampler with implicit AA filter; MERT's native rate is 24 kHz. |

---

## 2. Deviations NOT already in the README ledger (severity-ranked)

Ranking order per the brief: **math errors > undocumented deviations > style**. **No math errors
were found.** The list therefore opens at the undocumented-deviation tier.

### 2.1 — Undocumented deviation (MEDIUM): beats decoded from the *separate* beat-DBN, not the joint downbeat-DBN
`external/beatfm_repro/eval.py:47` (and `:50`)

Beat times come from `DBNBeatTrackingProcessor(act[:,0])`; downbeat times come from a *separate*
`DBNDownBeatTrackingProcessor` run whose own beat output is discarded. Taking beats from the
dedicated beat DBN is the conventional Böck/madmom route and generally yields **higher** beat-F
than reading beats off the bar-constrained joint DBN (which forces beats onto a `[3,4]` grid and
can break continuity on other meters). The paper does not state which DBN produced its beat
numbers. This is a reasonable, standard choice, but it is a real degree of freedom that shifts
Table I/II beat-F and is not captured anywhere in the ledger. Impact: small but nonzero on beat
F/CMLt.

### 2.2 — Undocumented deviation (LOW, immaterial): channel-attention scale reinterprets the paper's `√C`
`external/beatfm_repro/beatfm/msam.py:57`

`self.scale = qkv_dim**-0.5 = 1/√16 = 0.25`. The paper's Eq. (8) writes `Softmax(QK^T/√C)`. If `C`
denotes the channel count `n=13`, the literal value would be `1/√13 = 0.277`. The ledger (#4)
documents the `qkv_dim=16` choice and even records "scale = 1/√16", but does **not** flag that this
*reinterprets* the paper's `C`. Immaterial in effect: Q,K are learned conv projections, so any
constant rescaling is absorbed into the weights; and `1/√d_k` with `d_k=16` (the true inner-product
dimension of `QK^T`) is the textbook-correct scaled-dot-product form, arguably more principled than
the literal `√C`. Flagged only because the reinterpretation itself is undocumented.

### 2.3 — Style (immaterial): `normalize_wav` uses unbiased (ddof=1) variance
`external/beatfm_repro/beatfm/mert_frontend.py:58`

`var(..., unbiased=True)` divides by `N−1`; HF `Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm`
uses population variance (`np.var`, ddof=0). For a 15-s clip (`N ≈ 360 000`) the scale differs by
`√(N/(N−1)) ≈ 1 + 1.4e-6` — far below float32 feature noise. `eps=1e-7` matches HF exactly. Pure
docstring nit ("Wav2Vec2FeatureExtractor-style"); no measurable effect. (Note: the VBPM *reference*
frontend applies **no** normalization at all, so BeatFM is if anything *closer* to `do_normalize=true`.)

### 2.4 — Style (immaterial): int16 encode/decode scale asymmetry
`external/beatfm_repro/scripts/build_cache.py:35` vs `external/beatfm_repro/beatfm/data.py:72`

Cache encodes with `×32767` then decodes with `/32768.0` — a ~3e-5 relative gain error, erased by
the subsequent per-clip unit-variance normalization. No effect.

### 2.5 — Style/doc (immaterial): two docstring premises are version-dependent
`external/beatfm_repro/beatfm/mert_frontend.py:1-7`

The docstring states transformers ≥4.31 "silently RE-INITIALIZES" pos_conv. On the env's torch
2.6.0 there is a `_weight_norm_compat_hook` that already auto-remaps `weight_g/v → original0/1` at
load, so on this stack the manual remap is at worst **redundant** (still correct and harmless), and
the "silently re-initializes" premise is not universally true. Documentation accuracy only; the
code is safe either way (F13/F14).

### 2.6 — Minor: `--max-epochs 200` cap not in the paper
`external/beatfm_repro/train.py:65`

The paper specifies early stopping but no epoch cap. A 200-epoch ceiling is added. Almost always
inert (patience-20 early stopping fires first); could in principle truncate a still-improving run.
Negligible.

> Items already correctly recorded in the ledger (frame rate, broadening-as-values, invented MSAM
> dims, GTZAN averaging, val split, DBN tempo range, inference overlap, the remap fix, and all
> local-data caveats) are **not** repeated here; they were each verified true-to-code and appear in
> §3 where they carry result risk.

---

## 3. BeatFM-repro risk list — threats to matching the paper's numbers

Ranked by how much each could move reported accuracy vs Tables I–III. All were verified against the
code; ledger cross-references are given.

**R1 — HIGH — Harmonix training set excluded.** `README.md:47-49`; `beatfm/index.py:17` (`TRAIN_ONLY = ["beatles","rwc_popular"]`, no Harmonix anywhere).
The paper trains on **Beatles + RWC-Popular + Harmonix** (IV-A, verbatim). Harmonix (~900 pop songs
with downbeats) is a large fraction of the training pool and the main downbeat-supervision source.
The README itself predicts a cost "especially on GTZAN" (pop-heavy, test-only). This is a
*documented, deliberate* deviation (local Harmonix labels are misaligned with the available audio),
but it is the **dominant** reason repro numbers may fall short — most acutely on GTZAN beat/downbeat
F (Table I) and on downbeat metrics generally. Any comparison to the paper must caveat this.

**R2 — MEDIUM — 75 fps everywhere vs the baselines' 100 fps.** `README.md:55` (ledger #1); `mert_frontend.py:15-16`, `data.py:77`, `decode.py:13-24`.
MERT's hidden states are natively 75 fps, so 75 is the physically correct choice for a faithful
MERT repro; the risk is only that the paper may have interpolated to 100 fps to match baseline
protocol (unstated). Beat-F is tolerance-limited (70 ms ≫ 13.3 ms frame) so largely unaffected;
the residual risk is a slightly coarser DBN position/tempo grid → small possible CMLt/downbeat
movement.

**R3 — MEDIUM — Label broadening as soft target *values* vs loss *weights*.** `README.md:60` (ledger #3); `data.py:40`, `train.py:30`.
Training the net to emit 0.5 at ±1 frame **shapes the activation ridge** the DBN consumes. The
paper's "reduced weights" could instead mean per-frame loss weights on hard 0/1 labels (→ sharper
peaks, different DBN behaviour). Soft-target widening is the standard Böck/TCN reading the README
cites and the more likely intent, but it can move CMLt/AMLt/downbeat-F at the margin.

**R4 — MEDIUM — Beat/downbeat decode path (see §2.1).** `eval.py:47`.
Beats from the dedicated beat DBN vs a joint-DBN beat output is an unstated choice that shifts
beat-F. Standard practice, likely faithful, but not free.

**R5 — MEDIUM — GTZAN scored as mean single-model, not an ensemble.** `README.md:68` (ledger #7); `eval.py:74-76`.
GTZAN is scored by each of the 8 fold models and the per-fold aggregates are averaged → **mean
single-model** performance. A probability-level ensemble would generally score higher; a single
designated fold would differ too. The paper does not state which model evaluates GTZAN, so the
headline Table I number can differ by a small margin either way.

**R6 — MEDIUM (localized to SMC) — DBN tempo range fixed to 55–215 BPM.** `README.md:76` (ledger #11); `decode.py:15,23`.
madmom defaults, faithful *if* the paper used defaults. SMC contains sub-55-BPM songs
(e.g. smc_001 ≈ 50 BPM) the DBN cannot lock at true tempo — creditable only at AMLt — depressing
SMC beat-F/CMLt. SMC is already the hardest set (paper beat-F 61.3), so this is a plausible
point-or-two gap if the paper widened the range.

**R7 — LOW — Invented MSAM/classifier dimensions.** `README.md:57,62,64` (ledger #2/#4/#5); `msam.py:28-40,50-57,100-108`.
MS-Conv kernel=3 & MLP `4n→2n→n`; channel `qkv_dim=16`; classifier hidden 512 — all unstated by the
paper. Plausible, modest capacity sensitivity. Per the paper's own Table III, the channel branch
adds only ~0.6 beat-F, bounding the channel-attn risk to tenths.

**R8 — LOW — Validation split scheme.** `README.md:70` (ledger #8); `index.py:102-118`.
val = fold `(test_fold+1)%8` of every foldable dataset (Beatles/RWC fold files verified to exist,
so they are genuinely held out, not silently all-train). Song-exclusive and standard; early
stopping on this val loss could pick slightly different checkpoints than the paper's unstated scheme.

**R9 — LOW — Full-song inference boundary effects.** `README.md:66` (ledger #6); `inference.py:14-47`.
50%-overlap probability averaging vs a true whole-song forward → minor seam effects, small under
70 ms tolerance.

**R10 — LOW / benign — Local missing audio.** `README.md:81-84`; `index.py:55-93`.
Ballroom 672/685 (13 known duplicates, standard de-dup), Beatles 179/180, GTZAN 993/999,
RWC-P 100/100. All <1% per set; negligible on dataset-level aggregates. Ballroom de-dup if anything
improves evaluation validity.

---

## Appendix — verification provenance

- **Paper spec** re-fetched verbatim from `https://arxiv.org/html/2508.09790v2` (Secs. III-B, III-C,
  IV-A, IV-B). Confirmed the paper states **no** fps, **no** DBN parameters, and does **not** name
  MERT-v1-95M / 13 hidden states / 768-dim numerically (MERT referenced only as `[21]`, features as
  `h = Concat([h_1,…,h_n])`) — so every such value in the repro is a defensible implementer choice,
  logged in the ledger.
- **pos_conv orientation (F13)** confirmed by reading the env's `torch/nn/utils/parametrizations.py`
  (`_WeightNorm.right_inverse`/`forward` and `_weight_norm_compat_hook`) — not by inference.
- **MSAM, loss, decode, frontend** each re-derived by an independent adversarial reader; all
  broadcasting/dimension/convention checks passed with no math error.
- No GPU code was run (all four GPUs busy with live training); source-only, read-only audit.
