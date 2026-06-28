# Diagram-architecture results: the amortized-encoder wall is real, the fix is a differentiable filter

**Date:** 2026-06-28/29
**Branch:** faithful/strict-elbo
**TL;DR:** The diagram architecture (feature extractor → amortized encoder `q(z|h)` → latent-only
decoder, beats read **geometrically** from the bar-phase φ at deploy) is sound *given a correctly
rotating φ* — a clamp test gives **0.98**. But no amount of training makes the **amortized encoder**
produce an **audio-locked rotating φ** when free-run/h-only. Six independent fixes (phase-sup,
tempo-sup, fixed-emission, gated constant-within-beat, gated-sharp, He-2019 aggressive inference) all
land in exactly one of two bins — *audio-driven but won't rotate*, or *rotates but ignores audio* —
and never both. This is **posterior collapse of the amortized latent**, with a clean mechanism: the
per-frame KL gradient drags `q` to the prior faster than reconstruction can teach it to lock onto
audio. The decoder *does* learn to use `h` (its leak controls collapse), but it reads φ as a generic
feature, not as a geometric clock — which is precisely the head the design discards. **The route that
works is filtering inference, not amortized inference** (our particle-filter / "DBN" pipeline = 0.91 on
the same data). To keep the end-to-end-training contribution, the next step is a **differentiable
filter** (reparameterized/Gumbel particle filter, or DVBF-style backprop-through-transitions) that
replaces `q` while keeping the frontend and dynamics jointly trainable.

---

## 1. The architecture and its two read-outs

```
audio ─► feature extractor ─► h ─► [TRAIN ONLY: q(z|h, beats) encoder] ─► z=(m, φ, φ̇)
                              │                                              │
                              └────────► prior p(z_t|z_{t-1}) reads h ◄──────┘
                                                                            │
                            decoder p(b̂|z)  (TRAIN ONLY, never reads h) ◄───┘

DEPLOY (h-only, free-run, no posterior):
  • GEOMETRIC read-out (the design's deployment): beats = wraps of m·φ, downbeats = φ wraps (2π→0)
  • decoder read-out (a TRAIN-ONLY head, reported only as reference / oracle)
```

The contribution claim is **end-to-end trainability** (train the frontend + dynamics jointly; no
frozen-frontend-then-bolted-on-DBN). The headline metric is the **geometric** read-out. The decoder is
a training device and is reported only as a reference ceiling.

## 2. The decisive clamp proof — the read-out is fine, rotation is the only missing piece

`clamp_test.py`: clamp φ to the GT bar-phase ramp and read beats geometrically.

| condition | geometric beat | note |
|---|---:|---|
| φ clamped to GT ramp | **0.98** | the read-out + decoder are correct given correct φ |
| decoder following clamped φ | 0.73–0.90 | decoder tracks φ |
| tempo φ̇ under clamp | — | does **not** slave (1052–1551 BPM vs true ~128) |

**Conclusion:** every downstream component works. The entire problem reduces to one thing — making the
**free-run φ rotate and lock to this song's beats from audio alone**.

## 3. Two separable problems

- **#1 rotation**: does φ advance (revolve) at all? (clean ramp)
- **#2 audio-locking**: does the ramp lock to *this* song's beat positions? (the ramp must be driven
  by `h`, not be a generic metronome)

Free training: φ is static (~0 revolutions, tempo→0) → fails #1. Forcing rotation (#1) converts the
degeneracy into outright **input-independent rotation** → fails #2.

## 4. Campaign: pretrained-frozen (A), from-scratch end-to-end (B), SMC-MIREX

`run_campaign.sh` → `campaign.log`. Leak controls here are on the **decoder** read-out.

**A — pretrained-frozen Beat-This features + VAE (4 WaveBeat datasets, in-domain val, 600 steps):**
```
real h     : decoder beat 0.591  db 0.815   |  geometric beat 0.029  db 0.000
shuffled h : decoder beat 0.191  db 0.091
zero h     : decoder beat 0.000  db 0.000
```
The **decoder genuinely uses h** (leak collapses 0.591→0.191→0.000). But the **geometric pointer is
dead** (0.029 beat / 0.000 db). The decoder works by reading φ as a generic feature, not as a clock.

**B — from-scratch end-to-end TCN(log-mel)+VAE (4 datasets, random init, 1500 steps):**
```
real audio : decoder beat 0.547  db 0.382   |  geometric beat 0.471  db 0.206
shuffled   : decoder beat 0.226  db 0.068
zero       : decoder beat 0.000  db 0.000
```
End-to-end-from-scratch *trains* (decoder leak collapses → it uses audio), proving the joint
optimization is viable — but the geometric pointer is still weak and cross-dataset it falls apart.

**SMC-MIREX (out-of-domain, n=217; ref: Beat-This noDBN 0.626 / +DBN 0.575 / madmom 0.570):**
```
A  decoder beat-F 0.431  AMLt 0.276  | phase-readout 0.143
B  decoder beat-F 0.194  AMLt 0.087  | phase-readout 0.132
```

## 5. The six encoder-route fixes — all fail #2

Each test deploys h-only and reports the **geometric** read-out plus leak controls (real vs shuffled
vs zero). A genuine fix needs geometric **high AND leak collapsing**.

| test | file | result | bin |
|---|---|---|---|
| clamp (control) | `clamp_test.py` | geometric **0.98** given GT φ | proves read-out |
| phase-supervision | `phasesup_test.py` | under-rotates (φ-revs ~3); beat ~0.31 | audio-driven, won't rotate |
| integ-φ + tempo-sup | `integ_test.py` | generic metronome; leak fails (~0.24) | rotates, ignores audio |
| integ-φ + fixed emission | `fixedemis_test.py` | generic max-tempo grid; leak fails (~0.40, tempo pinned 250) | rotates, ignores audio |
| gated (tempo const within beat) | `gated_test.py` | generic grid; leak fails (~0.38) | rotates, ignores audio |
| gated + κ=20 + sharp pos-weight | `gated_sharp_test.py` | real 0.391 / shuffled 0.391 / zero 0.388 | rotates, ignores audio |
| **He-2019 aggressive (K=5)** | `he_test.py` | see below | rotates, ignores audio |

**He-2019** (the strongest encoder-side mechanism — K=5 encoder-only updates per decoder update, to
maximize `I(x;z)` before collapse sets in):
```
decoder ref        : 0.846                         ← the DISCARDED head un-collapses
geometric real     : beat 0.380  db 0.096  φ-revs 19.3
geometric shuffled : beat 0.390  db 0.104          ← does NOT collapse (HIGHER than real)
geometric zero     : beat 0.382  db 0.080          ← does NOT collapse
```
He did exactly what it is designed to do — it un-collapsed the **generative/decoder** path (0.000 →
0.846). But the **geometric pointer** is 0.380 on real audio, 0.390 on shuffled audio, 0.382 on zeroed
audio: **identical**. φ rotates a healthy 19.3 revolutions, but rotates the *same way regardless of
input*. It is a generic ~120 BPM metronome that clips 38% of beats on a 4/4 set by pure periodicity.
`I(audio; φ) ≈ 0`.

## 6. Verdict — the amortized-inference wall, with a mechanism

Every fix sorts into exactly two bins; **nothing ever produced audio-locked rotation**. That is not
six unlucky configs — it is posterior collapse of the amortized latent. Mechanism (consistent with our
earlier KL-gradient measurement, ~15–100× the reconstruction gradient on the inference net): the
**per-frame KL term drags `q(z|h)` to the prior faster than reconstruction can teach φ to lock onto
audio**. The decoder can still "use h" because it reads the latent as an arbitrary feature — but that
is the train-only head, not the geometric clock the design deploys.

This matches DEEP_RESEARCH_2 Finding G (Kalman-VAE) almost verbatim: *because the dynamics are
linear-Gaussian, the z-posterior is computed exactly by Kalman filter/smoother **rather than
amortized** — so it cannot be dragged to the prior by KL gradients.* The culprit is **amortizing the
latent posterior into an encoder**, not the bar-pointer model and not the read-out.

The counter-proof is on our own data:

| inference | geometric/beat deploy |
|---|---:|
| amortized encoder (this campaign) | ~0.03–0.47 (input-independent at the high end) |
| decoder feature-degeneracy (reference head) | ~0.59–0.85 |
| **filter (particle-filter / "DBN")** | **0.91** |

## 7. Forward: a differentiable filter (keeps the end-to-end contribution)

Replace the amortized `q` — the one component that provably cannot work here — with the one that
provably does (a filter), while keeping the frontend + dynamics jointly trainable:

- **Reparameterized / Gumbel particle filter** (differentiable resampling) over the bar-pointer state,
  trained end-to-end through the frontend. Our SMC pipeline already gets 0.91; making it differentiable
  preserves "no frozen-frontend-then-bolted-on-DBN".
- **DVBF-style backprop-through-transitions** (DEEP_RESEARCH_2 Finding G): forces the latent space to
  conform to the dynamics and *"enables realistic long-term prediction via free-running the learned
  dynamics"* — directly the free-run-deploy regime.
- Composable add-ons from DEEP_RESEARCH_2 if needed: latent overshooting (multi-step rollout
  consistency, mode 2) and a VRNN-style context-conditioned prior.

**Recommendation:** stop spending compute on the amortized encoder. The wall is a genuine, defensible
finding. Pivot to the differentiable filter.

---

### File index
- `run.py` — diagram architecture (BPVAE, rollout, elbo_loss, evaluate, leak controls), saves `checkpoints/diagram_A.pt`
- `e2e.py` — from-scratch TCN(log-mel)+VAE end-to-end on 4 WaveBeat datasets, saves `checkpoints/diagram_B.pt`
- `smc_eval.py` — SMC-MIREX eval (A cached feats / B audio→log-mel)
- `run_campaign.sh` — tmux runner (A → B → SMC), logs to `campaign.log`
- `clamp_test.py` — proves geometric read-out = 0.98 given correct φ
- `phasesup_test.py`, `integ_test.py`, `fixedemis_test.py`, `gated_test.py`, `gated_sharp_test.py`, `he_test.py` — the six encoder-route fixes
