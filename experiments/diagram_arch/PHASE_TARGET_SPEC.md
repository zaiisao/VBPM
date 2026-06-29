# Phase-target construction — from Oyama, Ishizuka & Yoshii, ISMIR 2021 (ref [42])

"Phase-Aware Joint Beat and Downbeat Estimation Based on Periodicity of Metrical Structure."
This is the exact recipe Chen & Su's label-embedding (ISMIR 2022) builds on. Copy it rather than reinvent.

## Core idea
Predict **beat *phase*** at each frame, not beat *presence*. Phase = a **semi-continuous sawtooth**:
reset to 0 at each beat frame, increases linearly to 2π by the next beat. Period = inter-beat interval.
Because every frame (not just the ~1.5% beat frames) carries a meaningful target, backprop gets a DENSE
signal — directly fixing our "rate-blind sparse Bernoulli" (link 1). The sawtooth's slope IS the tempo.

## 1. Phase target as K-class classification (NOT continuous regression)
The authors note continuous phase regression "often fails to decrease the estimation loss without
careful pretraining." So phase is **quantized into K classes**, resolution 2π/K.

Beat phase one-hot z^b_t ∈ {0,1}^K:
    z^b_{tk} = 1  iff   2π(k-1)/K ≤ φ^b_t < 2πk/K ,   else 0
where φ^b_t is the sawtooth (0 at beat → 2π at next beat).

**Blurry (soft) target** — they do NOT use a hard one-hot; around the active class k*:
    z*_{t,k*}   = 1.00
    z*_{t,k*±1} = 0.75
    z*_{t,k*±2} = 0.50
    z*_{t,k*±3} = 0.25
(circular neighbors mod K). This is the key trick that makes it trainable.

## 2. Downbeat / bar phase
Same construction, period = one **measure** (bar). Class index:
    k_d = round( K · S^p_t / N_t ),   N_t = BPB · S^v_t  (BPB = beats per bar)
i.e. a sawtooth over the bar. (Chen & Su used K_b=150 for beat, K_db=500 for bar.)

## 3. Loss (per-frame soft cross-entropy)
DNN outputs ψ*_t ∈ [0,1]^K (softmax). Train to maximize:
    J_phase = (1/T) Σ_t Σ_k  z*_{tk} · log ψ*_{tk}            (Eq. 1)

## 4. Network
log-spectrogram → Feature embedding → TCN×11 → Decoder → **softmax over K** (per frame).
(Böck [9] TCN with skip connections removed; Decoder ends in Dense→Softmax.) For us: replace the
front with our frozen Beat-This h_{1:T}; just learn h_t → softmax over K phase classes.

## 5. Decoding beat/downbeat TIMES at inference
A modified bar-pointer **DBN/HMM** whose observation model is the phase (not presence):
    p(X|S) ∝ Π_t ψ^b_{t,k_b} ψ^d_{t,k_d}                      (Eq. 12)
    k_b = round[ K · (S^p_t mod S^v_t)/S^v_t ],  k_d = round[ K · S^p_t / N_t ]   (Eq. 13–14)
So they STILL use a DBN for the final times — but on a far cleaner (phase) observation. A no-DBN
fallback: beat time = where the predicted sawtooth wraps (argmax class resets to ~0).

## 6. TEMPO is COMPUTED from the phase, NOT learned  ← directly confirms our diagnosis
Global tempo via DFT of the phase sinusoid:
    y_t = sin( (2π/K) · argmax_k ψ^b_t )                       (Eq. 15)
    DFT(y) → ω_max = angular velocity of the largest-magnitude coefficient
    V = 60 · ω_max · fps / (2π)   [BPM]                        (Eq. 16)
=> Even this phase-aware SOTA paper does **not** gradient-learn a tempo/rate latent. It predicts the
phase (well-conditioned classification) and reads tempo off the phase's frequency. This is exactly our
"compute the tempo, don't learn it" conclusion, and the sawtooth is the bridge.

## 7. Differentiable phase (for end-to-end multitask coupling)
To pass a phase into a downstream net differentiably:
    ẑ*_t = (2π/K) · aᵀ · Gumbel-softmax(ψ*_t),   a = [1,2,…,K]ᵀ   (Eq. 17)

## Implication for CHART
- Replace/augment the rate-blind decoder target with the **sawtooth phase (K-class, blurry, CE)** for
  both beat-phase and bar-phase. This gives φ a dense, rate-informative gradient and forbids the
  oscillation cheat (link 4) — without needing positional encoding or an explicit integrator.
- Get φ̇ (tempo) the way they do: **DFT of sin(φ)** (global) or local slope of φ (per-frame), i.e.
  computed from the predicted phase. Do NOT expect a free φ̇ latent to lock.
- Keep our generative prior if we want faithfulness: the supervised phase grounds φ; KL coupling
  φ_t ~ φ_{t-1}+φ̇_{t-1} then back-fills φ̇ from the supervised phase slope.
- Leak-test everything (real vs shuffle vs zero) and check φ-revs ≈ #bars, tempo ≈ GT.
