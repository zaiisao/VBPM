# ELBO for DBN — Jaehoon Ahn (March 2026)

> **Saved transcription of the source paper** (the binary PDF reaches the assistant only as
> rendered text, so this markdown is the durable copy). This is the authoritative spec the
> `faithful/` package is measured against. Section numbers match the paper.

## 1. Introduction
The bar pointer model [Whiteley 2006] jointly infers tempo, meter, rhythmic pattern. A pointer
sweeps one bar at a rate ∝ tempo; at the end of the bar it wraps. These models are DBNs; in the
beat-tracking literature the multiple hidden variables (tempo, bar position, meter) are usually
collapsed into one composite HMM state for exact Viterbi / forward-backward inference [Krebs 2015,
Whiteley 2006]. **Our model retains the full DBN structure with three separate latent variables,
each with its own distributional form, and uses variational inference rather than HMM decoding.**

## 2. Latent variables
z_t = [m_t, φ_t, φ̇_t] — meter (time signature), beat phase (position within the bar), tempo
(rate of phase advance).

## 3. Generation model (factorization)
p_ψ(z_t | z_{t-1}) = p_ψ(m_t | m_{t-1}, φ_t, φ_{t-1}) · p_ψ(φ_t | φ_{t-1}, φ̇_{t-1}) · p_ψ(φ̇_t | φ̇_{t-1})

### Tempo  p(φ̇_t | φ̇_{t-1})  — random walk
History the paper traces:
- Whiteley 2006: integer grid, ±1 step with prob p_n/2, stay with 1−p_n.
- Whiteley 2007 (particle filter): **continuous Gaussian random walk, BOUNDED**:
  p(φ̇_t|φ̇_{t-1}) ∝ N(φ̇_{t-1}, σ²) · 1[φ̇_min ≤ φ̇_t ≤ φ̇_max].
- Krebs 2015 / madmom: exponential `exp(−λ|φ̇_t/φ̇_{t-1} − 1|)` for φ_{t-1} ∈ B (beat positions),
  else `1[φ̇_t = φ̇_{t-1}]`. **"Note that this formulation restricts tempo changes to occur ONLY at
  beat boundaries; between beats, tempo is held constant."**  ← the between-beats condition, stated
  as the property of THIS prior work.

**OUR MODEL adopts a continuous Log-Normal random walk (a deliberate departure from the
between-beats-constant condition, for differentiability):**
> log φ̇_t ~ N( log φ̇_{t-1}, σ^p_φ̇² )

φ̇_t is a continuous positive value = angular advance in radians/frame. Justifications: (i) positive
without truncation; (ii) Log-Normal KL = Gaussian KL in log-space (closed form); (iii) σ^p_φ̇ is
**learned**, adapting the permitted rate of tempo change. NOTE: this is a **per-frame** random walk
conditioned only on φ̇_{t-1}; it does NOT gate on beat boundaries and adds NO explicit [min,max] bound.

### Phase  p(φ_t | φ_{t-1}, φ̇_{t-1})  — von Mises
- Does NOT condition on meter (decoupled): phase advances on a fixed circle [0, 2π) regardless of
  meter; meter is inferred from the phase trajectory.
- **φ_t ∈ [0, 2π) is BAR phase: "0 is the start of the bar and values approaching 2π are the end."**
- First time phase is stochastic: φ_t ~ vM(φ_{t-1} + φ̇_{t-1}, κ^p_φ). Mean = deterministic
  bar-pointer advance; κ^p_φ learned (uncertainty around the prediction).

### Meter  p(m_t | m_{t-1}, φ_t, φ_{t-1})
- A meter switch is triggered when the bar pointer crosses a bar boundary.
- **Our model:** bar-boundary detection uses the predicted MEAN (not the noisy sample):
  μ^p_φ,t = φ_{t-1} + φ̇_{t-1}; a crossing occurs when **φ_{t-1} + φ̇_{t-1} ≥ 2π**.
- Meter prior = full K×K transition matrix from a network:
  π^p_t = f^m_ψ(m_{t-1}, φ_t, φ_{t-1}, h_{1:T}); p_ψ(m_t=j | m_{t-1}=i, φ_t, φ_{t-1}, h) = π^p_{ij,t}.
  Generalizes madmom (per-meter/time-varying retention; non-uniform off-diagonal) vs madmom's fixed ε.

## 4. ELBO (result of the full derivation)
log p_θ(b_{1:T} | h_{1:T}) ≥
  Σ_t E_{q(z_t)}[ log p_θ(b_t | z_t, h_{1:T}) ]
  − D_KL( q(m_1) ‖ p(m_1) ) − D_KL( q(φ_1) ‖ p(φ_1) ) − D_KL( q(φ̇_1) ‖ p(φ̇_1) )
  − Σ_{t≥2} E_{q(z_{t-1})}[ D_KL(q(m_t) ‖ p(m_t|m_{t-1},φ_t,φ_{t-1},h)) + D_KL(q(φ_t) ‖ p(φ_t|φ_{t-1},φ̇_{t-1},h)) + D_KL(q(φ̇_t) ‖ p(φ̇_t|φ̇_{t-1},h)) ]

## 5. Concrete distributions
- **5.1 Meter — Categorical.** prior t=1: f^init_ψ(h); t≥2: f^m_ψ(m_{t-1},φ_t,φ_{t-1},h). posterior
  t=1: f^m_φ(b,h); t≥2: f^m_φ(b, z_{t-1}, h). KL = Σ_k π^q_k log(π^q_k/π^p_k). Gumbel-Softmax sampling, τ annealed.
- **5.2 Phase — von Mises.** prior μ^p_φ,t = φ_{t-1}+φ̇_{t-1}, κ^p = f^φ_ψ(h). posterior [μ^q,κ^q] = f^φ_φ(b,[z_{t-1}],h).
  KL = log(I0(κ^p)/I0(κ^q)) + A(κ^q)[κ^q − κ^p cos(μ^q−μ^p)]. Sampling = Best–Fisher rejection + implicit reparam (Alg 2).
- **5.3 Tempo — Log-Normal.** prior μ^p_φ̇,t = log φ̇_{t-1}, σ^p = f^φ̇_ψ(h). posterior [μ^q,σ^q] = f^φ̇_φ(b,[z_{t-1}],h).
  KL = log(σ^p/σ^q) + (σ^q² + (μ^q−μ^p)²)/(2σ^p²) − 1/2. Sampling = log φ̇ = μ^q + σ^q·ε.
- **5.4 Decoder — Bernoulli.** b_t ∈ {0,1} binary **beat** indicator. b̂_t = σ(NN_θ(z_t, h_{1:T})). **Decoder reads h.**
- **5.5 Loss** L = −ELBO, single MC sample z_{1:T} ~ q_φ: Σ_t BCE(b_t,b̂_t) + Σ (KL_meter+KL_phase+KL_tempo)
  for t=1 (initial) and t≥2 (transition). β = 1. For t≥2 all prior/posterior params evaluated at the
  SAMPLED ẑ_{t-1} (π^p_t = f^m_ψ(m̂_{t-1},φ̂_t,φ̂_{t-1},h), μ^p_φ,t = φ̂_{t-1}+φ̂̇_{t-1}, μ^p_φ̇,t = log φ̂̇_{t-1}).

## 6. Algorithms
- **Algorithm 1 (SGVB training):** per sequence, t=1 init then t=2..T transitions: compute posterior
  params from (b, ẑ_{t-1}, h); compute prior params (μ^p_φ̇=log φ̂̇_{t-1}, σ^p=f^φ̇_ψ(h); μ^p_φ=φ̂_{t-1}+φ̂̇_{t-1},
  κ^p=f^φ_ψ(h)); sample m̂ (Gumbel), φ̂ (von Mises, Alg 2), log φ̂̇ (Gaussian reparam); meter prior
  π^p_t = f^m_ψ(m̂_{t-1}, φ̂_t, φ̂_{t-1}, h) computed AFTER φ̂_t is sampled; accumulate KL; then decode
  b̂_t = σ(NN_θ(ẑ_t, h)) and BCE; L = L_recon + L_KL; one Adam step over θ,φ,ψ.
- **Algorithm 2 (VonMisesSample):** forward = Best–Fisher rejection (τ=1+√(1+4κ²), ρ=(τ−√(2τ))/(2κ),
  r=(1+ρ²)/(2ρ), accept κ(r−f)+log f−log r ≥ log u2, z=±arccos(f), φ̂=μ+z). backward = implicit
  reparam: ∂φ̂/∂μ=1, ∂φ̂/∂κ = −(∂F(z|κ)/∂κ)/p(z|0,κ), ∂F/∂κ via forward-mode AD on the von Mises CDF.

## References (key)
[1] Best & Fisher 1979 (vM sampler) · [2] Böck et al. madmom 2016 · [3] Figurnov 2018 (implicit reparam)
· [4] Heydari BeatNet 2021 · [5] Krebs/Böck/Widmer 2015 (efficient state space) · [6] Krebs PF 2015
· [7] Srinivasamurthy PF 2015 · [8] Whiteley/Cemgil/Godsill 2007 (PF, bounded Gaussian RW) · [9] Whiteley 2006 (bar pointer).

---
## Implementation reconciliation (faithful/ vs this paper)
- **Tempo = per-frame ungated Log-Normal random walk** (§3, §5.3): `faithful/elbo.py` MATCHES. The
  "between beats tempo is constant" sentence in §3 describes Krebs 2015 [5]; the paper's OWN model
  departs from it. So the clamp/OU in `svt_core` are bandages on a real weakness of the paper's
  continuous-tempo choice, NOT restorations of a paper requirement. No explicit [min,max] bound in
  the paper (it cites [8]'s bounded RW but does not adopt the bound).
- **φ = BAR phase (wrap at 2π = bar boundary = downbeat)** (§5.2, §"Our model"). Our data pipeline
  builds phase that wraps per BEAT and reads beats off φ-wraps — a likely DEVIATION to confirm.
- Phase/meter/decoder/ELBO/Alg-1/Alg-2 forms otherwise MATCH the paper.
