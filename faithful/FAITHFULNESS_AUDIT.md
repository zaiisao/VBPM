# Faithfulness Audit — `faithful/` strict-ELBO bar-pointer VAE

**Auditor:** synthesis (static auditors + adversarial verification + empirical numerical tests, independently re-verified against source).
**Date:** 2026-06-24
**Subject:** `faithful/{distributions,model,elbo,data,train}.py`, `faithful/__init__.py`, `faithful/README.md`
**References:** `notebooks/build_elbo_notebook.py` (the verified reference) and `faithful/README.md` (the faithfulness contract).
**Note on the paper:** the original `ELBO_for_DBN.pdf` is **not on disk**. Closed-form formulas were verified against (a) the reference notebook line-by-line, (b) standard textbook expressions, and (c) numerical properties (q==p ⇒ KL=0). Items that can only be settled by the PDF are flagged UNCERTAIN in §5.

> **CORRECTION (2026-06-26):** this audit's verdict on the von Mises sampler was **wrong**.
> `best_fisher_rejection` had an incorrect acceptance test, so sampled phase had a near-constant
> concentration (E[cos] ≈ 0.8) for *every* κ instead of the correct `A(κ)`. Empirical test #2 below
> only checked the sample **mean direction** (correct) and **never the spread**, so the bug was
> missed. The sampler has since been fixed and verified exact against `scipy.stats.vonmises`
> (κ=0.2…20). See `STRICT_ELBO_RESULTS.md` §0 for details and the corrected re-run. The other audit
> findings (closed-form KLs, ELBO assembly, gradient flow, bandage absence) stand.

---

## 1. Overall verdict

### **FAITHFUL-WITH-CAVEATS**

The *code* is faithful to the strict-ELBO specification and the reference notebook. The objective is exactly `L = Σ_t BCE(b_t, σ(decode(z_t, h))) + Σ_t [KL_meter + KL_phase + KL_tempo]` with β=1, a single MC sample, three latents, deterministic bar-pointer prior means with no audio correction, closed-form KLs, the Best–Fisher von Mises sampler with implicit reparameterisation, and **no bandages** (no free-bits, KL annealing, latent supervision, pos_weight, tempo clamps, scheduled sampling, extra latents, delta-VAE or DVBF). All 6 empirical tests pass.

The single caveat is a **documentation defect**: the "Faithfulness contract" in `faithful/__init__.py:12` states the *opposite* of the true (and faithful) decoder behavior. The runtime is correct; the contract text is wrong and internally self-contradictory. This is a major *documentation* deviation, not a behavioral one. Hence FAITHFUL-WITH-CAVEATS rather than fully FAITHFUL.

---

## 2. Per-dimension results

| Dimension | Faithful? | One-line note |
|---|:---:|---|
| Closed-form KL correctness (categorical / von Mises / log-normal) | ✅ | Char-identical to notebook L227–239; log_i0/A_kappa use i0e/i1e; q==p ⇒ KL=0 numerically. |
| von Mises Best–Fisher sampler + implicit reparam (Alg. 2) | ✅ | `distributions.py:62–117` is a line-by-line port of notebook L150–199; true implicit grad, not stop-grad. |
| Generative + inference structure (prior means, posterior conditioning, meter-prior ordering) | ✅ | Phase mean = φ_{t-1}+φ̇_{t-1}; tempo mean = log φ̇_{t-1}; meter prior uses **sampled** φ_t; posterior reads sampled ẑ_{t-1}. |
| ELBO assembly + decoder | ✅ | `elbo.py:94 loss=(recon+L_kl).mean()`; β=1; BCE has no pos_weight; decoder reads h by default. |
| Bandage absence (22 items) | ✅ | All 22 absent in code; verified by grep + read. |
| End-to-end from random weights (fixed log-mel, no pretrained/frozen, only VAE trained) | ⚠️ | Code path is faithful (LogMel not in optimizer, random init, no `torch.load`); **but** `__init__.py:12` contract bullet is inverted. |

Legend: ✅ verified faithful · ⚠️ faithful runtime with a documentation defect.

---

## 3. Upheld deviations

One deviation was raised, independently re-verified, and **upheld** (severity: **major**, category: documentation).

### D1 — `__init__.py:12` states the faithfulness contract backwards

- **Location:** `faithful/__init__.py:12`
- **Text (verbatim):** `* latent-only decoder  p_theta(b_t | z_t)  -- the decoder never reads the audio h`
- **Why it is wrong:** the surrounding docstring frames the bullet list as the *"Faithfulness contract"* (`__init__.py:9`) and closes with *"Anything that deviates from the above is, by definition, not faithful"* (`__init__.py:19–20`). Line 12 therefore **elevates the documented DEVIATION to a requirement** and would brand the actually-faithful default as a violation.
- **Contradicted by (the correct, faithful behavior):**
  - `model.py:30` — `latent_only: bool = False` (default reads h)
  - `model.py:62` — `dec_in = z_feat_dim if latent_only else z_feat_dim + hidden`
  - `model.py:103` — `x = z_feat if self.latent_only else torch.cat([z_feat, prior_ctx_t], dim=-1)`
  - `model.py:9–11` — explicitly tags `latent_only` as *"DOCUMENTED DEVIATION (the paper's §5.4 decoder is p_theta(b_t | z_t, h), i.e. it DOES read h)"*
  - `train.py:67` — prints `decoder reads h: True` by default
  - `README.md:40` — *"Bernoulli decoder σ(NN_θ(z_t, h)) (§5.4) | model.decode — **reads h**"*
- **Reference confirms decoder reads h:** `notebooks/build_elbo_notebook.py:345–346` — `def decode(self, z_feat, prior_ctx_t): return self.decoder(torch.cat([z_feat, prior_ctx_t]))...` (no `latent_only` branch; **always** concatenates the encoded-h context); §5.4 documented as `p_θ(b_t | z_t, h)` at notebook L56/L265/L798.
- **Impact:** documentation only. No effect on the trained model or the empirical results — the default code path is faithful. But because `__init__.py` is described as *"the reference the rest of the project is measured against"* (`__init__.py:20`), a wrong contract line is consequential.

**Recommended fix** — rewrite `__init__.py:12` to match `README.md:40` / `model.py:10`, e.g.:

```text
  * decoder p_theta(b_t | z_t, h) reads the audio h (paper §5.4); the optional
    --latent_only flag drops h from the decoder as a DOCUMENTED DEVIATION.
```

No other upheld deviations. All "absent bandage" claims (22 items) and all structural/objective claims were verified true.

---

## 4. Empirical test results

All tests run on RANDOM-INIT models with synthetic `h`/`b`; they verify mechanics/faithfulness of the objective and wiring, not training outcome. **All 6 PASS.**

| # | Test | Result | Key numbers |
|:--:|---|:--:|---|
| 1 | KL = 0 when q==p (all three KLs) within 1e-4 | ✅ | categorical/von-Mises/log-normal all 0.000e+00; von Mises also 0 across κ∈{0.1,1,5,20}. |
| 2 | von Mises sampler: circular mean ≈ μ; d(sample)/dκ finite & non-zero (implicit reparam, not detached) | ✅ | N=3000, μ=1.0,κ=4.0 → emp. mean 1.0163 (\|err\|<0.1); dφ/dκ=−2.04e-1 (single); Σ dcosφ/dκ=1.22e1; dφ/dμ flows (=1/elem). |
| 3 | ELBO has no hidden terms: loss == recon + KL_m + KL_φ + KL_τ (β=1) | ✅ | loss 45.772629 vs recomposed 45.772632 (\|Δ\|=3.6e-6); recon == independently-recomputed plain BCE (\|Δ\|=1.9e-6); no pos_weight. |
| 4 | forward+backward: finite loss, non-zero grad on ALL param groups (no dead sub-network) | ✅ | loss 36.0940; every group grad-norm >0 & finite (post_gru, prior_gru, heads, meter_prior, decoder, z0). |
| 5 | free_run phase_mu advances and wraps 2π→0 ≥ once over T=300 | ✅ | 48 wraps/300 frames; constant-increment sawtooth matches mean chain to circular max-err 2.89e-5; forced-tempo control wraps 4×. |
| 6 | latent_only decode is h-invariant; default decode is h-dependent | ✅ | latent_only=True: max\|Δ\|=0.000e+00 over two h-contexts; default: max\|Δ\|=0.3633. |

Empirical conclusion: the strict ELBO is a clean transcription of `run_algorithm_1`; the von Mises sampler is a true implicit-reparam sampler (gradients flow through κ and μ); the bar-pointer dynamics are correctly wired; and test 6 confirms the documented `latent_only` behavior — which is exactly what makes the inverted `__init__.py:12` contract line a documentation bug rather than a behavioral one.

---

## 5. Items checkable only against the (missing) PDF — UNCERTAIN

These are flagged honestly: the notebook is asserted to be the verified reference and the code matches it line-by-line, but the original §5.1–5.3 derivations could not be independently confirmed from the PDF.

1. **Exact §5.1–5.3 KL forms.** `kl_categorical`, `kl_von_mises`, `kl_log_normal` (`distributions.py:131–145`) are character-identical to notebook L227–239 and match the canonical literature forms (categorical cross-entropy; von Mises KL via `log I0(κ_p)−log I0(κ_q)+A(κ_q)(κ_q−κ_p cos(μ_q−μ_p))`; log-normal = Gaussian KL in log-space). They satisfy q==p ⇒ KL=0 and Gibbs ≥0. Confidence **high** but **not paper-confirmed**.
2. **KL magnitudes (not just zero-at-equality).** Von Mises KL was Monte-Carlo-cross-checked by an upstream auditor (4 cases within 6× SE of 500k-sample MC); log-normal KL was not independently MC-checked here. Both are the standard forms → low risk, **not paper-confirmed**.
3. **Prior structure details (§5.2–5.3): which terms read h.** Code reads h for phase κ, tempo σ, and the meter transition matrix only — never for the prior MEANS. This matches the notebook and the README's reading of §5; the precise §5.2/§5.3 statement that *only* concentrations/scales (not means) may depend on h could not be verified from the PDF. Confidence **high**, **not paper-confirmed**.
4. **§5.4 decoder signature `p_θ(b_t | z_t, h)`.** Verified against the notebook (L56/265/345–346/798) and README:40; the faithful default matches. The PDF itself was not consulted. (This is also the subject of deviation D1.)

None of these UNCERTAIN items showed any discrepancy against the available reference; they remain UNCERTAIN purely because the authoritative PDF is absent.

---

## 6. Bottom line

The `faithful/` package is a faithful, bandage-free implementation of the strict-ELBO bar-pointer VAE: correct closed-form KLs, a true implicit-reparam von Mises sampler, the exact Algorithm-1 rollout, deterministic prior means with no audio correction, and end-to-end training from random weights on a fixed (non-learned) log-mel front-end. The only defect is a single inverted line in the contract docstring (`__init__.py:12`), which misstates the decoder as latent-only when both the code and the paper/reference have it read h. Fix that one line and the package is fully FAITHFUL. Four formula/structure items remain UNCERTAIN only because the original PDF is unavailable; all match the verified notebook.
