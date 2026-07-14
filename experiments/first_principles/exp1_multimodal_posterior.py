"""EXPERIMENT 1 — The bar-pointer posterior is multimodal, and NO unimodal q can represent it.

Claim: beat evidence depends on bar phase phi only through cos(M*phi) (M = beats/bar), so the
posterior p(phi | evidence) has M equivalent modes (gauge ambiguity). A unimodal q must either
  (a) commit to ONE mode   [argmin KL(q||p), the VAE's mode-seeking direction], or
  (b) spread to uniform    [argmin KL(p||q), moment matching],
losing (M-1)/M of the truth or all phase information respectively. Exact grid inference keeps all M.

VENDOR CODE (nothing hand-rolled):
  q family:      scipy.stats.vonmises.pdf
  KL terms:      scipy.special.rel_entr   (rel_entr(a,b) = a*log(a/b), elementwise KL integrand)
  mode finding:  scipy.signal.find_peaks

Run: python exp1_multimodal_posterior.py     (deterministic, no sampling)
"""
import numpy as np
from scipy.stats import vonmises
from scipy.special import rel_entr
from scipy.signal import find_peaks

M = 4                      # beats per bar (4/4)
KAPPA_EVIDENCE = 5.0       # sharpness of the beat evidence
N_GRID = 3600

phi = np.linspace(0.0, 2 * np.pi, N_GRID, endpoint=False)

# true posterior: evidence says "we are ON a beat" -> von Mises in the BEAT phase M*phi
p = vonmises.pdf(M * phi, kappa=KAPPA_EVIDENCE)        # scipy: exp(k cos(x)) / (2 pi I0(k))
p = p / p.sum()

# mode structure (find_peaks on the circularly-padded posterior)
pk, _ = find_peaks(np.concatenate([p, p[:10]]))
pk = np.unique(pk % N_GRID)
print(f"true posterior: {len(pk)} modes at phi = {[round(float(phi[i]), 3) for i in pk]} "
      f"(spacing 2pi/{M} — the gauge ambiguity); mass per mode = {1/M:.2f}\n")

def q_of(mu, kappa):
    q = vonmises.pdf(phi, kappa=max(kappa, 1e-8), loc=mu)
    return q / q.sum()

mus = phi[::10]
kappas = np.concatenate([[1e-3], np.geomspace(0.01, 50, 60)])
best = {"q||p": (None, np.inf), "p||q": (None, np.inf)}
for mu in mus:
    for k in kappas:
        q = q_of(mu, k)
        kl_qp = rel_entr(q, p).sum()          # KL(q||p): the VAE's objective direction
        kl_pq = rel_entr(p, q).sum()          # KL(p||q): moment matching
        if kl_qp < best["q||p"][1]: best["q||p"] = ((mu, k), kl_qp)
        if kl_pq < best["p||q"][1]: best["p||q"] = ((mu, k), kl_pq)

(mu_a, k_a), _ = best["q||p"]
in_mode = np.cos(phi - mu_a) > np.cos(np.pi / M)          # q's own mode region (width 2pi/M)
print(f"(a) argmin KL(q||p): mu={mu_a:.3f}, kappa={k_a:.2f}")
print(f"    q commits to ONE mode; true mass inside q's region: {p[in_mode].sum():.2f} "
      f"(ignores {1 - p[in_mode].sum():.2f}) -> confidently WRONG about the downbeat w.p. {(M-1)/M:.2f}\n")

(mu_b, k_b), _ = best["p||q"]
print(f"(b) argmin KL(p||q): mu={mu_b:.3f}, kappa={k_b:.4f} -> (near-)uniform: phase info destroyed\n")

print(f"exact grid posterior keeps all {len(pk)} modes "
      f"(entropy {-(p * np.log(p + 1e-300)).sum():.2f} vs uniform {np.log(N_GRID):.2f})")
print("VERDICT: any unimodal q is either confidently-wrong (a) or uninformative (b);")
print("         exact inference is the only representation keeping the full gauge structure.")
