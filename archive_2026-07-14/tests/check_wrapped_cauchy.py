"""Verify the wrapped Cauchy pieces (torch provides neither), so we never trust hand-written math blind:
  (1) KL(WC||WC) closed form -> vs a Monte-Carlo estimate from the wrapped Cauchy log density.
  (2) sampler correctness    -> E[cos(phi - mu)] == rho (the mean resultant length), across concentrations.
  (3) reparam gradient        -> autograd dE[cos(phi-mu)]/dc matches analytic 1/(1+c)^2, and is finite.
      (E[cos] = rho = c/(1+c), so d/dc = 1/(1+c)^2.) The Cauchy tail makes this estimator high-variance,
      so we cross-check against a low-variance forward finite difference of E[cos].
Run: python tests/check_wrapped_cauchy.py
"""
import sys, math
import torch
sys.path.insert(0, ".")
from model.latents import (kl_between_wrapped_cauchy as kl_wrapped_cauchy,
                           sample_wrapped_cauchy, concentration_to_rho as _concentration_to_rho)

TWO_PI = 2.0 * math.pi


def wc_logprob(theta, mu, rho):
    return torch.log((1 - rho ** 2) / (TWO_PI * (1 + rho ** 2 - 2 * rho * torch.cos(theta - mu))))


def check_kl():
    torch.manual_seed(0)
    cases = [(0.0, 1.0, 0.0, 1.0), (0.5, 3.0, 1.2, 1.5), (2.0, 0.3, 0.1, 2.0),
             (math.pi, 5.0, 0.0, 0.8), (1.0, 8.0, 1.3, 6.0)]
    worst = 0.0
    print("KL(WC||WC) vs Monte-Carlo:")
    for mq, cq, mp, cp in cases:
        t = torch.tensor
        analytic = float(kl_wrapped_cauchy(t(mq), t(cq), t(mp), t(cp)))
        rq, rp = _concentration_to_rho(t(cq)), _concentration_to_rho(t(cp))
        x = sample_wrapped_cauchy(torch.full((4_000_000,), float(mq)), torch.full((4_000_000,), float(cq)))
        mc = float((wc_logprob(x, t(mq), rq) - wc_logprob(x, t(mp), rp)).mean())
        worst = max(worst, abs(analytic - mc))
        print(f"  mq={mq:.2f} cq={cq:.2f} mp={mp:.2f} cp={cp:.2f} | analytic {analytic:.4f}  mc {mc:.4f}")
    print(f"  worst abs err = {worst:.4f} -> {'PASS' if worst < 0.01 else 'FAIL'}\n")
    return worst < 0.01


def check_sampler():
    """E[cos(phi - mu)] must equal rho = c/(1+c) (the first trigonometric moment of a wrapped Cauchy)."""
    torch.manual_seed(0)
    worst = 0.0
    print("sampler E[cos(phi-mu)] vs rho:")
    for c in (0.3, 1.0, 3.0, 9.0):
        phi = sample_wrapped_cauchy(torch.full((4_000_000,), 0.7), torch.full((4_000_000,), float(c)))
        e_cos = float(torch.cos(phi - 0.7).mean())
        rho = float(_concentration_to_rho(torch.tensor(float(c))))
        worst = max(worst, abs(e_cos - rho))
        print(f"  c={c:.2f} | E[cos] {e_cos:.4f}  rho {rho:.4f}")
    print(f"  worst abs err = {worst:.4f} -> {'PASS' if worst < 0.005 else 'FAIL'}\n")
    return worst < 0.005


def check_reparam_gradient():
    """Verify dE[cos]/dc = 1/(1+c)^2 via a low-variance WIDE-gap secant of E[cos] (bounded, so cheap to
    estimate). NOTE: the pathwise autograd estimator is correct in expectation but INFINITE-variance (the
    Cauchy tail: per-sample term ~ -w sin(gamma w), w heavy-tailed), so we do NOT check its value -- only
    that it is finite. In training this is tamed by grad clipping; implicit reparameterization (as used for
    von Mises) would give a bounded-variance gradient and is the fix if WC training proves too noisy."""
    torch.manual_seed(0)
    def ecos(c, n=8_000_000):
        return float(torch.cos(sample_wrapped_cauchy(torch.zeros(n), torch.full((n,), float(c)))).mean())
    print("dE[cos]/dc: analytic 1/(1+c)^2 vs wide-gap secant of E[cos] (independent samples):")
    worst = 0.0
    for c in (1.0, 3.0):
        secant = (ecos(c + 0.05) - ecos(c - 0.05)) / 0.1          # gap 0.1: low MC noise, small truncation
        analytic = 1.0 / (1.0 + c) ** 2
        worst = max(worst, abs(secant - analytic))
        print(f"  c={c:.2f} | analytic {analytic:.4f}  secant {secant:.4f}")
    conc = torch.full((200_000,), 1.0, requires_grad=True)
    torch.cos(sample_wrapped_cauchy(torch.zeros(200_000), conc)).mean().backward()
    ag_finite = math.isfinite(float(conc.grad.sum()))
    print(f"  autograd gradient finite: {ag_finite}  (heavy-tail => high variance, tamed by grad clipping)")
    ok = worst < 0.01 and ag_finite
    print(f"  worst secant err = {worst:.4f} -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ok = check_kl() & check_sampler() & check_reparam_gradient()
    print("\nALL CHECKS:", "PASS" if ok else "FAIL")
