"""Verify the two von Mises pieces torch does not provide, so we never trust hand-written math blind:
  (1) KL(vM||vM) closed form  -> vs a Monte-Carlo estimate from torch's VonMises.log_prob.
  (2) implicit-reparam gradient through the concentration -> vs the analytic dE[cos(phi-mu)]/dkappa = A'(kappa),
      where A(k)=I1/I0 and A'(k) = 1 - A(k)/k - A(k)^2 (the only thing that depends on kappa here is E=A(k)).
Run: python tests/check_von_mises.py
"""
import sys, math
import torch
from torch.distributions import VonMises
sys.path.insert(0, ".")
from model.latents import kl_von_mises, sample_von_mises


def check_kl():
    torch.manual_seed(0)
    cases = [(0.0, 1.0, 0.0, 1.0), (0.5, 3.0, 1.2, 1.5), (2.0, 0.3, 0.1, 2.0),
             (math.pi, 5.0, 0.0, 0.8), (1.0, 8.0, 1.3, 6.0)]
    worst = 0.0
    print("KL(vM||vM) vs Monte-Carlo:")
    for mq, kq, mp, kp in cases:
        t = torch.tensor
        analytic = float(kl_von_mises(t(mq), t(kq), t(mp), t(kp)))
        q, p = VonMises(t(mq), t(kq)), VonMises(t(mp), t(kp))
        x = q.sample((2_000_000,))
        mc = float((q.log_prob(x) - p.log_prob(x)).mean())
        worst = max(worst, abs(analytic - mc))
        print(f"  mq={mq:.2f} kq={kq:.2f} mp={mp:.2f} kp={kp:.2f} | analytic {analytic:.4f}  mc {mc:.4f}")
    print(f"  worst abs err = {worst:.4f} -> {'PASS' if worst < 0.01 else 'FAIL'}\n")
    return worst < 0.01


def check_reparam_gradient():
    """Reparam grad of E[cos(phi-mu)] w.r.t kappa should match A'(kappa)."""
    torch.manual_seed(0)
    num = 400_000
    worst = 0.0
    print("implicit-reparam dE[cos(phi-mu)]/dkappa vs analytic A'(kappa):")
    for k in (0.5, 1.0, 3.0, 6.0):
        kappa = torch.full((num,), k, requires_grad=True)
        phi = sample_von_mises(torch.zeros(num), kappa)          # mu = 0
        (torch.cos(phi).mean()).backward()
        reparam = float(kappa.grad.sum())                        # unbiased estimator of dE/dkappa
        A = float(torch.special.i1e(torch.tensor(k)) / torch.special.i0e(torch.tensor(k)))
        analytic = 1.0 - A / k - A * A                           # A'(k)
        worst = max(worst, abs(reparam - analytic))
        print(f"  kappa={k:.2f} | reparam {reparam:.4f}  analytic {analytic:.4f}")
    print(f"  worst abs err = {worst:.4f} -> {'PASS' if worst < 0.02 else 'FAIL'}")
    return worst < 0.02


if __name__ == "__main__":
    ok = check_kl() & check_reparam_gradient()
    print("\nALL CHECKS:", "PASS" if ok else "FAIL")
