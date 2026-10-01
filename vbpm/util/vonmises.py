"""von Mises: closed-form KL, and two samplers of which only one is exact.

`sample_vonmises_icdf` is the mainline draw and the one the phase note derives:
phi = S_kappa(eps) with S the inverse CDF, whose kappa-derivative comes from
differentiating F_kappa(S_kappa(eps)) = eps rather than from any closed form.
Its gradient is unbiased at every concentration.

`sample_vonmises` is Best-Fisher rejection with a pathwise gradient. It is
cheaper per draw but its kappa-gradient is biased -- measured -48% at kappa = 2,
negligible above kappa ~ 50 -- so it is only safe where the concentration is
known to stay large. q's concentration is LEARNED and floored at KAPPA_Q_MIN =
0.01, so that condition cannot be assumed during training. It survives here for
the variants that already used it.
"""
from __future__ import annotations

import math

import torch

SMALL_KAPPA = 1e-4   # below this the von Mises is uniform to float64 precision, and the
                     # Best-Fisher rho = (tau - sqrt(2 tau)) / (2 kappa) is 0/0


def log_i0(kappa: torch.Tensor) -> torch.Tensor:
    """Return log I0(kappa), stable for large kappa via the exponentially scaled Bessel."""
    return torch.log(torch.special.i0e(kappa)) + kappa


ASYMPTOTIC_KAPPA = 50.0   # above this A'(kappa) is taken from the series: the exact form
                          # 1 - A/k - A^2 cancels two float32 quantities agreeing to ~5e-6
                          # at kappa 1e5, which flips the sign of every phase-kappa gradient


class _MeanResultant(torch.autograd.Function):
    """A(kappa) with a derivative that survives float32 at the operating concentration."""

    @staticmethod
    def forward(ctx, kappa):
        a = torch.special.i1e(kappa) / torch.special.i0e(kappa)
        ctx.save_for_backward(kappa, a)
        return a

    @staticmethod
    def backward(ctx, grad_out):
        kappa, a = ctx.saved_tensors
        k = kappa.double().clamp_min(1e-12)
        ad = a.double()
        exact = 1.0 - ad / k - ad * ad
        asym = 0.5 / k ** 2 + 0.25 / k ** 3 + 0.375 / k ** 4
        deriv = torch.where(k > ASYMPTOTIC_KAPPA, asym, exact)
        return grad_out * deriv.to(grad_out.dtype)


def mean_resultant(kappa: torch.Tensor) -> torch.Tensor:
    """A(kappa) = I1(kappa) / I0(kappa) = E[cos x] under vM(0, kappa)."""
    return _MeanResultant.apply(kappa)


def kl_vonmises(mu1, kappa1, mu2, kappa2):
    """KL( vM(mu1, kappa1) || vM(mu2, kappa2) ), closed form, elementwise."""
    return (log_i0(kappa2) - log_i0(kappa1)
            + mean_resultant(kappa1) * (kappa1 - kappa2 * torch.cos(mu1 - mu2)))


def _best_fisher_rho(kappa: torch.Tensor) -> torch.Tensor:
    """Best-Fisher's rho, differentiable in kappa (kappa clamped away from 0)."""
    safe = kappa.clamp_min(SMALL_KAPPA)
    tau = 1.0 + torch.sqrt(1.0 + 4.0 * safe * safe)
    return (tau - torch.sqrt(2.0 * tau)) / (2.0 * safe)


def sample_vonmises(kappa: torch.Tensor, max_rounds: int = 64) -> torch.Tensor:
    """Reparameterised sample from vM(0, kappa), elementwise, shape of ``kappa``."""
    work = kappa.double()
    shape = work.shape
    with torch.no_grad():
        rho = _best_fisher_rho(work)
        r = (1.0 + rho * rho) / (2.0 * rho)
        accepted = torch.zeros(shape, dtype=torch.bool, device=work.device)
        u1 = torch.zeros(shape, dtype=torch.float64, device=work.device)
        u3 = torch.zeros(shape, dtype=torch.float64, device=work.device)

        for _ in range(max_rounds):
            if bool(accepted.all()):
                break
            p1 = torch.rand(shape, dtype=torch.float64, device=work.device)
            p2 = torch.rand(shape, dtype=torch.float64, device=work.device)
            p3 = torch.rand(shape, dtype=torch.float64, device=work.device)
            z = torch.cos(math.pi * p1)
            f = (1.0 + r * z) / (r + z)
            c = work * (r - f)
            ok = ((c * (2.0 - c) - p2) > 0) | ((torch.log(c / p2.clamp_min(1e-300))
                                                + 1.0 - c) >= 0)
            take = ok & (~accepted)
            u1 = torch.where(take, p1, u1)
            u3 = torch.where(take, p3, u3)
            accepted = accepted | ok

        # kappa below SMALL_KAPPA is uniform: accept any draw, the f-path is degenerate
        tiny = work < SMALL_KAPPA

    rho = _best_fisher_rho(work)
    r = (1.0 + rho * rho) / (2.0 * rho)

    # u1 = 0 exactly (torch.rand includes it) drives 1 - f to 0; clamping costs 1e-7 of
    # the support and keeps every derivative below finite
    u1 = u1.clamp(1e-7, 1.0 - 1e-7)
    z = torch.cos(math.pi * u1)

    one_minus_f = ((r - 1.0) * (1.0 - z) / (r + z)).clamp(0.0, 2.0)
    sign = torch.where(u3 > 0.5, 1.0, -1.0)
    angle = 2.0 * sign * torch.asin(
        torch.sqrt(one_minus_f / 2.0 + 1e-24).clamp(max=1.0))

    uniform = (2.0 * u1 - 1.0) * math.pi
    angle = torch.where(tiny | (~accepted), uniform, angle)
    return angle.to(kappa.dtype)


TWO_PI = 2.0 * math.pi
VM_NODES = 512


def _vm_tables(kappa, nodes: int = VM_NODES):
    """(grid, cdf, dcdf_dkappa) for vM(0, kappa) on [-pi, pi], one row per element.

    The trapezoid rule on a periodic density converges geometrically, so a fixed grid
    replaces the root-find's quadrature entirely: F_kappa and its kappa-derivative are
    cumulative sums over the same nodes, and the inverse is a searchsorted away.
    """
    unit = torch.linspace(-1.0, 1.0, nodes, device=kappa.device, dtype=kappa.dtype)
    half = (8.0 / kappa.clamp(min=1e-3).sqrt()).clamp(max=math.pi)
    grid = half[..., None] * unit
    k = kappa[..., None]
    dens = torch.exp(k * (torch.cos(grid) - 1.0)) / (TWO_PI * torch.special.i0e(k))
    step = (2.0 * half / (nodes - 1))[..., None]
    trap = 0.5 * (dens[..., 1:] + dens[..., :-1]) * step
    cdf = torch.cat([torch.zeros_like(trap[..., :1]), trap.cumsum(-1)], dim=-1)
    cdf = cdf / cdf[..., -1:].clamp(min=1e-30)
    a = _MeanResultant.apply(kappa)[..., None]
    weight = (torch.cos(grid) - a) * dens
    dtrap = 0.5 * (weight[..., 1:] + weight[..., :-1]) * step
    dcdf = torch.cat([torch.zeros_like(dtrap[..., :1]), dtrap.cumsum(-1)], dim=-1)
    return grid, cdf, dcdf


class _VonMisesInvCDF(torch.autograd.Function):
    """phi = S_kappa(eps), the inverse von Mises CDF, differentiable in kappa.

    The inverse has no closed form, so the kappa-derivative comes from differentiating
    the identity F_kappa(S_kappa(eps)) = eps, which gives dS/dkappa = -dF/dkappa
    evaluated at phi, divided by the density there. Nothing needs the inverse in closed
    form: the value is read off the tabulated CDF and the gradient off the same table.
    """

    @staticmethod
    def forward(ctx, kappa, eps):
        grid, cdf, dcdf = _vm_tables(kappa)
        idx = torch.searchsorted(cdf.contiguous(), eps[..., None].contiguous())
        idx = idx.clamp(1, cdf.shape[-1] - 1)
        lo, hi = idx - 1, idx
        c_lo, c_hi = cdf.gather(-1, lo), cdf.gather(-1, hi)
        frac = ((eps[..., None] - c_lo) / (c_hi - c_lo).clamp(min=1e-30)).clamp(0.0, 1.0)
        g_lo, g_hi = grid.gather(-1, lo), grid.gather(-1, hi)
        phi = (g_lo + frac * (g_hi - g_lo))[..., 0]
        d_lo, d_hi = dcdf.gather(-1, lo), dcdf.gather(-1, hi)
        d_at = (d_lo + frac * (d_hi - d_lo))[..., 0]
        dens = torch.exp(kappa * (torch.cos(phi) - 1.0)) / (TWO_PI * torch.special.i0e(kappa))
        ctx.save_for_backward(-d_at / dens.clamp(min=1e-30))
        return phi

    @staticmethod
    def backward(ctx, grad_out):
        (dphi_dkappa,) = ctx.saved_tensors
        return grad_out * dphi_dkappa, None


def sample_vonmises_icdf(kappa: torch.Tensor) -> torch.Tensor:
    """Reparameterised vM(0, kappa) draw whose kappa-gradient is the implicit one."""
    eps = torch.rand_like(kappa).clamp(1e-6, 1.0 - 1e-6)
    return _VonMisesInvCDF.apply(kappa, eps)
