"""Score a trained VAEDBN against the synthetic ground-truth phase, velocity and meter."""
import argparse
import itertools
import json
import math

import torch
import torch.nn.functional as F

import vae_dbn as M


def wrap(a):
    """Wrap angles to (-pi, pi]."""
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


@torch.no_grad()
def prior_paths(model, x, n):
    """Sample n prior paths p_psi(z | x) as predict_labels does; v is the deterministic advance omega."""
    h = model.backbone_feats(x)
    B, T = x.shape[:2]
    phis, vs, ms = [], [], []
    for _ in range(n):
        phi_p = x.new_zeros(B)
        m_p = x.new_full((B, model.R), 1.0 / model.R)
        pp, pv, pm = [], [], []
        for k in range(T):
            hp = model.pri(torch.cat([h[:, k], model.feats(phi_p, m_p)], -1))
            v = model.omega(hp)
            if k == 0:
                mu, kappa = model.phase_params(model.pri_phase0(hp))
            else:
                mu = torch.remainder(phi_p + v * model.Delta + model.delta(hp), 2 * math.pi)
                kappa = F.softplus(model.pri_kappa(hp)).squeeze(-1) + model.kmin
            phi = M.vm_sample(mu, kappa, torch.rand(B, device=x.device))
            if k == 0:
                m = torch.multinomial(F.softmax(model.pri_meter(hp), -1), 1).squeeze(-1)
            pp.append(phi)
            pv.append(v)
            pm.append(m)
            phi_p, m_p = phi, F.one_hot(m, model.R).float()
        phis.append(torch.stack(pp, 1))
        vs.append(torch.stack(pv, 1))
        ms.append(torch.stack(pm, 1))
    return torch.stack(phis), torch.stack(vs), torch.stack(ms)


@torch.no_grad()
def posterior_path(model, x, b):
    """Posterior-mode latent path from encode_path, shaped [1, B, T]; the posterior has no tempo."""
    path, _ = M.encode_path(model, x, b)
    phi, m = (torch.stack(z, 1)[None] for z in zip(*path, strict=True))
    return phi, None, m


def score(phi, v, m, truth, n_meter):
    """Latent recovery of [N, B, T] paths against the [B, T] ground truth."""
    phi_t, v_t, m_t = truth
    d = wrap(phi - phi_t)
    offset = torch.atan2(d.sin().mean(-1, keepdim=True), d.cos().mean(-1, keepdim=True))
    perms = itertools.permutations(range(n_meter))
    meter_acc = max((torch.tensor(p, device=m.device)[m] == m_t[:, None]).float().mean().item() for p in perms)
    advance = wrap(phi[..., 1:] - phi[..., :-1])
    result = {
        "phase_err": d.abs().mean().item(),
        "phase_err_rotated": wrap(d - offset).abs().mean().item(),
        "advance_mean": advance.mean().item(),
        "advance_backward": (advance < 0).float().mean().item(),
        "meter_acc_best_perm": meter_acc,
        "meter_switch_rate": (m[..., 1:] != m[..., :-1]).float().mean().item(),
    }
    if v is None:
        return result
    return result | {
        "v_mean": v.mean().item(),
        "v_true_mean": v_t.mean().item(),
        "v_mae": (v - v_t).abs().mean().item(),
        "v_negative": (v < 0).float().mean().item(),
        "dv_mean": (v[..., 1:] - v[..., :-1]).abs().mean().item(),
        "dv_true_mean": (v_t[:, 1:] - v_t[:, :-1]).abs().mean().item(),
    }


def main():
    """Load vae_dbn.pt, regenerate the training batch and write the latent audit."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="vae_dbn.pt")
    ap.add_argument("--out", default="latent_audit.json")
    ap.add_argument("--N", type=int, default=16)
    args = ap.parse_args()
    torch.manual_seed(0)
    x, b, (phi_t, v_t, m_t) = M.synth(return_latents=True)
    x, b = x.to(M.DEV), b.to(M.DEV)
    truth = (phi_t.to(M.DEV), v_t.to(M.DEV), m_t.to(M.DEV))
    model = M.load_model(args.ckpt)
    b_hat, _ = M.predict_labels(model, x, N=64)
    majority = torch.bincount(b.flatten(), minlength=3).max().item() / b.numel()
    result = {
        "label_acc_prior": (b_hat == b).float().mean().item(),
        "label_majority": majority,
        "prior": score(*prior_paths(model, x, args.N), truth, model.R),
        "posterior": score(*posterior_path(model, x, b), truth, model.R),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
