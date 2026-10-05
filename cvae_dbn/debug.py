"""Per-term gradient routing used by this experiment's existing preflight."""
import math

import torch


def check(name, condition):
    print('CHECK', name, bool(condition), flush=True)
    if not condition:
        raise RuntimeError(name)


def gradient_routes(model, data):
    groups = dict(post_phase=tuple(model.post_phase.parameters()), post_velocity=tuple(model.post_vel.parameters()),
                  post_meter=tuple(model.post_meter.parameters()), prior_velocity=tuple(model.pri_vel.parameters()),
                  prior_kappa=tuple(model.pri_kappa.parameters()), emission=tuple(model.emit.parameters()))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(333)
        h, c = model.backbone_feats(data['x']), model.context(data['b'], data['x'])
        eq, kl = model.rollout(data['x'], h, c, data['b'], tau=.5, use_post=True)
        ep, _ = model.rollout(data['x'], h, b=data['b'], tau=.5, use_post=False)
        terms = dict(posterior_recon=-eq.mean(), KL=kl.mean(), prior_recon=-ep.mean())
        result = {}
        for name, parameters in groups.items():
            result[name] = {}
            for term, loss in terms.items():
                gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                result[name][term] = math.sqrt(sum(float(g.detach().square().sum()) for g in gradients if g is not None))
        check('Per-term head gradients finite', all(math.isfinite(value)
              for values in result.values() for value in values.values()))
        return result
