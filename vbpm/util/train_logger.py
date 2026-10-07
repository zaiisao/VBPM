"""train_logger.py -- a small, framework-light debugging logger for VAE-type training.

Writes two files:
  <name>.csv  : one row per logged step (open in Excel / pandas to plot curves)
  <name>.log  : human-readable, with [WARN] lines flagging common failure modes

It records whatever scalar dict you hand it, and provides helpers to build that dict:
  * gradient norms per module group  (catch vanishing / exploding / ZERO gradients)
  * tensor statistics mean/std/min/max (catch posterior collapse, saturation)
  * finite checks (catch NaN / Inf early)
and a HealthMonitor that turns the numbers into plain-language warnings:
  loss not decreasing, KL -> 0 (collapse), grad norm 0 (no signal / detached sample),
  NaN/Inf, von Mises kappa saturating, posterior sigma collapsed, bad update/param ratio.

Usage (see cvae_mnist.py / vae_dbn.py for full wiring):
    log = TrainLogger("run")                       # -> run.csv, run.log
    ...
    metrics = {"loss": loss.item(), "recon": r, "kl": k, **grad_norms(groups),
               **tensor_stats("mu_q", mu_q), **tensor_stats("sigma_q", sigma_q)}
    warns = health.check(metrics)
    log.log(step, metrics, warns)
    ...
    log.close()
"""

import math
import time
import csv
import torch


# ---------------------------------------------------------------- helpers
def global_grad_norm(parameters):
    """L2 norm of all gradients (0 if none)."""
    tot, seen = 0.0, False
    for p in parameters:
        if p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum())
            seen = True
    return math.sqrt(tot) if seen else 0.0


def grad_norms(groups):
    """groups: dict name -> module (or iterable of params). Returns {grad/<name>: norm}."""
    out = {}
    for name, m in groups.items():
        params = m.parameters() if hasattr(m, "parameters") else m
        out[f"grad/{name}"] = round(global_grad_norm(params), 6)
    return out


def tensor_stats(name, t):
    """mean/std/min/max of a tensor, as {name/mean: .., name/std: .., ...}."""
    t = t.detach().float()
    return {
        f"{name}/mean": round(float(t.mean()), 5),
        f"{name}/std": round(float(t.std()), 5),
        f"{name}/min": round(float(t.min()), 5),
        f"{name}/max": round(float(t.max()), 5),
    }


def finite_check(**tensors):
    """Return a list of warnings for any tensor containing NaN/Inf."""
    w = []
    for name, t in tensors.items():
        if t is None:
            continue
        t = t.detach()
        if torch.isnan(t).any():
            w.append(f"NaN in {name}")
        if torch.isinf(t).any():
            w.append(f"Inf in {name}")
    return w


def update_param_ratio(model, lr):
    """Rough |update|/|param| ratio = lr*||grad|| / ||param||, averaged over params.

    Healthy is ~1e-3; >>1e-2 means lr too high, <<1e-4 means too low / stuck.
    """
    gn = global_grad_norm(model.parameters())
    pn = math.sqrt(sum(float(p.detach().pow(2).sum()) for p in model.parameters()))
    return round(lr * gn / (pn + 1e-12), 6)


# ---------------------------------------------------------------- health monitor
class HealthMonitor:
    """Monitor training metrics for instability or stalled learning."""

    def __init__(
        self,
        loss_key="loss",
        window=20,
        kl_key="kl",
        kappa_key="kappa_q/mean",
        sigma_key="sigma_q/mean",
    ):
        self.loss_key, self.kl_key = loss_key, kl_key
        self.kappa_key, self.sigma_key = kappa_key, sigma_key
        self.window = window
        self.losses = []

    def check(self, m):
        """Return warnings for abnormal training metrics."""
        w = []
        # 1) loss trend
        if self.loss_key in m:
            self.losses.append(m[self.loss_key])
            if len(self.losses) > self.window:
                recent = self.losses[-self.window :]
                if recent[-1] >= recent[0] - 1e-6:
                    w.append(
                        f"loss not decreasing over last {self.window} steps "
                        f"({recent[0]:.3f} -> {recent[-1]:.3f})"
                    )
        # 2) posterior collapse
        if self.kl_key in m and m[self.kl_key] < 1e-2:
            w.append(
                f"{self.kl_key}={m[self.kl_key]:.2e} ~ 0 -> possible posterior collapse "
                f"(try KL annealing / free bits)"
            )
        # 3) zero / exploding gradients
        for k, v in m.items():
            if k.startswith("grad/"):
                if v == 0.0:
                    w.append(f"{k}=0 -> no gradient signal (detached sample? wrong branch?)")
                elif v > 1e4:
                    w.append(f"{k}={v:.1e} -> exploding gradient (clip / lower lr)")
        # 4) von Mises concentration sanity
        if self.kappa_key in m:
            kap = m[self.kappa_key]
            if kap > 60:
                w.append(f"{self.kappa_key}={kap:.1f} large -> check Bessel overflow (use i0e/i1e)")
        # 5) posterior sigma collapse
        if self.sigma_key in m and m[self.sigma_key] < 1e-3:
            w.append(f"{self.sigma_key}={m[self.sigma_key]:.2e} -> posterior variance collapsed")
        # 6) NaN/Inf already appended by finite_check at call site; nothing here
        return w


# ---------------------------------------------------------------- logger
class TrainLogger:
    """Write training metrics to CSV, logs, and optional console output."""

    def __init__(self, name, console=True):
        self.csv_path, self.log_path = f"{name}.csv", f"{name}.log"
        self.csvf = open(self.csv_path, "w", newline="")
        self.logf = open(self.log_path, "w")
        self.console = console
        self.t0 = time.time()
        self.keys = None
        self.writer = None

    def log(self, step, metrics, warns=None):
        """Record training metrics and optional warnings."""
        row = {"step": step, "t": round(time.time() - self.t0, 1)}
        row.update(metrics)
        if self.keys is None:  # fix column order on first call
            self.keys = list(row.keys())
            self.writer = csv.DictWriter(self.csvf, fieldnames=self.keys)
            self.writer.writeheader()
        self.writer.writerow({k: row.get(k, "") for k in self.keys})
        self.csvf.flush()
        # human-readable block
        pretty = "  ".join(f"{k}={row[k]}" for k in self.keys)
        self.logf.write(f"[step {step}] {pretty}\n")
        for wmsg in warns or []:
            self.logf.write(f"    [WARN] {wmsg}\n")
        self.logf.flush()
        if self.console:
            short = f"step {step:5d} | " + "  ".join(
                f"{k.split('/')[-1] if '/' in k else k}={row[k]}"
                for k in self.keys
                if k in ("step", "loss", "recon", "kl") or k.startswith("grad/total")
            )
            print(short)
            for wmsg in warns or []:
                print(f"   [WARN] {wmsg}")

    def close(self):
        """Close log files and return their paths."""
        self.csvf.close()
        self.logf.close()
        return self.csv_path, self.log_path
