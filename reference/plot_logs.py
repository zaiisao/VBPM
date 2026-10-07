"""
plot_logs.py -- draw training-health curves from a train_logger CSV.

Works for both cvae_mnist_run.csv and vae_dbn_run.csv: it picks panels based on which
columns are present. Produces <name>_curves.png with up to six panels:
  (A) losses       : loss / recon / GSNN recon, with KL on a twin axis
  (B) per-factor KL : kl_phase / kl_vel / kl_meter  (VAE-DBN), else total KL
  (C) grad norms   : every grad/<group> column, log scale  (dead/exploding modules jump out)
  (D) posterior    : kappa_q or sigma_q mean with min-max band; prior sigma for CVAE
  (E) collapse/entropy : active_units (CVAE) or meter_entropy (VAE-DBN)
  (F) upd/param    : update/param ratio, log scale, with the healthy ~1e-3 line

    python plot_logs.py vae_dbn_run.csv
    python plot_logs.py cvae_mnist_run.csv
"""
import sys, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    rows = list(csv.DictReader(open(path)))
    cols = {k: [] for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            try: cols[k].append(float(v))
            except (ValueError, TypeError): cols[k].append(float("nan"))
    return cols


def band(ax, step, c, base, label, color):
    """Plot mean with a min-max shaded band if those columns exist."""
    if f"{base}/mean" not in c: return
    ax.plot(step, c[f"{base}/mean"], color=color, label=f"{label} mean")
    if f"{base}/min" in c and f"{base}/max" in c:
        ax.fill_between(step, c[f"{base}/min"], c[f"{base}/max"], color=color, alpha=0.15)


def plot(path):
    c = load(path); step = c["step"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    # (A) losses + KL twin
    a = ax[0, 0]
    for k, col in [("loss", "k"), ("recon", "tab:blue"), ("gsnn_recon", "tab:green")]:
        if k in c: a.plot(step, c[k], col, label=k)
    a.set_title("(A) losses"); a.set_xlabel("step"); a.legend(loc="upper right", fontsize=8)
    if "kl" in c:
        a2 = a.twinx(); a2.plot(step, c["kl"], "tab:red", ls="--", lw=1, label="KL")
        a2.set_ylabel("KL", color="tab:red"); a2.tick_params(axis="y", labelcolor="tab:red")

    # (B) per-factor KL
    b = ax[0, 1]; got = False
    for k, col in [("kl_phase", "tab:purple"), ("kl_vel", "tab:orange"), ("kl_meter", "tab:brown")]:
        if k in c: b.plot(step, c[k], col, label=k); got = True
    if not got and "kl" in c: b.plot(step, c["kl"], "tab:red", label="kl")
    b.set_title("(B) per-factor KL"); b.set_xlabel("step"); b.legend(fontsize=8)

    # (C) gradient norms
    g = ax[0, 2]
    for k in [k for k in c if k.startswith("grad/")]:
        g.plot(step, c[k], label=k.split("/")[1], lw=1)
    g.set_yscale("log"); g.set_title("(C) grad norms (log)"); g.set_xlabel("step"); g.legend(fontsize=7)

    # (D) posterior parameter stats
    d = ax[1, 0]
    if "kappa_q/mean" in c: band(d, step, c, "kappa_q", "kappa_q", "tab:purple")
    band(d, step, c, "sigma_q", "sigma_q", "tab:blue")
    if "sigma_p/mean" in c: d.plot(step, c["sigma_p/mean"], "tab:gray", ls=":", label="sigma_p mean")
    d.set_title("(D) posterior params (mean +/- range)"); d.set_xlabel("step"); d.legend(fontsize=8)

    # (E) collapse / entropy
    e = ax[1, 1]
    if "active_units" in c:
        e.plot(step, c["active_units"], "tab:green", label="active units")
        if "z_dim" in c: e.axhline(c["z_dim"][0], color="gray", ls=":", label="z_dim")
        e.set_title("(E) active latent units (collapse watch)")
    elif "meter_entropy/mean" in c:
        band(e, step, c, "meter_entropy", "meter_entropy", "tab:brown")
        e.set_title("(E) meter posterior entropy")
    e.set_xlabel("step"); e.legend(fontsize=8)

    # (F) update/param ratio
    f = ax[1, 2]
    key = "upd/param" if "upd/param" in c else None
    if key:
        f.plot(step, c[key], "tab:red", label="upd/param")
        f.axhline(1e-3, color="gray", ls=":", label="~1e-3 healthy")
        f.set_yscale("log")
    f.set_title("(F) update/param ratio (log)"); f.set_xlabel("step"); f.legend(fontsize=8)

    fig.suptitle(f"Training health -- {path}", fontsize=13)
    fig.tight_layout()
    out = path.rsplit(".", 1)[0] + "_curves.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    for p in (sys.argv[1:] or ["vae_dbn_run.csv"]):
        plot(p)
