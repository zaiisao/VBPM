"""Smoke test for the combined attack: τ_bar latent + scheduled sampling.

Checks forward (ss off/on), τ_bar KL plumbing, backward, and free-running inference
for finite values and correct shapes. Run: python -m tests.smoke_combined
"""
import torch

from models.svt_core import SVTModel
from models.loss import compute_elbo_loss

torch.manual_seed(0)
B, T, C, K = 2, 96, 2, 8
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SVTModel(
    hidden_dim=128, nhead=4, num_layers=2, num_meter_classes=K, input_dim=C,
    phase_corr_scale=0.1, tempo_corr_scale=0.15, decoder_use_h_prior=False,
    tempo_anchor_mode="latent", tempo_reversion_alpha=0.1,
).to(device)

acts = torch.randn(B, T, C, device=device)
beats = (torch.rand(B, T, device=device) < 0.1).float()
dbeats = (torch.rand(B, T, device=device) < 0.03).float()


def run(tag, ss):
    model.scheduled_sampling_eps = ss
    out = model(acts, temperature=0.5, beat_targets=beats, downbeat_targets=dbeats)
    assert out["tempo_bar"] is not None, "tempo_bar missing"
    tb = out["tempo_bar"]
    for k, v in tb.items():
        assert torch.isfinite(v).all(), f"{tag}: tempo_bar[{k}] not finite"
        assert v.shape == (B,), f"{tag}: tempo_bar[{k}] shape {v.shape} != ({B},)"
    assert out["beat_logits"].shape == (B, T, 2)
    total, comps = compute_elbo_loss(
        beat_logits=out["beat_logits"], beat_targets=beats,
        posterior=out["posterior"], prior=out["prior"], beta=1.0,
        pos_weight=5.0, pos_weight_db=15.0,
        free_bits_phase=0.2, free_bits_tempo=0.1,
        downbeat_targets=dbeats, tempo_bar=out.get("tempo_bar"),
    )
    assert torch.isfinite(total).all(), f"{tag}: total loss not finite"
    assert "kl_taubar" in comps and torch.isfinite(comps["kl_taubar"]).all()
    model.zero_grad()
    total.backward()
    # τ_bar heads must receive gradient
    g = model.tempo_bar_post_head.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, \
        f"{tag}: tempo_bar_post_head got no/zero/NaN grad"
    print(f"  [{tag}] ss={ss} total={total.item():.4f} "
          f"kl_taubar={comps['kl_taubar'].item():.4f} "
          f"kl_phase={comps['kl_phase'].item():.4f} "
          f"taubar_KL_grad_ok bce={comps['bce'].item():.4f}")
    return out


print("=== forward + loss + backward ===")
run("ss_off", 0.0)
run("ss_on", 0.5)

print("=== free-running inference (sample_from_prior) ===")
model.eval()
with torch.no_grad():
    inf = model.sample_from_prior(acts, temperature=0.1)
for key in ("phase", "phase_mu", "log_tempo", "beat_logits"):
    v = inf[key]
    assert torch.isfinite(v).all(), f"inference {key} not finite"
    print(f"  inf[{key}] shape={tuple(v.shape)} "
          f"min={v.min().item():.3f} max={v.max().item():.3f}")
# free-running tempo should sit in the musical band, anchored by prior τ_bar
bpm = torch.exp(inf["log_tempo"]) * 86.1328125 / (2 * 3.141592653589793) * 60
print(f"  free-running tempo: {bpm.mean().item():.1f} BPM (mean), "
      f"[{bpm.min().item():.0f}, {bpm.max().item():.0f}]")
print("ALL SMOKE CHECKS PASSED")
