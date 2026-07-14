"""The VBPM's own deploy-time evidence head: frozen frontend features -> beat/downbeat
probabilities that the particle filter observes.

System-boundary directive (2026-07-12): the frontend contributes FROZEN FEATURES ONLY. The
observation the filter weights particles against must be produced by OUR module trained on OUR
(fold-honest) data -- never by the frontend's own task heads (Beat This act2 remains only in the
peak-pick BASELINE, which legitimately is Beat This). One class serves every frontend; only
``feature_dim`` changes (Beat This 512, MERT 768).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActivationHead(nn.Module):
    def __init__(self, feature_dim, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden_size), nn.GELU(),
                                 nn.Linear(hidden_size, 2))

    def forward(self, features):
        # [.., feature_dim] -> [.., 2] beat/downbeat probabilities in [0, 1]
        return torch.sigmoid(self.net(features))


def train_activation_head(songs, feature_dim, steps=2000, batch_size=32, crop_frames=1024,
                          learning_rate=1e-3, pos_weight=(20.0, 60.0), device="cuda", seed=0):
    """Supervised BCE on fold-honest caches. Sparse positives (one frame per beat) need a positive
    weight roughly matching the class ratio (~1 beat / 40 frames) or the head collapses to zero."""
    from data.dataset import sample_training_crops
    torch.manual_seed(seed)
    head = ActivationHead(feature_dim).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    weight = torch.tensor(pos_weight, device=device)
    for step in range(1, steps + 1):
        features, beats, downbeats = sample_training_crops(songs, crop_frames, batch_size, device)
        logits = head.net(features)
        targets = torch.stack([beats, downbeats], dim=-1)
        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 500 == 0 or step == steps:
            print(f"  [acthead] step {step} loss {float(loss):.4f}", flush=True)
    head.eval()
    return head
