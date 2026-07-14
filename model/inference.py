"""Exact inference on a (finite, Markov, factorized) HMM -- the payoff of conditions 1-3.

All three take log-space factors:
    log_pi : [B, K]        initial   log p(z_1)
    log_A  : [B, T, K, K]  transition logs (log_A[:, t] moves t-1 -> t; entries for t=1..T-1 used)
    log_B  : [B, T, K]     emission  logs
"""
import torch


def forward_log_likelihood(log_pi: torch.Tensor, log_A: torch.Tensor, log_B: torch.Tensor) -> torch.Tensor:
    """Forward algorithm. Returns the EXACT log p(obs | context): [B].

    This is THE training objective -- maximize it by gradient descent
    (end-to-end / generalized-EM training; no ELBO, no encoder).
    """
    raise NotImplementedError


def forward_backward(log_pi: torch.Tensor, log_A: torch.Tensor, log_B: torch.Tensor):
    """Returns posterior marginals log_gamma [B, T, K] and pairwise log_xi [B, T-1, K, K].
    (The EM E-step / diagnostics.)"""
    raise NotImplementedError


def viterbi(log_pi: torch.Tensor, log_A: torch.Tensor, log_B: torch.Tensor) -> torch.Tensor:
    """Max-product. Returns the EXACT MAP state path: [B, T] (long). Deployment decode."""
    raise NotImplementedError
