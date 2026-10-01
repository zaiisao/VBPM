"""Closed-form KL divergences between Gaussian and Laplace distributions."""
import math

import torch


def gaussian_kl(mu_q, sigma_q, mu_p, sigma_p):
    """KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)), closed form, elementwise."""
    return (torch.log(sigma_p / sigma_q)
            + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2.0 * sigma_p ** 2) - 0.5)


def gaussian_laplace_kl(mu_q, sigma_q, mu_p, scale_p):
    """KL(N(mu_q, sigma_q^2) || Laplace(mu_p, scale_p)), closed form, elementwise."""
    d = mu_q - mu_p
    mean_abs = (sigma_q * math.sqrt(2.0 / math.pi) * torch.exp(-d ** 2 / (2.0 * sigma_q ** 2))
                + d * torch.erf(d / (sigma_q * math.sqrt(2.0))))
    return (-0.5 * torch.log(2.0 * math.pi * math.e * sigma_q ** 2)
            + torch.log(2.0 * scale_p) + mean_abs / scale_p)
