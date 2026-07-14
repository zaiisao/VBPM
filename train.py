"""Train the neural bar-pointer HMM by maximizing the exact forward-algorithm log-likelihood
(end-to-end gradient / generalized EM). No ELBO, no encoder.
"""
import torch

from config import Config, load_config
from model import NeuralBarPointerHMM
from data import build_dataloader
from losses import objective


def build_model(cfg: Config) -> NeuralBarPointerHMM:
    """Instantiate the model from config."""
    raise NotImplementedError


def train(cfg: Config):
    """Main loop: for each batch  loss = objective(model, batch); backprop; step; eval/save periodically."""
    raise NotImplementedError


def main():
    cfg = load_config()
    train(cfg)


if __name__ == "__main__":
    main()
