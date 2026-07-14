"""Evaluate: decode each val song (Viterbi) and score beat/downbeat F, CMLt, AMLt with mir_eval."""
import torch

from config import load_config
from model import NeuralBarPointerHMM


def evaluate(model: NeuralBarPointerHMM, songs) -> dict:
    """Decode + score every song; return averaged
    {beatF, beatCMLt, beatAMLt, dbF, dbCMLt, dbAMLt}."""
    raise NotImplementedError


def main():
    cfg = load_config()
    raise NotImplementedError


if __name__ == "__main__":
    main()
