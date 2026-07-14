"""Bar-pointer Neural HMM (Baseline B).

Data flow:
    features x  --context-->  h_t
    (z_{t-1}, h_t)  --transition-->  log A        z_t  --emission-->  log B
    forward algorithm  ->  exact log p(obs | x)   (train)
    Viterbi            ->  MAP state path -> readout   (deploy)

Keep the transition Markov-in-z and the emission factorized; those two properties are the only
reason the forward algorithm / Viterbi are exact.
"""
from model.state_space import StateSpace
from model.transition import TransitionModel
from model.emission import EmissionModel
from model.neural_hmm import NeuralBarPointerHMM

__all__ = ["StateSpace", "TransitionModel", "EmissionModel", "NeuralBarPointerHMM"]
