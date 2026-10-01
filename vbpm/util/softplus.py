"""Softplus inverse for initialising softplus-parameterised heads."""
import math


def inverse_softplus(x: float) -> float:
    """The raw value whose softplus is x."""
    return x + math.log(-math.expm1(-x))
