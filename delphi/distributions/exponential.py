# Author: pstefanou12@
"""Exponential distribution in natural parameterization."""

import torch as ch
from torch.distributions import Exponential


def calc_exp_suff_stat(x):
    """Return sufficient statistics for exponential distribution."""
    return x


class ExponentialFamilyExponential(Exponential):  # pylint: disable=abstract-method
    """Exponential distribution parameterized by natural parameters."""

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        lambda_ = -theta
        super().__init__(lambda_)

    def log_prob(self, value):
        """Compute summed log probability over all dimensions."""
        result = super().log_prob(value)
        return result.sum(-1)
