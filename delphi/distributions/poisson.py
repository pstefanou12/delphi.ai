# Author: pstefanou12@
"""Poisson distribution in natural parameterization."""

import torch as ch
from torch.distributions import Poisson


def calc_poiss_suff_stat(x):
    """Return sufficient statistics for Poisson distribution."""
    return x


class ExponentialFamilyPoisson(Poisson):  # pylint: disable=abstract-method
    """Poisson distribution parameterized by natural parameters."""

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        lambda_ = ch.exp(theta)
        super().__init__(lambda_)

    def log_prob(self, value):
        """Compute summed log probability over all dimensions."""
        result = super().log_prob(value)
        return result.sum(-1)
