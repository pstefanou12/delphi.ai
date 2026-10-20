# Author: pstefanou12@
"""Log-normal distribution in natural parameterization."""

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class LogNormal(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.LogNormal,
):
    """Log-normal distribution parameterized by natural parameters.

    Theta holds ``loc / scale**2`` for every coordinate, then
    ``-1 / (2 * scale**2)``.
    """

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        variance = -0.5 / theta[dims:]
        super().__init__(theta[:dims] * variance, variance.sqrt())

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return ``[log x, log(x)**2]``, each one entry per coordinate."""
        log_x = ch.log(x)
        return ch.cat([log_x, log_x.pow(2)], -1)

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert canonical ``[loc, scale]`` to natural form."""
        dims = theta.size(-1) // 2
        loc, variance = theta[:dims], theta[dims:].pow(2)
        return ch.cat([loc / variance, -0.5 / variance])

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert natural parameters to canonical ``[loc, scale]``."""
        dims = theta.size(-1) // 2
        variance = -0.5 / theta[dims:]
        return ch.cat([theta[:dims] * variance, variance.sqrt()])

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        return super().log_prob(value).sum(-1)
