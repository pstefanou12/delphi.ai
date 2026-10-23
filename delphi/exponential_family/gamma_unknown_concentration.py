# Author: pstefanou12@
"""Gamma distribution with unknown shape, in natural parameterization."""

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class GammaUnknownConcentration(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.Gamma,
):
    """Gamma distribution with both shape and rate unknown.

    Theta holds ``concentration - 1`` for every coordinate, then ``-rate``.
    """

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        super().__init__(theta[:dims] + 1, -theta[dims:])

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return ``[log x, x]``, each one entry per coordinate."""
        return ch.cat([ch.log(x), x], -1)

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert canonical ``[concentration, rate]`` to natural form."""
        dims = theta.size(-1) // 2
        return ch.cat([theta[:dims] - 1, -theta[dims:]])

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert natural form to canonical ``[concentration, rate]``."""
        dims = theta.size(-1) // 2
        return ch.cat([theta[:dims] + 1, -theta[dims:]])

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        return super().log_prob(value).sum(-1)
