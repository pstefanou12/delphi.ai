# Author: pstefanou12@
"""Beta distribution in natural parameterization."""

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class Beta(
    exponential_family_distribution.ExponentialFamilyDistribution, distributions.Beta
):
    """Beta distribution parameterized by natural parameters.

    Theta holds ``a - 1`` for every coordinate, then ``b - 1``.
    """

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        super().__init__(theta[:dims] + 1, theta[dims:] + 1)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return ``[log x, log(1 - x)]``, each one entry per coordinate."""
        return ch.cat([ch.log(x), ch.log1p(-x)], -1)

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert canonical ``[a, b]`` to natural ``[a - 1, b - 1]``."""
        return theta - 1

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert natural ``[a - 1, b - 1]`` to canonical ``[a, b]``."""
        return theta + 1

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        return super().log_prob(value).sum(-1)
