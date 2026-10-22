# Author: pstefanou12@
"""Half-normal distribution in natural parameterization."""

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class HalfNormal(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.HalfNormal,
):
    """Half-normal distribution parameterized by ``-1 / (2 scale^2)``."""

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        super().__init__(HalfNormal.to_canonical(theta))

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return ``x^2``."""
        return x**2

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert the canonical scale to ``-1 / (2 scale^2)``."""
        return -0.5 / theta**2

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert ``-1 / (2 scale^2)`` to the scale."""
        return ch.sqrt(-0.5 / theta)

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        return super().log_prob(value).sum(-1)
