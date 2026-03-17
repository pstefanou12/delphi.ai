# Author: pstefanou12@
"""Normal distribution with known variance in natural parameterization."""

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class NormalKnownVariance(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.Normal,
):
    """Normal distribution parameterized by natural parameters with known variance."""

    def __init__(self, variance: ch.Tensor, theta: ch.Tensor):
        """Initialize with known variance and natural parameter theta."""
        loc = theta * variance
        scale = ch.sqrt(variance)
        super().__init__(loc, scale)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return sufficient statistics for normal with known variance."""
        return x

    @staticmethod
    def to_natural(theta: ch.Tensor, variance: ch.Tensor) -> ch.Tensor:
        """Convert canonical mean to natural parameters."""
        return theta / variance

    @staticmethod
    def to_canonical(theta: ch.Tensor, variance: ch.Tensor) -> ch.Tensor:
        """Convert natural parameters to canonical mean."""
        return theta * variance

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        result = super().log_prob(value)
        return result.sum(-1)
