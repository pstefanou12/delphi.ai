# Author: pstefanou12@
"""Normal distribution in natural parameterization."""

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class Normal(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.Normal,
):
    """Normal distribution parameterized by natural parameters.

    The natural parameter vector theta has two components:
    theta = [T, v] where T = -1/(2*variance) and v = loc/variance.
    The canonical parameter vector is [variance, loc].
    """

    def __init__(self, theta: ch.Tensor):
        """Initialize with natural parameter theta of shape (2,)."""
        T = theta[0]  # pylint: disable=invalid-name
        v = theta[1]
        variance = -0.5 / T
        loc = v * variance
        scale = ch.sqrt(variance)
        super().__init__(loc, scale)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return sufficient statistics [x^2, x] for normal distribution."""
        return ch.cat([x.pow(2), x], dim=1)

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert canonical [variance, loc] to natural parameters [T, v]."""
        variance = theta[0:1]
        loc = theta[1:2]
        return ch.cat([-0.5 / variance, loc / variance])

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert natural [T, v] to canonical [variance, loc]."""
        T = theta[0:1]  # pylint: disable=invalid-name
        v = theta[1:2]
        variance = -0.5 / T
        loc = v * variance
        return ch.cat([variance, loc])

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        result = super().log_prob(value)
        return result.sum(-1)
