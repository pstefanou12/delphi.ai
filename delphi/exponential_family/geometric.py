# Author: pstefanou12@
"""Geometric distribution in natural parameterization."""

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class Geometric(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.Geometric,
):
    """Geometric distribution over failures before the first success.

    The natural parameter is ``log(1 - probs)``.
    """

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        super().__init__(probs=-ch.expm1(theta))

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return sufficient statistics for the geometric distribution."""
        return x

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert the canonical success probability to natural form."""
        return ch.log1p(-theta)

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert the natural parameter to the canonical success probability."""
        return -ch.expm1(theta)

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        return super().log_prob(value).sum(-1)
