# Author: pstefanou12@
"""Rayleigh distribution in natural parameterization."""

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class Rayleigh(
    exponential_family_distribution.ExponentialFamilyDistribution, distributions.Weibull
):
    """Rayleigh distribution parameterized by ``-1 / (2 scale^2)``.

    It is the Weibull with shape 2 and scale ``sqrt(2) * scale``, which torch
    already samples and scores.
    """

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        super().__init__(ch.sqrt(-1.0 / theta), ch.full_like(theta, 2.0))

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
