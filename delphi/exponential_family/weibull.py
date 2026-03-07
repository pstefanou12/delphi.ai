# Author: pstefanou12@
"""Weibull distribution in natural parameterization."""

import torch as ch
import torch.distributions as distributions

import delphi.exponential_family.exponential_family_distribution as exponential_family_distribution


class ExponentialFamilyWeibull(
    exponential_family_distribution.ExponentialFamilyDistribution, distributions.Weibull
):
    """Weibull distribution parameterized by natural parameters."""

    def __init__(self, k: ch.Tensor, theta: ch.Tensor, dims: int):
        """Initialize with shape k, natural parameter theta, and dimension."""
        self.dims = dims
        lambda_ = (-1 / theta).pow(1 / k)
        super().__init__(lambda_, k)

    @staticmethod
    def calc_suff_stat(k: ch.Tensor, x: ch.Tensor) -> ch.Tensor:
        """Return sufficient statistics for Weibull distribution."""
        return x.pow(k)

    @staticmethod
    def to_natural(k: ch.Tensor, theta: ch.Tensor) -> ch.Tensor:
        """Convert canonical scale parameter to natural form."""
        return -1.0 / theta.pow(k)

    @staticmethod
    def to_canonical(k: ch.Tensor, theta: ch.Tensor) -> ch.Tensor:
        """Convert natural parameters to canonical scale parameter."""
        return (-1 / theta).pow(1 / k)

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        result = super().log_prob(value)
        return result.sum(-1)
