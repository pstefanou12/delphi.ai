# Author: pstefanou12@
"""Weibull distribution in natural parameterization."""

from typing import Union

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class Weibull(
    exponential_family_distribution.ExponentialFamilyDistribution, distributions.Weibull
):
    """Weibull distribution parameterized by natural parameters."""

    def __init__(self, k: Union[ch.Tensor, float], theta: ch.Tensor, dims: int):
        """Initialize with shape k, natural parameter theta, and dimension."""
        self.dims = dims
        lambda_ = (-1 / theta).pow(1 / k)
        super().__init__(lambda_, k)

    @staticmethod
    def calc_suff_stat(k: Union[ch.Tensor, float], x: ch.Tensor) -> ch.Tensor:
        """Return sufficient statistics for Weibull distribution."""
        return x.pow(k)

    @staticmethod
    def to_natural(k: Union[ch.Tensor, float], theta: ch.Tensor) -> ch.Tensor:
        """Convert canonical scale parameter to natural form."""
        return -1.0 / theta.pow(k)

    @staticmethod
    def to_canonical(k: Union[ch.Tensor, float], theta: ch.Tensor) -> ch.Tensor:
        """Convert natural parameters to canonical scale parameter."""
        return (-1 / theta).pow(1 / k)

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        result = super().log_prob(value)
        return result.sum(-1)
