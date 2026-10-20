# Author: pstefanou12@
"""Gamma distribution with known shape, in natural parameterization."""

from typing import Union

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class Gamma(
    exponential_family_distribution.ExponentialFamilyDistribution, distributions.Gamma
):
    """Gamma distribution with known shape, parameterized by its rate."""

    def __init__(
        self, concentration: Union[ch.Tensor, float], theta: ch.Tensor, dims: int
    ):
        """Initialize with known shape, natural parameter theta and dimension."""
        self.dims = dims
        super().__init__(concentration, -theta)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return sufficient statistics for the Gamma distribution."""
        return x

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert the canonical rate to natural form."""
        return -theta

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert the natural parameter to the canonical rate."""
        return -theta

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        return super().log_prob(value).sum(-1)
