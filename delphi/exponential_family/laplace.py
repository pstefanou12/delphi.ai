# Author: pstefanou12@
"""Laplace distribution with known location, in natural parameterization."""

from typing import Union

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class Laplace(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.Laplace,
):
    """Laplace distribution with known location, parameterized by its scale."""

    def __init__(self, loc: Union[ch.Tensor, float], theta: ch.Tensor, dims: int):
        """Initialize with known location, natural parameter theta and dimension."""
        self.dims = dims
        super().__init__(loc, -1.0 / theta)

    @staticmethod
    def calc_suff_stat(loc: Union[ch.Tensor, float], x: ch.Tensor) -> ch.Tensor:
        """Return the distance of each sample from the known location."""
        return (x - loc).abs()

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert the canonical scale to natural form."""
        return -1.0 / theta

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert the natural parameter to the canonical scale."""
        return -1.0 / theta

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        return super().log_prob(value).sum(-1)
