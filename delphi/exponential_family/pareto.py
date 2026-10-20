# Author: pstefanou12@
"""Pareto distribution with known scale, in natural parameterization."""

from typing import Union

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class Pareto(
    exponential_family_distribution.ExponentialFamilyDistribution, distributions.Pareto
):
    """Pareto distribution with known scale, parameterized by its tail index."""

    def __init__(self, scale: Union[ch.Tensor, float], theta: ch.Tensor, dims: int):
        """Initialize with known scale, natural parameter theta and dimension."""
        self.dims = dims
        super().__init__(scale, -theta - 1)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return sufficient statistics for the Pareto distribution."""
        return ch.log(x)

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert the canonical tail index to natural form."""
        return -(theta + 1)

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert the natural parameter to the canonical tail index."""
        return -theta - 1

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        return super().log_prob(value).sum(-1)
