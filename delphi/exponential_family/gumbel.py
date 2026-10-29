# Author: pstefanou12@
"""Gumbel distribution with known scale, in natural parameterization."""

from typing import Union

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class Gumbel(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.Gumbel,
):
    """Gumbel distribution with known scale, parameterized by ``-exp(loc / s)``.

    For ``s`` the known scale, ``-x / s`` is base measure and
    ``exp(-x / s)`` the sufficient statistic.
    """

    def __init__(self, scale: Union[ch.Tensor, float], theta: ch.Tensor, dims: int):
        """Initialize with known scale, natural parameter theta and dims."""
        self.dims = dims
        super().__init__(Gumbel.to_canonical(scale, theta), scale)

    @staticmethod
    def calc_suff_stat(scale: Union[ch.Tensor, float], x: ch.Tensor) -> ch.Tensor:
        """Return ``exp(-x / scale)``."""
        return ch.exp(-x / scale)

    @staticmethod
    def to_natural(scale: Union[ch.Tensor, float], theta: ch.Tensor) -> ch.Tensor:
        """Convert the canonical loc to ``-exp(loc / scale)``."""
        return -ch.exp(theta / scale)

    @staticmethod
    def to_canonical(scale: Union[ch.Tensor, float], theta: ch.Tensor) -> ch.Tensor:
        """Convert ``-exp(loc / scale)`` to the loc."""
        return scale * ch.log(-theta)

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        return super().log_prob(value).sum(-1)
