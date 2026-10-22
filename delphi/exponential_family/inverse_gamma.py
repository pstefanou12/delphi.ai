# Author: pstefanou12@
"""Inverse gamma distribution with known shape, in natural parameterization."""

from typing import Union

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class InverseGamma(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.InverseGamma,
):
    """Inverse gamma distribution with known shape, parameterized by ``-rate``.

    With the shape known, ``-(concentration + 1) log x`` is base measure and
    ``1 / x`` the sufficient statistic.
    """

    def __init__(
        self, concentration: Union[ch.Tensor, float], theta: ch.Tensor, dims: int
    ):
        """Initialize with known shape, natural parameter theta and dimension."""
        self.dims = dims
        super().__init__(concentration, -theta)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return ``1 / x``."""
        return 1.0 / x

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
