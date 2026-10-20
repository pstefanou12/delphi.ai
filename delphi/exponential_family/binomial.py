# Author: pstefanou12@
"""Binomial distribution with known trials, in natural parameterization."""

from typing import Union

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class Binomial(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.Binomial,
):
    """Binomial distribution with known trials, parameterized by log-odds."""

    def __init__(self, total_count: Union[ch.Tensor, int], theta: ch.Tensor, dims: int):
        """Initialize with known trials, natural parameter theta and dimension."""
        self.dims = dims
        super().__init__(total_count, logits=theta)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return sufficient statistics for the binomial distribution."""
        return x

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert the canonical success probability to log-odds."""
        return ch.logit(theta)

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert log-odds to the canonical success probability."""
        return ch.sigmoid(theta)

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        return super().log_prob(value).sum(-1)
