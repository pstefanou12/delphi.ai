# Author: pstefanou12@
"""Negative binomial distribution with known failures, in natural form."""

from typing import Union

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class NegativeBinomial(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.NegativeBinomial,
):
    """Negative binomial with known failures, parameterized by ``log(probs)``.

    As in torch, it counts successes before ``total_count`` failures.
    """

    def __init__(
        self, total_count: Union[ch.Tensor, float], theta: ch.Tensor, dims: int
    ):
        """Initialize with known failures, natural parameter theta and dimension."""
        self.dims = dims
        super().__init__(total_count, probs=ch.exp(theta))

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return sufficient statistics for the negative binomial distribution."""
        return x

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert the canonical success probability to natural form."""
        return ch.log(theta)

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert the natural parameter to the canonical success probability."""
        return ch.exp(theta)

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        return super().log_prob(value).sum(-1)
