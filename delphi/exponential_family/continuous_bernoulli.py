# Author: pstefanou12@
"""Continuous Bernoulli distribution in natural parameterization."""

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class ContinuousBernoulli(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.ContinuousBernoulli,
):
    """Continuous Bernoulli distribution parameterized by its logits."""

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        super().__init__(logits=theta)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return sufficient statistics for the continuous Bernoulli."""
        return x

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert the canonical probs to logits."""
        return ch.logit(theta)

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert the logits to the canonical probs."""
        return ch.sigmoid(theta)

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        return super().log_prob(value).sum(-1)
