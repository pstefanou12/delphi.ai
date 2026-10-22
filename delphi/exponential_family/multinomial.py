# Author: pstefanou12@
"""Multinomial distribution with known trials, in natural parameterization."""

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class Multinomial(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.Multinomial,
):
    """Multinomial distribution with known trials over ``dims`` categories.

    Theta holds ``log(probs_k / probs_last)`` for every category but the last:
    the counts sum to ``total_count``, so one category is redundant and the
    minimal representation drops it.
    """

    def __init__(self, total_count: int, theta: ch.Tensor, dims: int):
        """Initialize with known trials, natural parameter theta and categories."""
        self.dims = dims
        super().__init__(total_count, logits=ch.cat([theta, theta.new_zeros(1)]))

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return the counts of every category but the last."""
        return x[..., :-1]

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert canonical probs to ``log(probs_k / probs_last)``."""
        return ch.log(theta[:-1]) - ch.log(theta[-1])

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert ``log(probs_k / probs_last)`` to the probs."""
        return ch.softmax(ch.cat([theta, theta.new_zeros(1)]), -1)
