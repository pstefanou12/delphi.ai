# Author: pstefanou12@
"""Chi-squared distribution in natural parameterization."""

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class Chi2(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.Chi2,
):
    """Chi-squared distribution parameterized by ``df / 2 - 1``."""

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        super().__init__(2 * (theta + 1))

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return ``log x``; the ``-x / 2`` term is in the base measure."""
        return ch.log(x)

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert the canonical degrees of freedom to ``df / 2 - 1``."""
        return theta / 2 - 1

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert ``df / 2 - 1`` to the degrees of freedom."""
        return 2 * (theta + 1)

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        return super().log_prob(value).sum(-1)
