# Author: pstefanou12@
"""Dirichlet distribution in natural parameterization."""

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class Dirichlet(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.Dirichlet,
):
    """Dirichlet distribution parameterized by ``concentration - 1``.

    Unlike the other families, ``dims`` is the number of categories: one
    sample is a single point on the simplex, not ``dims`` independent draws.
    """

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and number of categories."""
        self.dims = dims
        super().__init__(theta + 1)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return ``log x``, one entry per category."""
        return ch.log(x)

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert the canonical concentration to ``concentration - 1``."""
        return theta - 1

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert ``concentration - 1`` to the concentration."""
        return theta + 1
