# Author: pstefanou12@
"""Gamma distribution with unknown shape, in natural parameterization."""

import torch as ch

from delphi.exponential_family import gamma


class GammaUnknownConcentration(gamma.Gamma):
    """Gamma distribution with both shape and rate unknown.

    Theta holds ``concentration - 1`` for every coordinate, then ``-rate``,
    which is the known-shape Gamma's theta with the shape prepended.
    """

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        super().__init__(theta[:dims] + 1, theta[dims:], dims)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return ``[log x, x]``, each one entry per coordinate."""
        return ch.cat([ch.log(x), gamma.Gamma.calc_suff_stat(x)], -1)

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert canonical ``[concentration, rate]`` to natural form."""
        concentration, rate = theta.chunk(2)
        return ch.cat([concentration - 1, gamma.Gamma.to_natural(rate)])

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert natural form to canonical ``[concentration, rate]``."""
        shifted, rate = theta.chunk(2)
        return ch.cat([shifted + 1, gamma.Gamma.to_canonical(rate)])
