# Author: pstefanou12@
"""Inverse gamma distribution with unknown shape, in natural form."""

import torch as ch

from delphi.exponential_family import inverse_gamma


class InverseGammaUnknownConcentration(inverse_gamma.InverseGamma):
    """Inverse gamma distribution with both shape and rate unknown.

    Theta holds ``-(concentration + 1)`` for every coordinate, then ``-rate``,
    which is the known-shape inverse gamma's theta with the shape prepended.
    """

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        super().__init__(-theta[:dims] - 1, theta[dims:], dims)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return ``[log x, 1 / x]``, each one entry per coordinate."""
        return ch.cat([ch.log(x), inverse_gamma.InverseGamma.calc_suff_stat(x)], -1)

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert canonical ``[concentration, rate]`` to natural form."""
        concentration, rate = theta.chunk(2)
        return ch.cat([-concentration - 1, inverse_gamma.InverseGamma.to_natural(rate)])

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert natural form to canonical ``[concentration, rate]``."""
        shifted, rate = theta.chunk(2)
        return ch.cat([-shifted - 1, inverse_gamma.InverseGamma.to_canonical(rate)])
