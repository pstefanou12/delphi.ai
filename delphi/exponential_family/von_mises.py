# Author: pstefanou12@
"""von Mises distribution in natural parameterization."""

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class VonMises(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.VonMises,
):
    """von Mises distribution on the circle.

    Theta holds ``concentration * cos(loc)`` for every coordinate, then
    ``concentration * sin(loc)``.
    """

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        loc, concentration = VonMises.to_canonical(theta).split(dims)
        # torch's sampler divides by the concentration, which theta can zero.
        super().__init__(loc, concentration.clamp(min=1e-6))

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return ``[cos x, sin x]``, each one entry per coordinate."""
        return ch.cat([ch.cos(x), ch.sin(x)], -1)

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert canonical ``[loc, concentration]`` to natural form."""
        loc, concentration = theta.chunk(2)
        return ch.cat([concentration * ch.cos(loc), concentration * ch.sin(loc)])

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert natural form to canonical ``[loc, concentration]``."""
        cos, sin = theta.chunk(2)
        return ch.cat([ch.atan2(sin, cos), ch.hypot(cos, sin)])

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        return super().log_prob(value).sum(-1)
