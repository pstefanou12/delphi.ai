# Author: pstefanou12@
"""Inverse Gaussian (Wald) distribution in natural parameterization."""

import math

import torch as ch
from torch.distributions import constraints

from delphi.exponential_family import exponential_family_distribution


class InverseGaussian(exponential_family_distribution.ExponentialFamilyDistribution):
    """Inverse Gaussian distribution with mean ``loc`` and shape ``concentration``.

    Theta holds ``-concentration / (2 loc^2)`` for every coordinate, then
    ``-concentration / 2``. torch has no inverse Gaussian, so this one
    samples by Michael, Schucany and Haas (1976).

    Attributes:
        loc: Mean, one per coordinate.
        concentration: Shape, one per coordinate.
    """

    arg_constraints = {
        "loc": constraints.positive,
        "concentration": constraints.positive,
    }
    support = constraints.positive

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        self.loc, self.concentration = InverseGaussian.to_canonical(theta).split(dims)
        super().__init__(self.loc.shape)

    @property
    def _natural_params(self) -> tuple[ch.Tensor, ch.Tensor]:
        """Return the natural parameters, read by torch's Bregman KL."""
        return (
            -self.concentration / (2.0 * self.loc**2),
            -self.concentration / 2.0,
        )

    def _log_normalizer(self, *natural):
        """Return ``-2 sqrt(eta_x eta_inv) - log(-2 eta_inv) / 2``."""
        first, second = natural
        return -2.0 * ch.sqrt(first * second) - 0.5 * ch.log(-2.0 * second)

    @property
    def mean(self) -> ch.Tensor:
        """Return the mean."""
        return self.loc

    @property
    def variance(self) -> ch.Tensor:
        """Return the variance, ``loc^3 / concentration``."""
        return self.loc**3 / self.concentration

    def sample(self, sample_shape=ch.Size()) -> ch.Tensor:
        """Draw samples shaped ``sample_shape + (dims,)``."""
        shape = self._extended_shape(sample_shape)
        with ch.no_grad():
            loc = self.loc.expand(shape)
            concentration = self.concentration.expand(shape)
            spread = loc * ch.randn(shape, dtype=loc.dtype, device=loc.device) ** 2
            # The smaller root cancels catastrophically; it is loc^2 / larger.
            larger = loc + loc / (2.0 * concentration) * (
                spread + ch.sqrt(4.0 * concentration * spread + spread**2)
            )
            smaller = loc**2 / larger
            uniform = ch.rand(shape, dtype=loc.dtype, device=loc.device)
            return ch.where(uniform * (loc + smaller) <= loc, smaller, larger)

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        if self._validate_args:
            self._validate_sample(value)
        log_density = (
            0.5 * ch.log(self.concentration / (2.0 * math.pi))
            - 1.5 * ch.log(value)
            - self.concentration * (value - self.loc) ** 2 / (2.0 * self.loc**2 * value)
        )
        return log_density.sum(-1)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return ``[x, 1 / x]``, each one entry per coordinate."""
        return ch.cat([x, 1.0 / x], -1)

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert canonical ``[loc, concentration]`` to natural form."""
        loc, concentration = theta.chunk(2)
        return ch.cat([-concentration / (2.0 * loc**2), -concentration / 2.0])

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert natural form to canonical ``[loc, concentration]``."""
        first, second = theta.chunk(2)
        return ch.cat([ch.sqrt(second / first), -2.0 * second])
