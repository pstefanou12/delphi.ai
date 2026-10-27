# Author: pstefanou12@
"""Maxwell-Boltzmann distribution in natural parameterization."""

import math

import torch as ch
from torch.distributions import constraints

from delphi.exponential_family import exponential_family_distribution


class Maxwell(exponential_family_distribution.ExponentialFamilyDistribution):
    """Maxwell distribution parameterized by ``-1 / (2 scale^2)``.

    It is the norm of a three-dimensional normal with standard deviation
    ``scale``, which is how it samples; torch has no Maxwell.

    Attributes:
        scale: Scale, one per coordinate.
    """

    arg_constraints = {"scale": constraints.positive}
    support = constraints.positive

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        self.scale = Maxwell.to_canonical(theta)
        super().__init__(self.scale.shape)

    @property
    def _natural_params(self) -> tuple[ch.Tensor]:
        """Return the natural parameter, read by torch's Bregman KL."""
        return (-0.5 / self.scale**2,)

    def _log_normalizer(self, *natural):
        """Return ``-1.5 log(-2 theta)``, which is ``log(scale^3)``."""
        (theta,) = natural
        return -1.5 * ch.log(-2.0 * theta)

    @property
    def mean(self) -> ch.Tensor:
        """Return the mean, ``2 scale sqrt(2 / pi)``."""
        return 2.0 * self.scale * math.sqrt(2.0 / math.pi)

    @property
    def variance(self) -> ch.Tensor:
        """Return the variance, ``scale^2 (3 - 8 / pi)``."""
        return self.scale**2 * (3.0 - 8.0 / math.pi)

    def sample(self, sample_shape=ch.Size()) -> ch.Tensor:
        """Draw samples shaped ``sample_shape + (dims,)``."""
        shape = self._extended_shape(sample_shape)
        with ch.no_grad():
            normal = ch.randn(
                (*shape, 3), dtype=self.scale.dtype, device=self.scale.device
            )
            return self.scale.expand(shape) * normal.norm(dim=-1)

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        if self._validate_args:
            self._validate_sample(value)
        log_density = (
            0.5 * math.log(2.0 / math.pi)
            + 2.0 * ch.log(value)
            - value**2 / (2.0 * self.scale**2)
            - 3.0 * ch.log(self.scale)
        )
        return log_density.sum(-1)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return ``x^2``."""
        return x**2

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert the canonical scale to ``-1 / (2 scale^2)``."""
        return -0.5 / theta**2

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert ``-1 / (2 scale^2)`` to the scale."""
        return ch.sqrt(-0.5 / theta)
