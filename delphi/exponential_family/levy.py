# Author: pstefanou12@
"""Lévy distribution with known location, in natural parameterization."""

import math
from typing import Union

import torch as ch
from torch.distributions import constraints

from delphi.exponential_family import exponential_family_distribution


class Levy(exponential_family_distribution.ExponentialFamilyDistribution):
    """Lévy distribution with known location, parameterized by ``-scale / 2``.

    ``x - loc`` is inverse gamma with shape 1/2 and rate ``scale / 2``, so it
    samples as ``loc + scale / Z^2`` for a standard normal Z; torch has no
    Lévy.

    Attributes:
        loc: Known location, one per coordinate.
        scale: Scale, one per coordinate.
    """

    arg_constraints = {"scale": constraints.positive}

    def __init__(self, loc: Union[ch.Tensor, float], theta: ch.Tensor, dims: int):
        """Initialize with known loc, natural parameter theta and dimension."""
        self.dims = dims
        self.scale = Levy.to_canonical(theta)
        self.loc = ch.as_tensor(loc, dtype=self.scale.dtype).expand_as(self.scale)
        super().__init__(self.scale.shape)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        """Return the open half-line above the location."""
        return constraints.greater_than(self.loc)

    @property
    def _natural_params(self) -> tuple[ch.Tensor]:
        """Return the natural parameter, read by torch's Bregman KL."""
        return (-self.scale / 2.0,)

    def _log_normalizer(self, *natural):
        """Return ``-log(-2 theta) / 2``, which is ``-log(scale) / 2``."""
        (theta,) = natural
        return -0.5 * ch.log(-2.0 * theta)

    def sample(self, sample_shape=ch.Size()) -> ch.Tensor:
        """Draw samples shaped ``sample_shape + (dims,)``."""
        shape = self._extended_shape(sample_shape)
        with ch.no_grad():
            normal = ch.randn(shape, dtype=self.scale.dtype, device=self.scale.device)
            return self.loc.expand(shape) + self.scale.expand(shape) / normal**2

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        if self._validate_args:
            self._validate_sample(value)
        shifted = value - self.loc
        log_density = (
            0.5 * ch.log(self.scale / (2.0 * math.pi))
            - 1.5 * ch.log(shifted)
            - self.scale / (2.0 * shifted)
        )
        return log_density.sum(-1)

    @staticmethod
    def calc_suff_stat(loc: Union[ch.Tensor, float], x: ch.Tensor) -> ch.Tensor:
        """Return ``1 / (x - loc)``."""
        return 1.0 / (x - loc)

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert the canonical scale to ``-scale / 2``."""
        return -theta / 2.0

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert ``-scale / 2`` to the scale."""
        return -2.0 * theta
