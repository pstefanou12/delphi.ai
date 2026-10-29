# Author: pstefanou12@
"""Logarithmic (log-series) distribution in natural parameterization."""

import torch as ch
from torch.distributions import constraints

from delphi.exponential_family import exponential_family_distribution


class Logarithmic(exponential_family_distribution.ExponentialFamilyDistribution):
    """Logarithmic distribution, parameterized by ``log(probs)``.

    ``P(k) = -probs^k / (k log(1 - probs))`` on the positive integers. torch
    has no logarithmic distribution, so it samples by Kemp's (1981) LK
    algorithm.

    Attributes:
        probs: Success probability, one per coordinate.
    """

    arg_constraints = {"probs": constraints.half_open_interval(0.0, 1.0)}
    support = constraints.positive_integer

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        self.probs = Logarithmic.to_canonical(theta)
        super().__init__(self.probs.shape)

    @property
    def _natural_params(self) -> tuple[ch.Tensor]:
        """Return the natural parameter, read by torch's Bregman KL."""
        return (ch.log(self.probs),)

    def _log_normalizer(self, *natural):
        """Return ``log(-log(1 - exp(theta)))``."""
        (theta,) = natural
        return ch.log(-ch.log1p(-ch.exp(theta)))

    @property
    def mean(self) -> ch.Tensor:
        """Return the mean, ``-probs / ((1 - probs) log(1 - probs))``."""
        return -self.probs / ((1.0 - self.probs) * ch.log1p(-self.probs))

    def sample(self, sample_shape=ch.Size()) -> ch.Tensor:
        """Draw samples shaped ``sample_shape + (dims,)``."""
        shape = self._extended_shape(sample_shape)
        with ch.no_grad():
            probs = self.probs.expand(shape)
            # Uniforms on (0, 1]: a zero sends log(first) / log(q) to inf.
            first = 1.0 - ch.rand(shape, dtype=probs.dtype, device=probs.device)
            second = 1.0 - ch.rand(shape, dtype=probs.dtype, device=probs.device)
            q = -ch.expm1(ch.log1p(-probs) * second)
            tail = ch.floor(1.0 + ch.log(first) / ch.log(q)).clamp(min=1.0)
            small = ch.where(first >= q, 1.0, 2.0).to(probs.dtype)
            draws = ch.where(first <= q**2, tail, small)
            return ch.where(first >= probs, ch.ones_like(draws), draws)

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        if self._validate_args:
            self._validate_sample(value)
        (theta,) = self._natural_params
        log_mass = value * theta - ch.log(value) - self._log_normalizer(theta)
        return log_mass.sum(-1)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return ``x``."""
        return x

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert the canonical probs to ``log(probs)``."""
        return ch.log(theta)

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert ``log(probs)`` to the probs."""
        return ch.exp(theta)
