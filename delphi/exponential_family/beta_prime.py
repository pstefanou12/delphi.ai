# Author: pstefanou12@
"""Beta prime distribution in natural parameterization."""

import torch as ch
from torch import distributions
from torch.distributions import constraints

from delphi.exponential_family import exponential_family_distribution


class BetaPrime(exponential_family_distribution.ExponentialFamilyDistribution):
    """Beta prime distribution with two shapes, both unknown.

    Theta holds ``concentration1 - 1`` for every coordinate, then
    ``-(concentration1 + concentration0)``. ``x / (1 + x)`` is Beta with the
    same shapes; torch has no beta prime, so it samples as a ratio of Gammas.

    Attributes:
        concentration1: First shape, one per coordinate.
        concentration0: Second shape, one per coordinate.
    """

    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    support = constraints.positive

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        self.concentration1, self.concentration0 = BetaPrime.to_canonical(theta).split(
            dims
        )
        super().__init__(self.concentration1.shape)

    @property
    def _natural_params(self) -> tuple[ch.Tensor, ch.Tensor]:
        """Return the natural parameters, read by torch's Bregman KL."""
        return (
            self.concentration1 - 1.0,
            -(self.concentration1 + self.concentration0),
        )

    def _log_normalizer(self, *natural):
        """Return the log Beta function of the two shapes."""
        first, second = natural
        concentration1 = first + 1.0
        concentration0 = -second - concentration1
        return (
            ch.lgamma(concentration1)
            + ch.lgamma(concentration0)
            - ch.lgamma(concentration1 + concentration0)
        )

    def sample(self, sample_shape=ch.Size()) -> ch.Tensor:
        """Draw samples shaped ``sample_shape + (dims,)``."""
        with ch.no_grad():
            numerator = distributions.Gamma(self.concentration1, 1.0).sample(
                sample_shape
            )
            denominator = distributions.Gamma(self.concentration0, 1.0).sample(
                sample_shape
            )
            return numerator / denominator

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute summed log probability over all dimensions."""
        if self._validate_args:
            self._validate_sample(value)
        statistic = BetaPrime.calc_suff_stat(value)
        first, second = self._natural_params
        log_density = (
            statistic[..., : self.dims] * first
            + statistic[..., self.dims :] * second
            - self._log_normalizer(first, second)
        )
        return log_density.sum(-1)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return ``[log x, log(1 + x)]``, each one entry per coordinate."""
        return ch.cat([ch.log(x), ch.log1p(x)], -1)

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert canonical ``[concentration1, concentration0]`` to natural."""
        concentration1, concentration0 = theta.chunk(2)
        return ch.cat([concentration1 - 1.0, -(concentration1 + concentration0)])

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert natural to canonical ``[concentration1, concentration0]``."""
        first, second = theta.chunk(2)
        return ch.cat([first + 1.0, -second - first - 1.0])
