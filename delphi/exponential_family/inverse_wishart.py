# Author: pstefanou12@
"""Inverse Wishart distribution in natural parameterization, over vech rows."""

import math

import torch as ch
from torch import distributions
from torch.distributions import constraints

from delphi.exponential_family import exponential_family_distribution, wishart
from delphi.utils import helpers


class InverseWishart(exponential_family_distribution.ExponentialFamilyDistribution):
    """Inverse Wishart distribution whose samples are lower triangles.

    Theta holds ``(df + size + 1) / 2``, then the lower triangle of
    ``-0.5 * scale``. ``X^{-1}`` is Wishart with the same df and scale
    ``inverse(scale)``, which is how it samples, and its statistic is the
    Wishart's of ``X^{-1}``. torch has no inverse Wishart.

    Attributes:
        theta: The natural parameter, which ``log_prob`` reads directly.
        df: Degrees of freedom.
        scale: Scale matrix.
    """

    arg_constraints = {"df": constraints.positive}
    support = constraints.real_vector

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and the vech width dims."""
        self.dims = dims
        self.size = helpers.vech_side(dims)
        self.theta = theta
        canonical = InverseWishart.to_canonical(theta)
        self.df = canonical[0]
        self.scale = helpers.unvech(canonical[1:], self.size)
        super().__init__(event_shape=ch.Size([dims]), validate_args=False)

    @property
    def _natural_params(self) -> tuple[ch.Tensor]:
        """Return the natural parameter, read by torch's Bregman KL."""
        return (self.theta,)

    def _log_normalizer(self, *natural):
        """Return ``log Gamma_p(df/2) + df p log(2)/2 - df log|scale|/2``."""
        (theta,) = natural
        canonical = InverseWishart.to_canonical(theta)
        df = canonical[0]
        scale = helpers.unvech(canonical[1:], self.size)
        return (
            -0.5 * df * ch.logdet(scale)
            + 0.5 * df * self.size * math.log(2.0)
            + ch.mvlgamma(df / 2.0, self.size)
        )

    def sample(self, sample_shape=ch.Size()) -> ch.Tensor:
        """Draw samples as vech rows, shaped ``sample_shape + (dims,)``."""
        with ch.no_grad():
            precision = distributions.Wishart(
                df=self.df, precision_matrix=self.scale, validate_args=False
            ).sample(sample_shape)
            return helpers.vech(ch.linalg.inv(precision))  # pylint: disable=not-callable

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Return ``<theta, T(X)> - A(theta)``."""
        return InverseWishart.calc_suff_stat(value) @ self.theta - self._log_normalizer(
            self.theta
        )

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return the Wishart statistic of ``X^{-1}``."""
        size = helpers.vech_side(x.size(-1))
        inverse = ch.linalg.inv(helpers.unvech(x, size))  # pylint: disable=not-callable
        return wishart.Wishart.calc_suff_stat(helpers.vech(inverse))

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert canonical ``[df, vech(scale)]`` to natural form."""
        size = helpers.vech_side(theta.numel() - 1)
        return ch.cat([(theta[:1] + size + 1) / 2.0, -0.5 * theta[1:]])

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert natural form to canonical ``[df, vech(scale)]``."""
        size = helpers.vech_side(theta.numel() - 1)
        return ch.cat([2.0 * theta[:1] - size - 1, -2.0 * theta[1:]])
