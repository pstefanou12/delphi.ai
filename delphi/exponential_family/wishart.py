# Author: pstefanou12@
"""Wishart distribution in natural parameterization, over vech rows."""

import math

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution
from delphi.utils import helpers


class Wishart(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.Wishart,
):
    """Wishart distribution whose samples are the lower triangles of matrices.

    Theta holds ``(df - size - 1) / 2``, then the lower triangle of
    ``-0.5 * inverse(scale)``. Samples are rows of ``vech(X)``, so a batch is
    shaped (n, size * (size + 1) / 2) like every other family's.

    Attributes:
        theta: The natural parameter, which ``log_prob`` reads directly.
    """

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and the vech width dims."""
        self.dims = dims
        self.theta = theta
        canonical = Wishart.to_canonical(theta)
        scale = helpers.unvech(canonical[1:], helpers.vech_side(dims))
        # Validating every batch of samples took a quarter of a fit;
        # _constraints keeps theta valid instead.
        super().__init__(
            df=canonical[0],
            covariance_matrix=0.5 * (scale + scale.mT),
            validate_args=False,
        )

    def sample(self, sample_shape=ch.Size()) -> ch.Tensor:
        """Draw samples as vech rows, shaped ``sample_shape + (dims,)``."""
        return helpers.vech(super().sample(sample_shape))

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Return ``<theta, T(X)> - A(theta)``, cheaper than torch's solves."""
        size = helpers.vech_side(self.dims)
        log_partition = (
            0.5 * self.df * ch.logdet(self.covariance_matrix)
            + 0.5 * self.df * size * math.log(2.0)
            + ch.mvlgamma(self.df / 2.0, size)
        )
        return Wishart.calc_suff_stat(value) @ self.theta - log_partition

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return ``[log det X, vech(X) with off-diagonals doubled]``."""
        size = helpers.vech_side(x.size(-1))
        rows, cols = ch.tril_indices(size, size)
        # An off-diagonal stands for two entries of the full matrix.
        multiplicity = ch.where(rows == cols, 1.0, 2.0).to(x.dtype)
        logdet = ch.linalg.slogdet(helpers.unvech(x, size))[1]  # pylint: disable=not-callable
        return ch.cat([logdet[..., None], x * multiplicity], -1)

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert canonical ``[df, vech(scale)]`` to natural form."""
        size = helpers.vech_side(theta.numel() - 1)
        precision = helpers.unvech(theta[1:], size).inverse()
        return ch.cat([(theta[:1] - size - 1) / 2.0, helpers.vech(-0.5 * precision)])

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert natural form to canonical ``[df, vech(scale)]``."""
        size = helpers.vech_side(theta.numel() - 1)
        scale = (-2.0 * helpers.unvech(theta[1:], size)).inverse()
        return ch.cat([2.0 * theta[:1] + size + 1, helpers.vech(scale)])
