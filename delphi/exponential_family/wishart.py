# Author: pstefanou12@
"""Wishart distribution in natural parameterization, over vech rows."""

import math

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution
from delphi.utils import helpers


def side(width: int) -> int:
    """Return the side of the matrix whose lower triangle has ``width`` entries.

    Args:
        width: Length of the lower triangle, ``side * (side + 1) / 2``.

    Returns:
        The side length.
    """
    return int(round((math.sqrt(8 * width + 1) - 1) / 2))


def to_vech(matrix: ch.Tensor) -> ch.Tensor:
    """Return the lower triangles of a batch of matrices, shaped (..., width)."""
    rows, cols = ch.tril_indices(matrix.size(-1), matrix.size(-1))
    return matrix[..., rows, cols]


def from_vech(vector: ch.Tensor) -> ch.Tensor:
    """Rebuild a batch of symmetric matrices from their lower triangles."""
    size = side(vector.size(-1))
    rows, cols = ch.tril_indices(size, size)
    matrix = vector.new_zeros(vector.shape[:-1] + (size, size))
    matrix[..., rows, cols] = vector
    matrix[..., cols, rows] = vector
    return matrix


class Wishart(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.Wishart,
):
    """Wishart distribution whose samples are the lower triangles of matrices.

    Theta holds ``(df - size - 1) / 2``, then the lower triangle of
    ``-0.5 * inverse(scale)``. Samples are rows of ``vech(X)``, so a batch is
    shaped (n, size * (size + 1) / 2) like every other family's, and a
    truncation set reads them as such.
    """

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and the vech width dims."""
        self.dims = dims
        self.theta = theta
        canonical = Wishart.to_canonical(theta)
        scale = from_vech(canonical[1:])
        # A numerical inverse is symmetric only to rounding, which torch's
        # PositiveDefinite check rejects. Validation is off because it
        # checked every batch of samples, a quarter of a fit's time;
        # _constraints keeps theta valid instead.
        super().__init__(
            df=canonical[0],
            covariance_matrix=0.5 * (scale + scale.mT),
            validate_args=False,
        )

    def sample(self, sample_shape=ch.Size()) -> ch.Tensor:
        """Draw samples as vech rows, shaped ``sample_shape + (dims,)``."""
        return to_vech(super().sample(sample_shape))

    def log_prob(self, value: ch.Tensor) -> ch.Tensor:
        """Compute the log density of vech rows, as ``<theta, T(X)> - A``.

        torch's own solves against the scale once per sample, which made it
        the costliest call in a fit.
        """
        size = side(self.dims)
        df = self.df
        log_partition = (
            0.5 * df * ch.logdet(self.covariance_matrix)
            + 0.5 * df * size * math.log(2.0)
            + ch.mvlgamma(df / 2.0, size)
        )
        return Wishart.calc_suff_stat(value) @ self.theta - log_partition

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return ``[log det X, vech(X) with off-diagonals doubled]``."""
        size = side(x.size(-1))
        rows, cols = ch.tril_indices(size, size)
        # An off-diagonal stands for two entries of the full matrix.
        multiplicity = ch.where(rows == cols, 1.0, 2.0).to(x.dtype)
        logdet = ch.linalg.slogdet(from_vech(x))[1]  # pylint: disable=not-callable
        return ch.cat([logdet[..., None], x * multiplicity], -1)

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert canonical ``[df, vech(scale)]`` to natural form."""
        size = side(theta.numel() - 1)
        precision = from_vech(theta[1:]).inverse()
        return ch.cat([(theta[:1] - size - 1) / 2.0, helpers.vech(-0.5 * precision)])

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert natural form to canonical ``[df, vech(scale)]``."""
        size = side(theta.numel() - 1)
        scale = (-2.0 * from_vech(theta[1:])).inverse()
        return ch.cat([2.0 * theta[:1] + size + 1, helpers.vech(scale)])
