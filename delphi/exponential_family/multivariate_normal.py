# Author: pstefanou12@
"""Multivariate normal distribution in natural parameterization."""

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution
from delphi.utils import helpers


class MultivariateNormal(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.MultivariateNormal,
):
    """Multivariate normal parameterized by natural parameters.

    The matrix half of theta is the lower triangle of ``-0.5 * precision``.
    Carrying the full matrix would repeat every off-diagonal, leaving
    ``Cov[T(x)]`` singular.
    """

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        size = helpers.vech_size(dims)
        mat_t = helpers.unvech(theta[:size], dims)
        v = theta[size:]
        covariance_matrix = ch.inverse(-2 * mat_t)
        # A numerical inverse is symmetric only to rounding, which torch's
        # PositiveDefinite check rejects.
        covariance_matrix = 0.5 * (covariance_matrix + covariance_matrix.T)
        mu = (covariance_matrix @ v).view(dims)
        super().__init__(mu, covariance_matrix)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return sufficient statistics for multivariate normal.

        Args:
            x: Samples of shape (n, dims).

        Returns:
            ``[vech(xx^T) with off-diagonals doubled, x]``, shaped
            (n, dims * (dims + 1) / 2 + dims).
        """
        dims = x.size(-1)
        rows, cols = ch.tril_indices(dims, dims)
        outer = ch.bmm(x.unsqueeze(2), x.unsqueeze(1))
        # An off-diagonal stands for two entries of the full matrix.
        multiplicity = ch.where(rows == cols, 1.0, 2.0).to(x.dtype)
        return ch.cat([outer[:, rows, cols] * multiplicity, x], 1)

    @staticmethod
    def to_natural(theta: ch.Tensor, dims: int) -> ch.Tensor:
        """Convert canonical parameters to natural form.

        Args:
            theta: ``[covariance_matrix.flatten(), loc]``.
            dims: Number of dimensions.

        Returns:
            ``[vech(-0.5 * precision), precision @ loc]``.
        """
        cov_matrix = theta[: dims**2].view(dims, dims)
        loc = theta[dims**2 :]
        mat_t = cov_matrix.inverse()
        v = mat_t @ loc
        return ch.cat([helpers.vech(-0.5 * mat_t), v.flatten()])

    @staticmethod
    def to_canonical(theta: ch.Tensor, dims: int) -> ch.Tensor:
        """Convert natural parameters to canonical form.

        Args:
            theta: ``[vech(-0.5 * precision), precision @ loc]``.
            dims: Number of dimensions.

        Returns:
            ``[covariance_matrix.flatten(), loc]``, the full matrix, since
            the canonical form is a container rather than a statistic.
        """
        size = helpers.vech_size(dims)
        mat_t = helpers.unvech(theta[:size], dims)
        v = theta[size:]
        covariance_matrix = (-2 * mat_t).inverse()
        covariance_matrix = 0.5 * (covariance_matrix + covariance_matrix.T)
        loc = covariance_matrix @ v
        return ch.cat([covariance_matrix.flatten(), loc.flatten()])
