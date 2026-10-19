# Author: pstefanou12@
"""Multivariate normal distribution in natural parameterization."""

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class MultivariateNormal(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.MultivariateNormal,
):
    """Multivariate normal parameterized by natural parameters.

    The matrix half of the natural parameter is held as the lower triangle of
    a symmetric matrix, ``dims * (dims + 1) / 2`` entries rather than
    ``dims ** 2``. Carrying the full matrix repeats every off-diagonal, which
    makes the coordinates of the sufficient statistic linearly dependent and
    ``Cov[T(x)]`` singular. Both the Daskalakis and the Karatapanis analyses
    are stated for a minimal representation and need a positive smallest
    eigenvalue, so the repeated entries are not merely wasteful.

    The statistic carries the multiplicity instead: an off-diagonal appears
    once, doubled, so the inner product with the parameter still equals the
    one taken over the whole matrix.
    """

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        size = self.natural_size(dims)
        mat_t = self.unvech(theta[:size], dims)  # pylint: disable=invalid-name
        v = theta[size:]
        covariance_matrix = ch.inverse(-2 * mat_t)
        mu = (covariance_matrix @ v).view(dims)
        super().__init__(mu, covariance_matrix)

    @staticmethod
    def natural_size(dims: int) -> int:
        """Return the number of entries the matrix half of theta occupies.

        Args:
            dims: Number of dimensions.

        Returns:
            ``dims * (dims + 1) / 2``.
        """
        return dims * (dims + 1) // 2

    @staticmethod
    def vech(matrix: ch.Tensor) -> ch.Tensor:
        """Return the lower triangle of a symmetric matrix as a vector.

        Args:
            matrix: Symmetric matrix of shape (dims, dims).

        Returns:
            Vector of length ``dims * (dims + 1) / 2``.
        """
        rows, cols = ch.tril_indices(matrix.size(-1), matrix.size(-1))
        return matrix[rows, cols]

    @staticmethod
    def unvech(vector: ch.Tensor, dims: int) -> ch.Tensor:
        """Rebuild a symmetric matrix from its lower triangle.

        Args:
            vector: Lower triangle, as produced by ``vech``.
            dims: Number of dimensions.

        Returns:
            Symmetric matrix of shape (dims, dims).
        """
        rows, cols = ch.tril_indices(dims, dims)
        matrix = ch.zeros(dims, dims, dtype=vector.dtype, device=vector.device)
        matrix[rows, cols] = vector
        return matrix + matrix.T - ch.diag(ch.diag(matrix))

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
        mat_t = cov_matrix.inverse()  # pylint: disable=invalid-name
        v = mat_t @ loc  # pylint: disable=invalid-name
        return ch.cat([MultivariateNormal.vech(-0.5 * mat_t), v.flatten()])

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
        size = MultivariateNormal.natural_size(dims)
        mat_t = MultivariateNormal.unvech(theta[:size], dims)  # pylint: disable=invalid-name
        v = theta[size:]
        covariance_matrix = (-2 * mat_t).inverse()
        loc = covariance_matrix @ v
        return ch.cat([covariance_matrix.flatten(), loc.flatten()])
