# Author: pstefanou12@
"""Multivariate normal distribution in natural parameterization."""

import torch as ch
import torch.distributions


class ExponentialFamilyMultivariateNormalKnownCovariance(  # pylint: disable=abstract-method
    ch.distributions.MultivariateNormal
):
    """Multivariate normal parameterized by natural parameters with known covariance."""

    def __init__(self, covariance_matrix: ch.Tensor, theta: ch.Tensor, dims: int):
        """Initialize with covariance matrix, natural parameter theta, and dimension."""
        self.dims = dims
        v = theta
        mu = (covariance_matrix @ v).view(self.dims)
        super().__init__(mu, covariance_matrix)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return sufficient statistics for multivariate normal with known covariance."""
        return x

    @staticmethod
    def to_natural(theta: ch.Tensor, covariance_matrix: ch.Tensor) -> ch.Tensor:
        """Convert canonical mean to natural parameters."""
        inv_cov = covariance_matrix.inverse()
        v = theta @ inv_cov  # pylint: disable=invalid-name
        return v.flatten()

    @staticmethod
    def to_canonical(theta: ch.Tensor, covariance_matrix: ch.Tensor) -> ch.Tensor:
        """Convert natural parameters to canonical mean."""
        loc = theta @ covariance_matrix
        return loc.flatten()


class ExponentialFamilyMultivariateNormal(  # pylint: disable=abstract-method
    ch.distributions.MultivariateNormal
):
    """Multivariate normal parameterized by natural parameters."""

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        T, v = theta[: self.dims**2], theta[self.dims**2 :]  # pylint: disable=invalid-name
        covariance_matrix = ch.inverse(-2 * T.view(self.dims, self.dims))
        mu = (covariance_matrix @ v).view(self.dims)
        super().__init__(mu, covariance_matrix)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return sufficient statistics for multivariate normal."""
        return ch.cat([ch.bmm(x.unsqueeze(2), x.unsqueeze(1)).flatten(1), x], 1)

    @staticmethod
    def to_natural(theta: ch.Tensor, dims: int) -> ch.Tensor:
        """Convert canonical parameters to natural form."""
        cov_matrix = theta[: dims**2].view(dims, dims)
        loc = theta[dims**2 :]
        mat_t = cov_matrix.inverse()  # pylint: disable=invalid-name
        v = loc @ mat_t  # pylint: disable=invalid-name
        return ch.cat([-0.5 * mat_t.flatten(), v.flatten()])

    @staticmethod
    def to_canonical(theta: ch.Tensor, dims: int) -> ch.Tensor:
        """Convert natural parameters to canonical form."""
        mat_t = theta[: dims**2].view(dims, dims)  # pylint: disable=invalid-name
        v = theta[dims**2 :]
        covariance_matrix = (-2 * mat_t).inverse()
        loc = v @ covariance_matrix
        return ch.cat([covariance_matrix.flatten(), loc.flatten()])
