# Author: pstefanou12@
"""Multivariate normal distribution in natural parameterization."""

import torch as ch
from torch import Tensor
from torch.distributions import MultivariateNormal


class ExponentialFamilyMultivariateNormalKnownCovariance(  # pylint: disable=abstract-method
    MultivariateNormal
):
    """Multivariate normal parameterized by natural parameters with known covariance."""

    def __init__(self, covariance_matrix: Tensor, theta: Tensor, dims: int):
        """Initialize with covariance matrix, natural parameter theta, and dimension."""
        self.dims = dims
        v = theta
        mu = (covariance_matrix @ v).view(self.dims)
        super().__init__(mu, covariance_matrix)

    @staticmethod
    def calc_suff_stat(x: Tensor) -> Tensor:
        """Return sufficient statistics for multivariate normal with known covariance."""
        return x

    @staticmethod
    def to_natural(theta: Tensor, covariance_matrix: Tensor) -> Tensor:
        """Convert canonical mean to natural parameters."""
        inv_cov = covariance_matrix.inverse()
        v = theta @ inv_cov  # pylint: disable=invalid-name
        return v.flatten()

    @staticmethod
    def to_canonical(theta: Tensor, covariance_matrix: Tensor) -> Tensor:
        """Convert natural parameters to canonical mean."""
        loc = theta @ covariance_matrix
        return loc.flatten()


class ExponentialFamilyMultivariateNormal(MultivariateNormal):  # pylint: disable=abstract-method
    """Multivariate normal parameterized by natural parameters."""

    def __init__(self, theta: Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        T, v = theta[: self.dims**2], theta[self.dims**2 :]  # pylint: disable=invalid-name
        covariance_matrix = ch.inverse(-2 * T.view(self.dims, self.dims))
        mu = (covariance_matrix @ v).view(self.dims)
        super().__init__(mu, covariance_matrix)

    @staticmethod
    def calc_suff_stat(x: Tensor) -> Tensor:
        """Return sufficient statistics for multivariate normal."""
        return ch.cat([ch.bmm(x.unsqueeze(2), x.unsqueeze(1)).flatten(1), x], 1)

    @staticmethod
    def to_natural(theta: Tensor, dims: int) -> Tensor:
        """Convert canonical parameters to natural form."""
        cov_matrix = theta[: dims**2].view(dims, dims)
        loc = theta[dims**2 :]
        mat_t = cov_matrix.inverse()  # pylint: disable=invalid-name
        v = loc @ mat_t  # pylint: disable=invalid-name
        return ch.cat([-0.5 * mat_t.flatten(), v.flatten()])

    @staticmethod
    def to_canonical(theta: Tensor, dims: int) -> Tensor:
        """Convert natural parameters to canonical form."""
        mat_t = theta[: dims**2].view(dims, dims)  # pylint: disable=invalid-name
        v = theta[dims**2 :]
        covariance_matrix = (-2 * mat_t).inverse()
        loc = v @ covariance_matrix
        return ch.cat([covariance_matrix.flatten(), loc.flatten()])
