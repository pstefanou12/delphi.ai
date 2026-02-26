# Author: pstefanou12@
"""Multivariate normal distribution in natural parameterization."""

import torch as ch
from torch.distributions import MultivariateNormal


def calc_multi_norm_suff_stat_known_cov(x):
    """Return sufficient statistics for multivariate normal with known covariance."""
    return x


def calc_multi_norm_suff_stat(x):
    """Return sufficient statistics for multivariate normal."""
    return ch.cat([ch.bmm(x.unsqueeze(2), x.unsqueeze(1)).flatten(1), x], 1)


class ExponentialFamilyMultivariateNormalKnownCovariance(  # pylint: disable=abstract-method
    MultivariateNormal
):
    """Multivariate normal parameterized by natural parameters with known covariance."""

    def __init__(self, covariance_matrix: ch.Tensor, theta: ch.Tensor, dims: int):
        """Initialize with covariance matrix, natural parameter theta, and dimension."""
        self.dims = dims
        v = theta
        mu = (covariance_matrix @ v).view(self.dims)
        super().__init__(mu, covariance_matrix)


class ExponentialFamilyMultivariateNormal(MultivariateNormal):  # pylint: disable=abstract-method
    """Multivariate normal parameterized by natural parameters."""

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        T, v = theta[: self.dims**2], theta[self.dims**2 :]  # pylint: disable=invalid-name
        covariance_matrix = ch.inverse(-2 * T.view(self.dims, self.dims))
        mu = (covariance_matrix @ v).view(self.dims)
        super().__init__(mu, covariance_matrix)
