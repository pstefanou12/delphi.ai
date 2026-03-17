# Author: pstefanou12@
"""Multivariate normal distribution with known covariance in natural parameterization."""

import torch as ch
from torch import distributions

from delphi.exponential_family import exponential_family_distribution


class MultivariateNormalKnownCovariance(
    exponential_family_distribution.ExponentialFamilyDistribution,
    distributions.MultivariateNormal,
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
