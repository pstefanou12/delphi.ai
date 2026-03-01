# Author: pstefanou12@
"""Truncated normal distribution with oracle access (known truncation set)."""

# pylint: disable=duplicate-code

from collections.abc import Callable

import torch as ch

from delphi.truncated.distributions.truncated_multivariate_normal_known_covariance import (
    TruncatedMultivariateNormalConfig,
    TruncatedMultivariateNormalKnownCovariance,
)
from delphi.truncated.distributions.truncated_multivariate_normal import (
    TruncatedMultivariateNormal,
)
from delphi.utils.configs import make_config


class TruncatedNormalKnownCovariance(TruncatedMultivariateNormalKnownCovariance):
    """Truncated normal distribution with known covariance.

    Inherits all constructor arguments from
    TruncatedMultivariateNormalKnownCovariance:
        args (Parameters): hyperparameter object
        phi (Callable): truncation set oracle
        alpha (float): survival probability lower bound
        dims (int): number of dimensions
        covariance_matrix (Optional[Tensor]): known covariance matrix
        sampler (Callable): optional sampler override
    """

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated normal distribution known covariance"


class TruncatedNormalUnknownVariance(TruncatedMultivariateNormal):
    """Truncated normal distribution with unknown variance.

    Inherits all constructor arguments from TruncatedMultivariateNormal:
        args (Parameters): hyperparameter object
        phi (Callable): truncation set oracle
        alpha (float): survival probability lower bound
        dims (int): number of dimensions
        sampler (Callable): optional sampler override
    """

    @property
    def best_variance_(self):
        """Best variance estimate based on lowest training loss."""
        return self.best_covariance_matrix_

    @property
    def final_variance_(self):
        """Final variance estimate at the end of training."""
        return self.final_covariance_matrix_

    @property
    def ema_variance_(self):
        """Exponential moving-average variance estimate."""
        return self.ema_covariance_matrix_

    @property
    def avg_variance_(self):
        """Running-average variance estimate."""
        return self.avg_covariance_matrix_

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated normal distribution"


def TruncatedNormal(  # pylint: disable=invalid-name
    args: dict | TruncatedMultivariateNormalConfig,
    phi: Callable,
    alpha: float,
    dims: int,
    variance: ch.Tensor | None = None,
    sampler: Callable = None,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Factory function for truncated normal distributions.

    Returns a known-variance model if variance is provided,
    otherwise returns an unknown-variance model.

    Args:
        args: Hyperparameter dict or Pydantic config.
        phi: Truncation set oracle.
        alpha: Survival probability lower bound.
        dims: Number of dimensions.
        variance: Known variance; if None, variance is estimated.
        sampler: Optional sampler override.

    Returns:
        TruncatedNormalKnownCovariance if variance is provided, else
        TruncatedNormalUnknownVariance.
    """
    args = make_config(args, TruncatedMultivariateNormalConfig)
    if variance is not None:
        return TruncatedNormalKnownCovariance(args, phi, alpha, dims, variance, sampler)
    return TruncatedNormalUnknownVariance(args, phi, alpha, dims, sampler)
