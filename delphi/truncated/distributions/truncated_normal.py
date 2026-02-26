# Author: pstefanou12@
"""
Truncated normal distribution with oracle access (ie. known truncation set).
"""

# pylint: disable=duplicate-code

from typing import Callable, Optional

import torch as ch

from delphi.distributions.truncated_multivariate_normal import (
    TruncatedMultivariateNormalKnownCovariance,
    TruncatedMultivariateNormalUnknownCovariance,
)
from delphi.utils.helpers import Parameters
from delphi.utils.defaults import check_and_fill_args, TRUNC_MULTI_NORM_DEFAULTS


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


class TruncatedNormalUnknownVariance(TruncatedMultivariateNormalUnknownCovariance):
    """Truncated normal distribution with unknown variance.

    Inherits all constructor arguments from
    TruncatedMultivariateNormalUnknownCovariance:
        args (Parameters): hyperparameter object
        phi (Callable): truncation set oracle
        alpha (float): survival probability lower bound
        dims (int): number of dimensions
        sampler (Callable): optional sampler override
    """

    @property
    def best_variance_(self):
        """Best variance estimate based on lowest training loss."""
        return self.best_params[: self.dims**2].view(self.dims, self.dims)

    @property
    def final_variance_(self):
        """Final variance estimate at the end of training."""
        self.final_params[: self.dims**2].view(self.dims, self.dims)

    @property
    def ema_variance_(self):
        """Exponential moving-average variance estimate."""
        return self.ema_params[: self.dims**2].view(self.dims, self.dims)

    @property
    def avg_variance_(self):
        """Running-average variance estimate."""
        return self.avg_params[: self.dims**2].view(self.dims, self.dims)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated normal distribution"


def TruncatedNormal(  # pylint: disable=invalid-name
    args: Parameters,
    phi: Callable,
    alpha: float,
    dims: int,
    variance: Optional[ch.Tensor] = None,
    sampler: Callable = None,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """
    Factory function for truncated normal distributions.

    Returns a known-variance model if variance is provided,
    otherwise returns an unknown-variance model.

    Args:
        args (Parameters): hyperparameter object
        phi (Callable): truncation set oracle
        alpha (float): survival probability lower bound
        dims (int): number of dimensions
        variance (Optional[Tensor]): known variance; if None, variance is estimated
        sampler (Callable): optional sampler override

    Returns:
        TruncatedNormalKnownCovariance if variance is provided, else
        TruncatedNormalUnknownVariance.
    """
    assert isinstance(args, Parameters), (
        "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
    )
    args = check_and_fill_args(args, TRUNC_MULTI_NORM_DEFAULTS)
    if variance is not None:
        return TruncatedNormalKnownCovariance(args, phi, alpha, dims, variance, sampler)
    return TruncatedNormalUnknownVariance(args, phi, alpha, dims, sampler)
