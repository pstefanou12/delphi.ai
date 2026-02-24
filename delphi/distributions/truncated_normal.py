"""
Truncated normal distribution with oracle access (ie. known truncation set).
"""

# pylint: disable=duplicate-code

from typing import Callable, Optional

import torch as ch

from .truncated_multivariate_normal import (
    TruncatedMultivariateNormalKnownCovariance,
    TruncatedMultivariateNormalUnknownCovariance,
)
from ..utils.helpers import Parameters
from ..utils.defaults import check_and_fill_args, TRUNC_MULTI_NORM_DEFAULTS


class TruncatedNormalKnownCovariance(TruncatedMultivariateNormalKnownCovariance):
    """
    Truncated normal distribution class with known covariance.

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
        return "truncated normal distribution known covariance"


class TruncatedNormalUnknownVariance(TruncatedMultivariateNormalUnknownCovariance):
    """
    Truncated normal distribution class with unknown variance.

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
        """
        Returns the best covariance matrix estimate for the multivariate normal
        distribution based off of the loss function.
        """
        return self.best_params[: self.dims**2].view(self.dims, self.dims)

    @property
    def final_variance_(self):
        """
        Returns the final covariance matrix estimate for the multivariate normal
        distribution based off of the loss function.
        """
        self.final_params[: self.dims**2].view(self.dims, self.dims)

    @property
    def ema_variance_(self):
        """
        Returns the ema covariance matrix estimate for the multivariate normal
        distribution based off of the loss function.
        """
        return self.ema_params[: self.dims**2].view(self.dims, self.dims)

    @property
    def avg_variance_(self):
        """
        Returns the avg covariance matrix estimate for the multivariate normal
        distribution based off of the loss function.
        """
        return self.avg_params[: self.dims**2].view(self.dims, self.dims)

    def __str__(self):
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
    """
    assert isinstance(args, Parameters), (
        "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
    )
    args = check_and_fill_args(args, TRUNC_MULTI_NORM_DEFAULTS)
    if variance is not None:
        return TruncatedNormalKnownCovariance(args, phi, alpha, dims, variance, sampler)
    return TruncatedNormalUnknownVariance(args, phi, alpha, dims, sampler)
