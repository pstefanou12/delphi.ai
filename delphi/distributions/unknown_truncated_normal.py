# Author: pstefanou12@
"""
Truncated normal distribution without oracle access (ie. unknown truncation set).
"""

# pylint: disable=duplicate-code

from typing import Optional

import torch as ch

from delphi.distributions.unknown_truncated_multivariate_normal import (
    UnknownTruncationMultivariateNormalUnknownCovariance,
    UnknownTruncationMultivariateNormalKnownCovariance,
)
from delphi.utils.helpers import Parameters


class UnknownTruncationNormalKnownVariance(
    UnknownTruncationMultivariateNormalKnownCovariance
):
    """Truncated normal distribution with known variance and unknown truncation.

    Inherits all constructor arguments from
    UnknownTruncationMultivariateNormalKnownCovariance:
        args (Parameters): hyperparameter object
        k (int): number of nearest neighbors for oracle
        alpha (float): survival probability lower bound
        dims (int): number of dimensions
        covariance_matrix (Optional[Tensor]): known covariance matrix
    """

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return (
            "truncated normal distribution with unknown truncation and known variance"
        )


class UnknownTruncatedNormalUnknownVariance(
    UnknownTruncationMultivariateNormalUnknownCovariance
):
    """Truncated normal distribution with unknown variance and unknown truncation.

    Inherits all constructor arguments from
    UnknownTruncationMultivariateNormalUnknownCovariance:
        args (Parameters): hyperparameter object
        k (int): number of nearest neighbors for oracle
        alpha (float): survival probability lower bound
        dims (int): number of dimensions
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
        return "truncated normal distribution with unknown truncation"


def UnknownTruncationNormal(  # pylint: disable=invalid-name
    args: Parameters,
    k: int,
    alpha: float,
    dims: int,
    variance: Optional[ch.Tensor] = None,
):
    """
    Factory function for unknown-truncation normal distributions.

    Returns a known-variance model if variance is provided,
    otherwise returns an unknown-variance model.

    Args:
        args (Parameters): hyperparameter object
        k (int): number of nearest neighbours for the oracle
        alpha (float): survival probability lower bound
        dims (int): number of dimensions
        variance (Optional[Tensor]): known variance; if None, variance is estimated

    Returns:
        UnknownTruncationNormalKnownVariance if variance is provided, else
        UnknownTruncatedNormalUnknownVariance.
    """
    assert isinstance(args, Parameters), (
        "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
    )
    if variance is not None:
        return UnknownTruncationNormalKnownVariance(args, k, alpha, dims, variance)
    return UnknownTruncatedNormalUnknownVariance(args, k, alpha, dims)
