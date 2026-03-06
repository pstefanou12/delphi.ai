# Author: pstefanou12@
"""Truncated normal distribution without oracle access (unknown truncation set)."""

# pylint: disable=duplicate-code

import torch as ch

from delphi.truncated.distributions import unknown_truncated_multivariate_normal
from delphi.utils import helpers


class UnknownTruncationNormalKnownVariance(
    unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormalKnownCovariance
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
    unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormalUnknownCovariance
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
    args: helpers.Parameters,
    k: int,
    alpha: float,
    dims: int,
    variance: ch.Tensor | None = None,
):
    """Factory function for unknown-truncation normal distributions.

    Returns a known-variance model if variance is provided,
    otherwise returns an unknown-variance model.

    Args:
        args: Hyperparameter object.
        k: Number of nearest neighbours for the oracle.
        alpha: Survival probability lower bound.
        dims: Number of dimensions.
        variance: Known variance; if None, variance is estimated.

    Returns:
        UnknownTruncationNormalKnownVariance if variance is provided, else
        UnknownTruncatedNormalUnknownVariance.

    Raises:
        TypeError: If args is not a Parameters instance.
    """
    if not isinstance(args, helpers.Parameters):
        raise TypeError(f"args is type {type(args).__name__}; expected Parameters.")
    if variance is not None:
        return UnknownTruncationNormalKnownVariance(args, k, alpha, dims, variance)
    return UnknownTruncatedNormalUnknownVariance(args, k, alpha, dims)
