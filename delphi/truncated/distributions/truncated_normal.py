# Author: pstefanou12@
"""Truncated normal distribution with oracle access (known truncation set)."""

from delphi.truncated.distributions.truncated_multivariate_normal import (
    TruncatedMultivariateNormal,
)


class TruncatedNormal(TruncatedMultivariateNormal):
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
