# Author: pstefanou12@
"""Truncated normal distribution with oracle access (known truncation set)."""

from collections.abc import Callable
from typing import Optional

from delphi.truncated.distributions import truncated_multivariate_normal


class TruncatedNormal(truncated_multivariate_normal.TruncatedMultivariateNormal):
    """Truncated normal distribution with unknown variance."""

    estimate_suffixes = ("loc_", "variance_")

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedNormal.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, 1, alpha)

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
