# Author: pstefanou12@
"""Truncated normal distribution with oracle access (known truncation set)."""

from collections.abc import Callable

from delphi.truncated.distributions import truncated_multivariate_normal
from delphi.utils import configs


class TruncatedNormal(truncated_multivariate_normal.TruncatedMultivariateNormal):
    """Truncated normal distribution with unknown variance."""

    def __init__(
        self,
        args: dict | configs.TruncatedMultivariateNormalConfig,
        phi: Callable,
        alpha: float,
        sampler: Callable = None,
    ):
        """Initialize TruncatedNormal.

        Args:
            args: Hyperparameter dict or Pydantic config.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            sampler: Optional sampler override.
        """
        super().__init__(args, phi, alpha, 1, sampler)

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
