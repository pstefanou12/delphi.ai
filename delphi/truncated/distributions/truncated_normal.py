# Author: pstefanou12@
"""Truncated normal distribution with oracle access (known truncation set)."""

from collections.abc import Callable
from typing import Optional

from delphi.truncated.distributions import truncated_multivariate_normal


class TruncatedNormal(truncated_multivariate_normal.TruncatedMultivariateNormal):
    """Truncated normal distribution with unknown variance."""

    estimate_suffixes = ("loc_", "scale_")

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
    def best_scale_(self):
        """Return the best standard deviation estimate, as torch.Normal's scale."""
        return self.best_covariance_matrix_.diagonal().sqrt()

    @property
    def final_scale_(self):
        """Return the final standard deviation estimate, as torch.Normal's scale."""
        return self.final_covariance_matrix_.diagonal().sqrt()

    @property
    def ema_scale_(self):
        """Return the EMA standard deviation estimate, as torch.Normal's scale."""
        return self.ema_covariance_matrix_.diagonal().sqrt()

    @property
    def avg_scale_(self):
        """Return the averaged standard deviation estimate, as torch.Normal's scale."""
        return self.avg_covariance_matrix_.diagonal().sqrt()

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated normal distribution"
