# Author: pstefanou12@
"""Truncated normal distribution with oracle access (known truncation set)."""

from collections.abc import Callable
from typing import Optional

from delphi.truncated.distributions import known_truncated_multivariate_normal


class TruncatedNormal(
    known_truncated_multivariate_normal.KnownTruncationMultivariateNormal
):
    """Truncated normal distribution with unknown variance."""

    estimate_suffixes = ("loc_", "scale_")

    def __init__(
        self,
        phi: Optional[Callable],
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize TruncatedNormal.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            alpha: Survival probability lower bound.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, 1, alpha, hyperparameters=hyperparameters)

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
