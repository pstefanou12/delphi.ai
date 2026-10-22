# Author: pstefanou12@
"""Univariate normal estimation from samples truncated to an unknown set."""

from typing import Optional

import torch as ch

from delphi.truncated.distributions import unknown_truncated_multivariate_normal


class UnknownTruncationNormal(
    unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormal
):
    """Mean and standard deviation of a normal truncated to an unknown set."""

    estimate_suffixes = ("loc_", "scale_")

    def __init__(self, hyperparameters: dict, alpha: Optional[float] = None):
        """Initialize UnknownTruncationNormal.

        Args:
            hyperparameters: Hyperparameter dict.
            alpha: Survival mass of the set in (0, 1), if known.
        """
        super().__init__(hyperparameters, 1, alpha)

    @property
    def best_scale_(self) -> ch.Tensor:
        """Standard deviation at the best validation loss, as torch.Normal's scale."""
        return self.best_covariance_matrix_.diagonal().sqrt()

    @property
    def final_scale_(self) -> ch.Tensor:
        """Standard deviation at the last iterate, as torch.Normal's scale."""
        return self.final_covariance_matrix_.diagonal().sqrt()

    @property
    def ema_scale_(self) -> ch.Tensor:
        """Standard deviation at the moving average, as torch.Normal's scale."""
        return self.ema_covariance_matrix_.diagonal().sqrt()

    @property
    def avg_scale_(self) -> ch.Tensor:
        """Standard deviation at the averaged iterate, as torch.Normal's scale."""
        return self.avg_covariance_matrix_.diagonal().sqrt()

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "normal distribution under unknown truncation"


class UnknownTruncationNormalKnownVariance(
    unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormalKnownCovariance
):
    """Mean of a normal with known variance truncated to an unknown set."""

    def __init__(
        self,
        hyperparameters: dict,
        covariance_matrix: ch.Tensor,
        alpha: Optional[float] = None,
    ):
        """Initialize UnknownTruncationNormalKnownVariance.

        Args:
            hyperparameters: Hyperparameter dict.
            covariance_matrix: Known variance as a 1x1 matrix.
            alpha: Survival mass of the set in (0, 1), if known.
        """
        super().__init__(hyperparameters, 1, covariance_matrix, alpha)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "normal distribution with known variance under unknown truncation"
