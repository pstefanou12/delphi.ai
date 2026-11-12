# Author: pstefanou12@
"""Univariate normal with known variance, truncated to an unknown set."""

from typing import Optional

import torch as ch

from delphi.truncated.distributions.unknown_truncated_multivariate_normal_known_covariance import (
    UnknownTruncationMultivariateNormalKnownCovariance,
)


class UnknownTruncationNormalKnownVariance(
    UnknownTruncationMultivariateNormalKnownCovariance
):
    """Mean of a normal with known variance truncated to an unknown set."""

    def __init__(
        self,
        covariance_matrix: ch.Tensor,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize UnknownTruncationNormalKnownVariance.

        Args:
            covariance_matrix: Known variance as a 1x1 matrix.
            alpha: Survival mass of the set in (0, 1), if known.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(covariance_matrix, alpha, hyperparameters=hyperparameters)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "normal distribution with known variance under unknown truncation"
