# Author: pstefanou12@
"""Truncated normal distribution with known variance."""

from collections.abc import Callable
from typing import Optional

import torch as ch

from delphi.truncated.distributions import (
    known_truncated_multivariate_normal_known_covariance,
)


class TruncatedNormalKnownVariance(
    known_truncated_multivariate_normal_known_covariance.KnownTruncationMultivariateNormalKnownCovariance
):
    """Truncated normal distribution with known variance."""

    estimate_suffixes = ("loc_",)

    def __init__(
        self,
        phi: Optional[Callable],
        covariance_matrix: ch.Tensor,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize TruncatedNormalKnownVariance.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            alpha: Survival probability lower bound.
            covariance_matrix: Known variance as a 1×1 matrix.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(
            phi, 1, covariance_matrix, alpha, hyperparameters=hyperparameters
        )

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated normal distribution known variance"
