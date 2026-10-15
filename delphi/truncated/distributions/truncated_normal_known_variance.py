# Author: pstefanou12@
"""Truncated normal distribution with known variance."""

from collections.abc import Callable
from typing import Optional

import torch as ch

from delphi.truncated.distributions import (
    truncated_multivariate_normal_known_covariance,
)


class TruncatedNormalKnownVariance(
    truncated_multivariate_normal_known_covariance.TruncatedMultivariateNormalKnownCovariance
):
    """Truncated normal distribution with known variance."""

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        covariance_matrix: ch.Tensor,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedNormalKnownVariance.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            covariance_matrix: Known variance as a 1×1 matrix.
        """
        super().__init__(hyperparameters, phi, 1, covariance_matrix, alpha)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated normal distribution known variance"
