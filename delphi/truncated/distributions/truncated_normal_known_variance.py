# Author: pstefanou12@
"""Truncated normal distribution with known variance."""

from collections.abc import Callable

import torch as ch

from delphi.truncated.distributions import (
    truncated_multivariate_normal_known_covariance,
)
from delphi.utils import configs


class TruncatedNormalKnownVariance(
    truncated_multivariate_normal_known_covariance.TruncatedMultivariateNormalKnownCovariance
):
    """Truncated normal distribution with known variance."""

    def __init__(
        self,
        args: dict | configs.TruncatedMultivariateNormalConfig,
        phi: Callable,
        alpha: float,
        covariance_matrix: ch.Tensor | None,
        sampler: Callable = None,
    ):
        """Initialize TruncatedNormalKnownVariance.

        Args:
            args: Hyperparameter dict or Pydantic config.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            covariance_matrix: Known variance as a 1×1 matrix.
            sampler: Optional sampler override.
        """
        super().__init__(args, phi, alpha, 1, covariance_matrix, sampler)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated normal distribution known variance"
