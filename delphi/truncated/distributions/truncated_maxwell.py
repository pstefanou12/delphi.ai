# Author: pstefanou12@
"""Truncated Maxwell distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
import torch.nn

from delphi.configs.distributions.truncated import (
    exponential_family as exponential_family_configs,
)
from delphi.exponential_family import maxwell
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedMaxwell(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Maxwell distributions."""

    dist = maxwell.Maxwell
    defaults = exponential_family_configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("scale_",)

    def __init__(
        self,
        phi: Optional[Callable],
        dims: int,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize TruncatedMaxwell.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, dims, alpha, hyperparameters=hyperparameters)

    def _calc_emp_model(self):
        """Initialize theta at the MLE scale ``sqrt(mean(x^2) / 3)``."""
        dataset_s = self.train_loader_.dataset.S
        self.emp_theta = maxwell.Maxwell.to_natural(
            ch.sqrt((dataset_s**2).mean(0) / 3.0)
        )
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta to be strictly negative, so the scale is real."""
        return ch.clamp(theta, max=-1e-6)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated maxwell distribution"
