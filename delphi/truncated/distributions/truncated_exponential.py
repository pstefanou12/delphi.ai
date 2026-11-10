# Author: pstefanou12@
"""Truncated Exponential Distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
import torch.nn

from delphi.configs.distributions.truncated import (
    exponential_family as exponential_family_configs,
)
from delphi.exponential_family import exponential
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedExponential(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated exponential distributions to be passed into trainer."""

    dist = exponential.Exponential
    defaults = exponential_family_configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("rate_",)

    def __init__(
        self,
        phi: Optional[Callable],
        dims: int,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize TruncatedExponential.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, dims, alpha, hyperparameters=hyperparameters)

    def _calc_emp_model(self):
        """Initialize theta at the natural parameter corresponding to the empirical rate."""
        dataset_s = self.train_loader_.dataset.S  # pylint: disable=invalid-name
        emp_rate = 1.0 / dataset_s.mean(0)
        self.emp_theta = exponential.Exponential.to_natural(emp_rate)
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta to be strictly negative."""
        return ch.clamp(theta, max=-1e-6)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated exponential distribution"
