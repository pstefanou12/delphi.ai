# Author: pstefanou12@
"""Truncated continuous Bernoulli distribution."""

from collections.abc import Callable
from typing import Optional

import torch.nn

from delphi.configs.distributions.truncated import (
    exponential_family as exponential_family_configs,
)
from delphi.exponential_family import continuous_bernoulli
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedContinuousBernoulli(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated continuous Bernoulli distributions."""

    family = continuous_bernoulli.ContinuousBernoulli
    defaults = exponential_family_configs.WarmStartedExponentialFamilyDistributionConfig
    estimate_suffixes = ("probs_",)

    def __init__(
        self,
        phi: Optional[Callable],
        dims: int,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize TruncatedContinuousBernoulli.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, dims, alpha, hyperparameters=hyperparameters)

    def _calc_emp_model(self):
        """Start at ``logit(mean)``, refined by the warm start.

        The mean is not ``probs`` here, but lies on the same side of 0.5.
        """
        mean = self.train_loader_.dataset.S.mean(0).clamp(1e-3, 1 - 1e-3)
        self.emp_theta = continuous_bernoulli.ContinuousBernoulli.to_natural(mean)
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated continuous bernoulli distribution"
