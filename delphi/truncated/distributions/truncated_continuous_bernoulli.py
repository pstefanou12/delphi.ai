# Author: pstefanou12@
"""Truncated continuous Bernoulli distribution."""

from collections.abc import Callable
from typing import Optional

from torch import nn

from delphi import configs
from delphi.exponential_family import continuous_bernoulli
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedContinuousBernoulli(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated continuous Bernoulli distributions."""

    dist = continuous_bernoulli.ContinuousBernoulli
    defaults = configs.WarmStartedExponentialFamilyDistributionConfig
    estimate_suffixes = ("probs_",)

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedContinuousBernoulli.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, dims, alpha)

    def _calc_emp_model(self):
        """Start at ``logit(mean)``, refined by the warm start.

        The mean is not ``probs`` here, but lies on the same side of 0.5.
        """
        mean = self.train_loader_.dataset.S.mean(0).clamp(1e-3, 1 - 1e-3)
        self.emp_theta = continuous_bernoulli.ContinuousBernoulli.to_natural(mean)
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    @property
    def best_probs_(self):
        """Return the best probs estimate."""
        return continuous_bernoulli.ContinuousBernoulli.to_canonical(self.best_params)

    @property
    def final_probs_(self):
        """Return the final probs estimate."""
        return continuous_bernoulli.ContinuousBernoulli.to_canonical(self.final_params)

    @property
    def ema_probs_(self):
        """Return the EMA probs estimate."""
        return continuous_bernoulli.ContinuousBernoulli.to_canonical(self.ema_params)

    @property
    def avg_probs_(self):
        """Return the averaged probs estimate."""
        return continuous_bernoulli.ContinuousBernoulli.to_canonical(self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated continuous bernoulli distribution"
