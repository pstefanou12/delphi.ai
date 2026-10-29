# Author: pstefanou12@
"""Truncated logarithmic (log-series) distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import logarithmic
from delphi.truncated.distributions import truncated_exponential_family_distributions

# Halvings of (0, 1) for the MLE, which leave an error below float64's.
BISECTION_STEPS = 60


class TruncatedLogarithmic(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated logarithmic distributions."""

    dist = logarithmic.Logarithmic
    defaults = configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("probs_",)

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedLogarithmic.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, dims, alpha)

    def _calc_emp_model(self):
        """Initialize theta at the MLE, the probs whose mean is the sample's.

        The mean ``-p / ((1 - p) log(1 - p))`` increases in p but has no
        closed-form inverse, so it is bisected.
        """
        sample_mean = self.train_loader_.dataset.S.mean(0)
        low, high = ch.zeros_like(sample_mean), ch.ones_like(sample_mean)
        for _ in range(BISECTION_STEPS):
            probs = (low + high) / 2.0
            mean = logarithmic.Logarithmic(ch.log(probs), self.dims).mean
            low = ch.where(mean < sample_mean, probs, low)
            high = ch.where(mean < sample_mean, high, probs)
        self.emp_theta = logarithmic.Logarithmic.to_natural((low + high) / 2.0)
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta to be strictly negative, so the probs stay below one."""
        return ch.clamp(theta, max=-1e-6)

    @property
    def best_probs_(self):
        """Return the best probs estimate."""
        return logarithmic.Logarithmic.to_canonical(self.best_params)

    @property
    def final_probs_(self):
        """Return the final probs estimate."""
        return logarithmic.Logarithmic.to_canonical(self.final_params)

    @property
    def ema_probs_(self):
        """Return the EMA probs estimate."""
        return logarithmic.Logarithmic.to_canonical(self.ema_params)

    @property
    def avg_probs_(self):
        """Return the averaged probs estimate."""
        return logarithmic.Logarithmic.to_canonical(self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated logarithmic distribution"
