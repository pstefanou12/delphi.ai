# Author: pstefanou12@
"""Truncated Maxwell distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import maxwell
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedMaxwell(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Maxwell distributions."""

    dist = maxwell.Maxwell
    defaults = configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("scale_",)

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedMaxwell.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, dims, alpha)

    def _calc_emp_model(self):
        """Initialize theta at the MLE scale ``sqrt(mean(x^2) / 3)``."""
        dataset_s = self.train_loader_.dataset.S
        self.emp_theta = maxwell.Maxwell.to_natural(
            ch.sqrt((dataset_s**2).mean(0) / 3.0)
        )
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta to be strictly negative, so the scale is real."""
        return ch.clamp(theta, max=-1e-6)

    @property
    def best_scale_(self):
        """Return the best scale estimate."""
        return maxwell.Maxwell.to_canonical(self.best_params)

    @property
    def final_scale_(self):
        """Return the final scale estimate."""
        return maxwell.Maxwell.to_canonical(self.final_params)

    @property
    def ema_scale_(self):
        """Return the EMA scale estimate."""
        return maxwell.Maxwell.to_canonical(self.ema_params)

    @property
    def avg_scale_(self):
        """Return the averaged scale estimate."""
        return maxwell.Maxwell.to_canonical(self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated maxwell distribution"
