# Author: pstefanou12@
"""Truncated Rayleigh distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
import torch.nn

from delphi.configs.distributions.truncated import (
    exponential_family as exponential_family_configs,
)
from delphi.exponential_family import rayleigh
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedRayleigh(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Rayleigh distributions."""

    dist = rayleigh.Rayleigh
    defaults = exponential_family_configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("scale_",)

    def __init__(
        self,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize TruncatedRayleigh.

        Args:
            phi: Truncation set oracle.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, dims, alpha, hyperparameters=hyperparameters)

    def _calc_emp_model(self):
        """Initialize theta at the MLE scale ``sqrt(mean(x^2) / 2)``."""
        dataset_s = self.train_loader_.dataset.S
        self.emp_theta = rayleigh.Rayleigh.to_natural(
            ch.sqrt((dataset_s**2).mean(0) / 2.0)
        )
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta to be strictly negative, so the scale is real."""
        return ch.clamp(theta, max=-1e-6)

    @property
    def best_scale_(self):
        """Return the best scale estimate."""
        return rayleigh.Rayleigh.to_canonical(self.best_params)

    @property
    def final_scale_(self):
        """Return the final scale estimate."""
        return rayleigh.Rayleigh.to_canonical(self.final_params)

    @property
    def ema_scale_(self):
        """Return the EMA scale estimate."""
        return rayleigh.Rayleigh.to_canonical(self.ema_params)

    @property
    def avg_scale_(self):
        """Return the averaged scale estimate."""
        return rayleigh.Rayleigh.to_canonical(self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated rayleigh distribution"
