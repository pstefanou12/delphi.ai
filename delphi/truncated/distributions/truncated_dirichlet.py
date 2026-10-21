# Author: pstefanou12@
"""Truncated Dirichlet distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import dirichlet
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedDirichlet(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Dirichlet distributions."""

    dist = dirichlet.Dirichlet
    defaults = configs.WarmStartedExponentialFamilyDistributionConfig
    estimate_suffixes = ("concentration_",)
    # The categories share one normalizer, so they are coupled.
    independent_coordinates = False

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedDirichlet.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of categories.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, dims, alpha)

    def _calc_emp_model(self):
        """Start at the moment estimate, refined by the warm start."""
        dataset_s = self.train_loader_.dataset.S
        mean, var = dataset_s.mean(0), dataset_s.var(0)
        # Every category gives the total concentration; average them.
        total = (mean * (1 - mean) / var - 1).mean()
        self.emp_theta = dirichlet.Dirichlet.to_natural((mean * total).clamp(min=1e-3))
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta above -1, so every concentration is positive."""
        return ch.clamp(theta, min=-1.0 + 1e-6)

    @property
    def best_concentration_(self):
        """Return the best concentration estimate."""
        return dirichlet.Dirichlet.to_canonical(self.best_params)

    @property
    def final_concentration_(self):
        """Return the final concentration estimate."""
        return dirichlet.Dirichlet.to_canonical(self.final_params)

    @property
    def ema_concentration_(self):
        """Return the EMA concentration estimate."""
        return dirichlet.Dirichlet.to_canonical(self.ema_params)

    @property
    def avg_concentration_(self):
        """Return the averaged concentration estimate."""
        return dirichlet.Dirichlet.to_canonical(self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated dirichlet distribution"
