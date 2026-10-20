# Author: pstefanou12@
"""Truncated geometric distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import geometric
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedGeometric(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated geometric distributions."""

    dist = geometric.Geometric
    defaults = configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("probs_",)

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedGeometric.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, dims, alpha)

    def _calc_emp_model(self):
        """Initialize theta at the MLE ``probs = 1 / (1 + mean)``."""
        mean = self.train_loader_.dataset.S.mean(0)
        emp = (1.0 / (1.0 + mean)).clamp(1e-6, 1 - 1e-6)
        self.emp_theta = geometric.Geometric.to_natural(emp)
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta to be strictly negative, so probs lies in (0, 1)."""
        return ch.clamp(theta, max=-1e-6)

    @property
    def best_probs_(self):
        """Return the best success probability estimate."""
        return geometric.Geometric.to_canonical(self.best_params)

    @property
    def final_probs_(self):
        """Return the final success probability estimate."""
        return geometric.Geometric.to_canonical(self.final_params)

    @property
    def ema_probs_(self):
        """Return the EMA success probability estimate."""
        return geometric.Geometric.to_canonical(self.ema_params)

    @property
    def avg_probs_(self):
        """Return the averaged success probability estimate."""
        return geometric.Geometric.to_canonical(self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated geometric distribution"
