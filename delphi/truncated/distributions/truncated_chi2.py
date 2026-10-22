# Author: pstefanou12@
"""Truncated chi-squared distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import chi2
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedChi2(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated chi-squared distributions."""

    dist = chi2.Chi2
    defaults = configs.WarmStartedExponentialFamilyDistributionConfig
    estimate_suffixes = ("df_",)

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedChi2.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, dims, alpha)

    def _calc_emp_model(self):
        """Start at the moment estimate ``df = mean``, refined by the warm start."""
        mean = self.train_loader_.dataset.S.mean(0)
        self.emp_theta = chi2.Chi2.to_natural(mean.clamp(min=1e-3))
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta above -1, so the degrees of freedom are positive."""
        return ch.clamp(theta, min=-1.0 + 1e-6)

    @property
    def best_df_(self):
        """Return the best degrees of freedom estimate."""
        return chi2.Chi2.to_canonical(self.best_params)

    @property
    def final_df_(self):
        """Return the final degrees of freedom estimate."""
        return chi2.Chi2.to_canonical(self.final_params)

    @property
    def ema_df_(self):
        """Return the EMA degrees of freedom estimate."""
        return chi2.Chi2.to_canonical(self.ema_params)

    @property
    def avg_df_(self):
        """Return the averaged degrees of freedom estimate."""
        return chi2.Chi2.to_canonical(self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated chi-squared distribution"
