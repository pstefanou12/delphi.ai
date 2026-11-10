# Author: pstefanou12@
"""Truncated chi-squared distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
import torch.nn

from delphi.configs.distributions.truncated import (
    exponential_family as exponential_family_configs,
)
from delphi.exponential_family import chi2
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedChi2(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated chi-squared distributions."""

    dist = chi2.Chi2
    defaults = exponential_family_configs.WarmStartedExponentialFamilyDistributionConfig
    estimate_suffixes = ("df_",)

    def __init__(
        self,
        phi: Optional[Callable],
        dims: int,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize TruncatedChi2.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, dims, alpha, hyperparameters=hyperparameters)

    def _calc_emp_model(self):
        """Start at the moment estimate ``df = mean``, refined by the warm start."""
        mean = self.train_loader_.dataset.S.mean(0)
        self.emp_theta = chi2.Chi2.to_natural(mean.clamp(min=1e-3))
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta above -1, so the degrees of freedom are positive."""
        return ch.clamp(theta, min=-1.0 + 1e-6)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated chi-squared distribution"
