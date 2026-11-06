# Author: pstefanou12@
"""Truncated Poisson Distribution."""

from collections.abc import Callable
from typing import Optional

import torch.nn

from delphi.configs.distributions.truncated import (
    exponential_family as exponential_family_configs,
)
from delphi.exponential_family import poisson
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedPoisson(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Poisson distributions to be passed into trainer."""

    dist = poisson.Poisson
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
        """Initialize TruncatedPoisson.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, dims, alpha, hyperparameters=hyperparameters)

    def _calc_emp_model(self):
        """Initialize theta at the log of the empirical rate."""
        dataset_s = self.train_loader_.dataset.S  # pylint: disable=invalid-name
        # A coordinate observed only at zero would otherwise give log(0).
        emp = dataset_s.mean(0).clamp(min=1e-6)
        self.emp_theta = poisson.Poisson.to_natural(emp)
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    @property
    def best_rate_(self):
        """Return the best rate parameter estimate."""
        return poisson.Poisson.to_canonical(self.best_params)

    @property
    def final_rate_(self):
        """Return the final rate parameter estimate."""
        return poisson.Poisson.to_canonical(self.final_params)

    @property
    def ema_rate_(self):
        """Return the EMA rate parameter estimate."""
        return poisson.Poisson.to_canonical(self.ema_params)

    @property
    def avg_rate_(self):
        """Return the averaged rate parameter estimate."""
        return poisson.Poisson.to_canonical(self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated poisson distribution"
