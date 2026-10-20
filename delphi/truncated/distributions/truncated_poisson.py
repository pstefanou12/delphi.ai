# Author: pstefanou12@
"""Truncated Poisson Distribution."""

from collections.abc import Callable
from typing import Optional

from torch import nn

from delphi import configs
from delphi.exponential_family import poisson
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedPoisson(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Poisson distributions to be passed into trainer."""

    dist = poisson.Poisson
    defaults = configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("rate_",)

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedPoisson.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
        """
        super().__init__(hyperparameters, phi, dims, alpha)

    def _calc_emp_model(self):
        """Initialize theta at the log of the empirical rate."""
        dataset_s = self.train_loader_.dataset.S  # pylint: disable=invalid-name
        # A coordinate observed only at zero would otherwise give log(0).
        emp = dataset_s.mean(0).clamp(min=1e-6)
        self.emp_theta = poisson.Poisson.to_natural(emp)
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

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
