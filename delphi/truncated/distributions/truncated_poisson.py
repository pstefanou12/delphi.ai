# Author: pstefanou12@
"""Truncated Poisson Distribution."""

from collections.abc import Callable

from delphi import configs
from delphi.exponential_family import poisson
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedPoisson(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Poisson distributions to be passed into trainer."""

    dist = poisson.Poisson
    defaults = configs.TruncatedExponentialFamilyDistributionConfig

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        alpha: float,
        dims: int,
    ):
        """Initialize TruncatedPoisson.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
        """
        super().__init__(hyperparameters, phi, alpha, dims)

    @property
    def best_lambda_(self):
        """Return the best rate parameter estimate."""
        return poisson.Poisson.to_canonical(self.best_params)

    @property
    def final_lambda_(self):
        """Return the final rate parameter estimate."""
        return poisson.Poisson.to_canonical(self.final_params)

    @property
    def ema_lambda_(self):
        """Return the EMA rate parameter estimate."""
        return poisson.Poisson.to_canonical(self.ema_params)

    @property
    def avg_lambda_(self):
        """Return the averaged rate parameter estimate."""
        return poisson.Poisson.to_canonical(self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated poisson distribution"
