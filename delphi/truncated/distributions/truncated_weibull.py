# Author: pstefanou12@
"""Truncated Weibull Distribution."""

from collections.abc import Callable
from functools import partial
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import weibull
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedWeibull(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Weibull distributions to be passed into trainer.

    Attributes:
        k (int): Weibull shape parameter.
    """

    defaults = configs.TruncatedExponentialFamilyDistributionConfig

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        k: int,
        alpha: Optional[float] = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedWeibull.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
            k: Weibull shape parameter.
        """
        super().__init__(hyperparameters, phi, dims, alpha)
        self.k = k

    def _build_distribution(self):
        """Bind the known shape into the distribution and its statistic.

        Returns:
            The Weibull factory and sufficient statistic, both closed over k.
        """
        return (
            partial(weibull.Weibull, self.k),
            partial(weibull.Weibull.calc_suff_stat, self.k),
        )

    def _calc_emp_model(self):
        """Initialize theta at the natural parameter corresponding to the empirical scale."""
        dataset_s = self.train_loader_.dataset.S  # pylint: disable=invalid-name
        # MLE: λ̂^k = mean(x^k), so λ̂ = mean(x^k)^(1/k).
        emp_scale = self.calc_suff_stat(dataset_s).mean(0).pow(1.0 / self.k)
        self.emp_theta = weibull.Weibull.to_natural(self.k, emp_scale)
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta to be strictly negative."""
        return ch.clamp(theta, max=-1e-6)

    @property
    def best_lambda_(self):
        """Return the best scale parameter estimate."""
        return weibull.Weibull.to_canonical(self.k, self.best_params)

    @property
    def final_lambda_(self):
        """Return the final scale parameter estimate."""
        return weibull.Weibull.to_canonical(self.k, self.final_params)

    @property
    def ema_lambda_(self):
        """Return the EMA scale parameter estimate."""
        return weibull.Weibull.to_canonical(self.k, self.ema_params)

    @property
    def avg_lambda_(self):
        """Return the averaged scale parameter estimate."""
        return weibull.Weibull.to_canonical(self.k, self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated weibull distribution"
