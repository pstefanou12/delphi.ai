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
        concentration (ch.Tensor): Known shape, one per coordinate.
    """

    defaults = configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("scale_",)

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        concentration: ch.Tensor,
        alpha: Optional[float] = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedWeibull.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
            concentration: Known shape, one per coordinate.
        """
        super().__init__(hyperparameters, phi, dims, alpha)
        self.concentration = concentration

    def _build_distribution(self):
        """Bind the known shape into the distribution and its statistic.

        Returns:
            The Weibull factory and sufficient statistic, both closed over the shape.
        """
        return (
            partial(weibull.Weibull, self.concentration),
            partial(weibull.Weibull.calc_suff_stat, self.concentration),
        )

    def _calc_emp_model(self):
        """Initialize theta at the natural parameter corresponding to the empirical scale."""
        dataset_s = self.train_loader_.dataset.S  # pylint: disable=invalid-name
        # MLE: scale^k = mean(x^k), so scale = mean(x^k)^(1/k).
        emp_scale = self.calc_suff_stat(dataset_s).mean(0).pow(1.0 / self.concentration)
        self.emp_theta = weibull.Weibull.to_natural(self.concentration, emp_scale)
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta to be strictly negative."""
        return ch.clamp(theta, max=-1e-6)

    @property
    def best_scale_(self):
        """Return the best scale parameter estimate."""
        return weibull.Weibull.to_canonical(self.concentration, self.best_params)

    @property
    def final_scale_(self):
        """Return the final scale parameter estimate."""
        return weibull.Weibull.to_canonical(self.concentration, self.final_params)

    @property
    def ema_scale_(self):
        """Return the EMA scale parameter estimate."""
        return weibull.Weibull.to_canonical(self.concentration, self.ema_params)

    @property
    def avg_scale_(self):
        """Return the averaged scale parameter estimate."""
        return weibull.Weibull.to_canonical(self.concentration, self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated weibull distribution"
