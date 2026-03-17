# Author: pstefanou12@
"""Truncated Weibull Distribution."""

from collections.abc import Callable
from functools import partial
import logging

import torch as ch
from torch import nn

from delphi import delphi_logger
from delphi.exponential_family import weibull
from delphi.truncated.distributions import truncated_exponential_family_distributions
from delphi.utils import configs


class TruncatedWeibull(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Weibull distributions to be passed into trainer.

    Attributes:
        k (int): Weibull shape parameter.
    """

    def __init__(
        self,
        args: dict | configs.TruncatedExponentialFamilyDistributionConfig,
        phi: Callable,
        alpha: float,
        dims: int,
        k: int,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedWeibull.

        Args:
            args: Hyperparameter dict or Pydantic config.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
            k: Weibull shape parameter.
        """
        args = configs.make_config(
            args, configs.TruncatedExponentialFamilyDistributionConfig
        )

        logger = (
            delphi_logger.delphiLogger()
            if args.verbose
            else delphi_logger.delphiLogger(level=logging.CRITICAL)
        )
        self.k = k
        self.dist = partial(weibull.Weibull, k)
        self.dist.calc_suff_stat = partial(weibull.Weibull.calc_suff_stat, k)
        super().__init__(
            args,
            phi,
            alpha,
            dims,
            logger,
        )

    def _calc_emp_model(self):
        """Initialize theta at the natural parameter corresponding to the empirical scale."""
        dataset_s = self.train_loader_.dataset.S  # pylint: disable=invalid-name
        # MLE: λ̂^k = mean(x^k), so λ̂ = mean(x^k)^(1/k).
        emp_scale = self.dist.calc_suff_stat(dataset_s).mean(0).pow(1.0 / self.k)
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
