# Author: pstefanou12@
"""Truncated Exponential Distribution."""

import logging
from collections.abc import Callable

import torch as ch
from torch import nn

from delphi import delphi_logger
from delphi.exponential_family import exponential
from delphi.truncated.distributions import truncated_exponential_family_distributions
from delphi.utils import configs


class TruncatedExponential(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated exponential distributions to be passed into trainer."""

    dist = exponential.ExponentialFamilyExponential

    def __init__(
        self,
        args: dict | configs.TruncatedExponentialFamilyDistributionConfig,
        phi: Callable,
        alpha: float,
        dims: int,
    ):
        """Initialize TruncatedExponential.

        Args:
            args: Hyperparameter dict or Pydantic config.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
        """
        args = configs.make_config(
            args, configs.TruncatedExponentialFamilyDistributionConfig
        )

        logger = (
            delphi_logger.delphiLogger()
            if args.verbose
            else delphi_logger.delphiLogger(level=logging.CRITICAL)
        )
        super().__init__(
            args,
            phi,
            alpha,
            dims,
            logger,
        )

    def _calc_emp_model(self):
        """Initialize theta at the natural parameter corresponding to the empirical rate."""
        dataset_s = self.train_loader_.dataset.S  # pylint: disable=invalid-name
        emp_rate = 1.0 / dataset_s.mean(0)
        self.emp_theta = exponential.ExponentialFamilyExponential.to_natural(emp_rate)
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta to be strictly negative."""
        return ch.clamp(theta, max=-1e-6)

    @property
    def best_lambda_(self):
        """Return the best rate (canonical) parameter estimate."""
        return exponential.ExponentialFamilyExponential.to_canonical(self.best_params)

    @property
    def final_lambda_(self):
        """Return the final rate (canonical) parameter estimate."""
        return exponential.ExponentialFamilyExponential.to_canonical(self.final_params)

    @property
    def ema_lambda_(self):
        """Return the EMA rate (canonical) parameter estimate."""
        return exponential.ExponentialFamilyExponential.to_canonical(self.ema_params)

    @property
    def avg_lambda_(self):
        """Return the averaged rate (canonical) parameter estimate."""
        return exponential.ExponentialFamilyExponential.to_canonical(self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated exponential distribution"
