"""
Truncated Weibull Distribution.
"""

import logging
from functools import partial
from typing import Callable

import torch as ch

from delphi.distributions.truncated_exponential_family_distributions import (
    TruncatedExponentialFamilyDistribution,
)
from delphi.delphi_logger import delphiLogger
from delphi.grad import ExponentialFamilyWeibull, calc_weibull_suff_stat
from delphi.utils.helpers import Parameters
from delphi.utils.defaults import check_and_fill_args, TRUNC_WEIBULL_DEFAULTS


class TruncatedWeibull(TruncatedExponentialFamilyDistribution):
    """
    Model for truncated exponential distributions to be passed into trainer.
    """

    def __init__(
        self, args: Parameters, phi: Callable, alpha: float, dims: int, k: int
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """
        Args:
            args (cox.utils.Parameters) : parameter object holding hyperparameters
            phi (Callable): truncation set oracle
            alpha (float): survival probability lower bound
            dims (int): number of dimensions
            k (int): Weibull shape parameter
        """
        assert isinstance(args, Parameters), (
            "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
        )
        args = check_and_fill_args(args, TRUNC_WEIBULL_DEFAULTS)

        logger = (
            delphiLogger() if args.verbose else delphiLogger(level=logging.CRITICAL)
        )
        super().__init__(
            args,
            phi,
            alpha,
            dims,
            partial(ExponentialFamilyWeibull, k),
            partial(calc_weibull_suff_stat, k),
            logger,
        )
        self.k = k

    def _constraints(self, theta):
        """Clamp theta to be strictly negative."""
        return ch.clamp(theta, max=-1e-6)

    def _reparameterize_nat_form(self, theta):
        """Convert canonical scale parameter to natural form."""
        return -1.0 / theta.pow(self.k)

    def _reparameterize_canon_form(self, theta):
        """Convert natural parameters to canonical scale parameter."""
        return (-1 / theta).pow(1 / self.k)

    @property
    def best_lambda_(self):
        """Return the best scale parameter estimate."""
        return self.best_params

    @property
    def final_lambda_(self):
        """Return the final scale parameter estimate."""
        return self.final_params

    @property
    def ema_lambda_(self):
        """Return the EMA scale parameter estimate."""
        return self.ema_params

    @property
    def avg_lambda_(self):
        """Return the averaged scale parameter estimate."""
        return self.avg_params

    def __str__(self):
        return "truncated weibull distribution"
