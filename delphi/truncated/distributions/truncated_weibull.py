# Author: pstefanou12@
"""Truncated Weibull Distribution."""

import logging
from collections.abc import Callable
from functools import partial

import torch as ch

from delphi.truncated.distributions.truncated_exponential_family_distributions import (
    TruncatedExponentialFamilyDistribution,
)
from delphi.delphi_logger import delphiLogger
from delphi.distributions.weibull import (
    ExponentialFamilyWeibull,
    calc_weibull_suff_stat,
)
from delphi.utils.helpers import Parameters
from delphi.utils.defaults import check_and_fill_args, TRUNC_WEIBULL_DEFAULTS


class TruncatedWeibull(TruncatedExponentialFamilyDistribution):
    """Model for truncated Weibull distributions to be passed into trainer.

    Attributes:
        k (int): Weibull shape parameter.
    """

    def __init__(
        self, args: Parameters, phi: Callable, alpha: float, dims: int, k: int
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedWeibull.

        Args:
            args: Parameter object holding hyperparameters.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
            k: Weibull shape parameter.

        Raises:
            TypeError: If args is not a Parameters instance.
        """
        if not isinstance(args, Parameters):
            raise TypeError(f"args is type {type(args).__name__}; expected Parameters.")
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
        """Return a human-readable name for this distribution."""
        return "truncated weibull distribution"
