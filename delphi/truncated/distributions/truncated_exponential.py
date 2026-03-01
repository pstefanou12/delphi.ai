# Author: pstefanou12@
"""Truncated Exponential Distribution."""

import logging
from collections.abc import Callable

import torch as ch

from delphi.truncated.distributions.truncated_exponential_family_distributions import (
    TruncatedExponentialFamilyDistribution,
)
from delphi.delphi_logger import delphiLogger
from delphi.distributions.exponential import (
    ExponentialFamilyExponential,
    calc_exp_suff_stat,
)
from delphi.utils.helpers import Parameters
from delphi.utils.defaults import check_and_fill_args, TRUNC_EXP_DEFAULTS


class TruncatedExponential(TruncatedExponentialFamilyDistribution):
    """Model for truncated exponential distributions to be passed into trainer."""

    def __init__(
        self,
        args: Parameters,
        phi: Callable,
        alpha: float,
        dims: int,
    ):
        """Initialize TruncatedExponential.

        Args:
            args: Parameter object holding hyperparameters.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.

        Raises:
            TypeError: If args is not a Parameters instance.
        """
        if not isinstance(args, Parameters):
            raise TypeError(f"args is type {type(args).__name__}; expected Parameters.")
        args = check_and_fill_args(args, TRUNC_EXP_DEFAULTS)

        logger = (
            delphiLogger() if args.verbose else delphiLogger(level=logging.CRITICAL)
        )
        super().__init__(
            args,
            phi,
            alpha,
            dims,
            ExponentialFamilyExponential,
            logger,
        )

    @staticmethod
    def _calc_suff_stat(x):
        """Compute sufficient statistics for the exponential distribution."""
        return calc_exp_suff_stat(x)

    def _constraints(self, theta):
        """Clamp theta to be strictly negative."""
        return ch.clamp(theta, max=-1e-6)

    def _reparameterize_nat_form(self, theta):
        """Convert canonical rate parameter to natural form."""
        return -theta

    def _reparameterize_canon_form(self, theta):
        """Convert natural parameters to canonical rate parameter."""
        return -theta

    @property
    def best_lambda_(self):
        """Return the best rate parameter estimate."""
        return self.best_params

    @property
    def final_lambda_(self):
        """Return the final rate parameter estimate."""
        return self.final_params

    @property
    def ema_lambda_(self):
        """Return the EMA rate parameter estimate."""
        return self.ema_params

    @property
    def avg_lambda_(self):
        """Return the averaged rate parameter estimate."""
        return self.avg_params

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated exponential distribution"
