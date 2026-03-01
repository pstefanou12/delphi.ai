# Author: pstefanou12@
"""Truncated Poisson Distribution."""

import logging
from collections.abc import Callable

import torch as ch

from delphi.truncated.distributions.truncated_exponential_family_distributions import (
    TruncatedExponentialFamilyDistribution,
)
from delphi.delphi_logger import delphiLogger
from delphi.distributions.poisson import (
    ExponentialFamilyPoisson,
    calc_poiss_suff_stat,
)
from delphi.utils.helpers import Parameters
from delphi.utils.defaults import check_and_fill_args, TRUNC_POISS_DEFAULTS


class TruncatedPoisson(TruncatedExponentialFamilyDistribution):
    """Model for truncated Poisson distributions to be passed into trainer."""

    def __init__(
        self,
        args: Parameters,
        phi: Callable,
        alpha: float,
        dims: int,
    ):
        """Initialize TruncatedPoisson.

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
        args = check_and_fill_args(args, TRUNC_POISS_DEFAULTS)

        logger = (
            delphiLogger() if args.verbose else delphiLogger(level=logging.CRITICAL)
        )
        super().__init__(
            args,
            phi,
            alpha,
            dims,
            ExponentialFamilyPoisson,
            logger,
        )

    @staticmethod
    def _calc_suff_stat(x):
        """Compute sufficient statistics for the Poisson distribution."""
        return calc_poiss_suff_stat(x)

    def _reparameterize_nat_form(self, theta):
        """Convert canonical rate parameter to natural log form."""
        return ch.log(theta)

    def _reparameterize_canon_form(self, theta):
        """Convert natural parameters to canonical rate parameter."""
        return ch.exp(theta)

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
        return "truncated poisson distribution"
