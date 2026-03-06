# Author: pstefanou12@
"""Truncated Poisson Distribution."""

from collections.abc import Callable
import logging

import torch as ch

from delphi import delphi_logger
from delphi.distributions import poisson
from delphi.truncated.distributions import truncated_exponential_family_distributions
from delphi.utils import defaults, helpers


class TruncatedPoisson(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Poisson distributions to be passed into trainer."""

    def __init__(
        self,
        args: helpers.Parameters,
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
        if not isinstance(args, helpers.Parameters):
            raise TypeError(f"args is type {type(args).__name__}; expected Parameters.")
        args = defaults.check_and_fill_args(args, defaults.TRUNC_POISS_DEFAULTS)

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
            poisson.ExponentialFamilyPoisson,
            logger,
        )

    @staticmethod
    def _calc_suff_stat(x):
        """Compute sufficient statistics for the Poisson distribution."""
        return poisson.calc_poiss_suff_stat(x)

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
