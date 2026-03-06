# Author: pstefanou12@
"""Truncated Boolean Product Distributions."""

import logging
from collections.abc import Callable

import torch as ch

from delphi import delphi_logger
from delphi.distributions import boolean_product
from delphi.truncated.distributions import truncated_exponential_family_distributions
from delphi.utils import defaults, helpers


class TruncatedBooleanProduct(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated boolean product distributions to be passed into trainer."""

    def __init__(
        self, args: helpers.Parameters, phi: Callable, alpha: float, dims: int
    ):
        """Initialize TruncatedBooleanProduct.

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
        args = defaults.check_and_fill_args(args, defaults.TRUNC_BOOL_PROD_DEFAULTS)

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
            boolean_product.ExponentialFamilyBooleanProduct,
            logger,
        )

    @staticmethod
    def _calc_suff_stat(x):
        """Compute sufficient statistics for the Boolean product distribution."""
        return boolean_product.calc_bool_prod_suff_stat(x)

    def _reparameterize_nat_form(self, theta):
        """Convert canonical probability parameter to natural log-odds form."""
        return ch.log(theta / (1 - theta))

    def _reparameterize_canon_form(self, theta):
        """Convert natural log-odds to canonical probability parameter."""
        return ch.exp(theta) / (1 + ch.exp(theta))

    @property
    def best_p_(self):
        """Return the best probability parameter estimate."""
        return self.best_params

    @property
    def final_p_(self):
        """Return the final probability parameter estimate."""
        return self.final_params

    @property
    def ema_p_(self):
        """Return the EMA probability parameter estimate."""
        return self.ema_params

    @property
    def avg_p_(self):
        """Return the averaged probability parameter estimate."""
        return self.avg_params

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated boolean product distribution"
