# Author: pstefanou12@
"""Truncated Weibull Distribution."""

from collections.abc import Callable
from functools import partial
import logging

import torch as ch

from delphi import delphi_logger
from delphi.distributions import weibull
from delphi.truncated.distributions import truncated_exponential_family_distributions
from delphi.utils import defaults, helpers


class TruncatedWeibull(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Weibull distributions to be passed into trainer.

    Attributes:
        k (int): Weibull shape parameter.
    """

    def __init__(
        self,
        args: helpers.Parameters,
        phi: Callable,
        alpha: float,
        dims: int,
        k: int,
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
        if not isinstance(args, helpers.Parameters):
            raise TypeError(f"args is type {type(args).__name__}; expected Parameters.")
        args = defaults.check_and_fill_args(args, defaults.TRUNC_WEIBULL_DEFAULTS)

        logger = (
            delphi_logger.delphiLogger()
            if args.verbose
            else delphi_logger.delphiLogger(level=logging.CRITICAL)
        )
        self.k = k
        self.dist = partial(weibull.ExponentialFamilyWeibull, k)
        super().__init__(
            args,
            phi,
            alpha,
            dims,
            logger,
        )

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
