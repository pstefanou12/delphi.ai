# Author: pstefanou12@
"""Truncated Poisson Distribution."""

from collections.abc import Callable
import logging


from delphi import delphi_logger
from delphi.exponential_family import poisson
from delphi.truncated.distributions import truncated_exponential_family_distributions
from delphi.utils import configs


class TruncatedPoisson(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Poisson distributions to be passed into trainer."""

    dist = poisson.ExponentialFamilyPoisson

    def __init__(
        self,
        args: dict | configs.TruncatedExponentialFamilyDistributionConfig,
        phi: Callable,
        alpha: float,
        dims: int,
    ):
        """Initialize TruncatedPoisson.

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

    @property
    def best_lambda_(self):
        """Return the best rate parameter estimate."""
        return poisson.ExponentialFamilyPoisson.to_canonical(self.best_params)

    @property
    def final_lambda_(self):
        """Return the final rate parameter estimate."""
        return poisson.ExponentialFamilyPoisson.to_canonical(self.final_params)

    @property
    def ema_lambda_(self):
        """Return the EMA rate parameter estimate."""
        return poisson.ExponentialFamilyPoisson.to_canonical(self.ema_params)

    @property
    def avg_lambda_(self):
        """Return the averaged rate parameter estimate."""
        return poisson.ExponentialFamilyPoisson.to_canonical(self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated poisson distribution"
