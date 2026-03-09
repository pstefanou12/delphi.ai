# Author: pstefanou12@
"""Truncated Boolean Product Distributions."""

import logging
from collections.abc import Callable


from delphi import delphi_logger
from delphi.exponential_family import boolean_product
from delphi.truncated.distributions import truncated_exponential_family_distributions
from delphi.utils import configs


class TruncatedBooleanProduct(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated boolean product distributions to be passed into trainer."""

    dist = boolean_product.ExponentialFamilyBooleanProduct

    def __init__(
        self,
        args: dict | configs.TruncatedExponentialFamilyDistributionConfig,
        phi: Callable,
        alpha: float,
        dims: int,
    ):
        """Initialize TruncatedBooleanProduct.

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
    def best_p_(self):
        """Return the best probability parameter estimate."""
        return boolean_product.ExponentialFamilyBooleanProduct.to_canonical(
            self.best_params
        )

    @property
    def final_p_(self):
        """Return the final probability parameter estimate."""
        return boolean_product.ExponentialFamilyBooleanProduct.to_canonical(
            self.final_params
        )

    @property
    def ema_p_(self):
        """Return the EMA probability parameter estimate."""
        return boolean_product.ExponentialFamilyBooleanProduct.to_canonical(
            self.ema_params
        )

    @property
    def avg_p_(self):
        """Return the averaged probability parameter estimate."""
        return boolean_product.ExponentialFamilyBooleanProduct.to_canonical(
            self.avg_params
        )

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated boolean product distribution"
