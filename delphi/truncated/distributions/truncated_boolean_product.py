# Author: pstefanou12@
"""Truncated Boolean Product Distributions."""

from collections.abc import Callable

from delphi import configs
from delphi.exponential_family import boolean_product
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedBooleanProduct(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated boolean product distributions to be passed into trainer."""

    dist = boolean_product.BooleanProduct
    defaults = configs.TruncatedExponentialFamilyDistributionConfig

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        alpha: float,
        dims: int,
    ):
        """Initialize TruncatedBooleanProduct.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
        """
        super().__init__(hyperparameters, phi, alpha, dims)

    @property
    def best_p_(self):
        """Return the best probability parameter estimate."""
        return boolean_product.BooleanProduct.to_canonical(self.best_params)

    @property
    def final_p_(self):
        """Return the final probability parameter estimate."""
        return boolean_product.BooleanProduct.to_canonical(self.final_params)

    @property
    def ema_p_(self):
        """Return the EMA probability parameter estimate."""
        return boolean_product.BooleanProduct.to_canonical(self.ema_params)

    @property
    def avg_p_(self):
        """Return the averaged probability parameter estimate."""
        return boolean_product.BooleanProduct.to_canonical(self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated boolean product distribution"
