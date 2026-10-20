# Author: pstefanou12@
"""Truncated Boolean Product Distributions."""

from collections.abc import Callable
from typing import Optional

from torch import nn

from delphi import configs
from delphi.exponential_family import boolean_product
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedBooleanProduct(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated boolean product distributions to be passed into trainer."""

    dist = boolean_product.BooleanProduct
    defaults = configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("probs_",)

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedBooleanProduct.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
        """
        super().__init__(hyperparameters, phi, dims, alpha)

    def _calc_emp_model(self):
        """Initialize theta at the log-odds of the empirical success rate."""
        dataset_s = self.train_loader_.dataset.S  # pylint: disable=invalid-name
        # A coordinate the set pins to 0 or 1 would otherwise give an
        # infinite log-odds.
        emp = dataset_s.mean(0).clamp(1e-6, 1 - 1e-6)
        self.emp_theta = boolean_product.BooleanProduct.to_natural(emp)
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    @property
    def best_probs_(self):
        """Return the best probability parameter estimate."""
        return boolean_product.BooleanProduct.to_canonical(self.best_params)

    @property
    def final_probs_(self):
        """Return the final probability parameter estimate."""
        return boolean_product.BooleanProduct.to_canonical(self.final_params)

    @property
    def ema_probs_(self):
        """Return the EMA probability parameter estimate."""
        return boolean_product.BooleanProduct.to_canonical(self.ema_params)

    @property
    def avg_probs_(self):
        """Return the averaged probability parameter estimate."""
        return boolean_product.BooleanProduct.to_canonical(self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated boolean product distribution"
