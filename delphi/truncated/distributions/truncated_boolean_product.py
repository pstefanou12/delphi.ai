# Author: pstefanou12@
"""Truncated Boolean Product Distributions."""

from collections.abc import Callable
from typing import Optional

import torch.nn

from delphi.configs.distributions.truncated import (
    exponential_family as exponential_family_configs,
)
from delphi.exponential_family import boolean_product
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedBooleanProduct(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated boolean product distributions to be passed into trainer."""

    dist = boolean_product.BooleanProduct
    defaults = exponential_family_configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("probs_",)

    def __init__(
        self,
        phi: Optional[Callable],
        dims: int,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize TruncatedBooleanProduct.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, dims, alpha, hyperparameters=hyperparameters)

    def _calc_emp_model(self):
        """Initialize theta at the log-odds of the empirical success rate."""
        dataset_s = self.train_loader_.dataset.S  # pylint: disable=invalid-name
        # A coordinate the set pins to 0 or 1 would otherwise give an
        # infinite log-odds.
        emp = dataset_s.mean(0).clamp(1e-6, 1 - 1e-6)
        self.emp_theta = boolean_product.BooleanProduct.to_natural(emp)
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated boolean product distribution"
