# Author: pstefanou12@
"""Truncated Beta distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
import torch.nn

from delphi.configs.distributions.truncated import (
    exponential_family as exponential_family_configs,
)
from delphi.exponential_family import beta
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedBeta(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Beta distributions."""

    family = beta.Beta
    defaults = exponential_family_configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("concentration1_", "concentration0_")
    # Each coordinate's a and b are coupled, so the Hessian is block-diagonal
    # and its spectrum needs eigenvalues rather than the diagonal.
    independent_coordinates = False

    def __init__(
        self,
        phi: Optional[Callable],
        dims: int,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize TruncatedBeta.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, dims, alpha, hyperparameters=hyperparameters)

    def _calc_emp_model(self):
        """Initialize theta by the method of moments; the MLE has no closed form."""
        dataset_s = self.train_loader_.dataset.S
        mean, var = dataset_s.mean(0), dataset_s.var(0)
        common = mean * (1 - mean) / var - 1
        self.emp_theta = beta.Beta.to_natural(
            ch.cat([mean * common, (1 - mean) * common]).clamp(min=1e-3)
        )
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta above -1, so both shape parameters are positive."""
        return ch.clamp(theta, min=-1.0 + 1e-6)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated beta distribution"
