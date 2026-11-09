# Author: pstefanou12@
"""Truncated Gamma distribution with unknown shape."""

from collections.abc import Callable
from typing import Optional

import torch as ch
import torch.nn

from delphi.configs.distributions.truncated import (
    gamma_unknown_concentration as gamma_unknown_concentration_configs,
)
from delphi.exponential_family import gamma_unknown_concentration
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedGammaUnknownConcentration(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Gamma distributions with unknown shape and rate."""

    dist = gamma_unknown_concentration.GammaUnknownConcentration
    defaults = (
        gamma_unknown_concentration_configs.TruncatedGammaUnknownConcentrationConfig
    )
    estimate_suffixes = ("concentration_", "rate_")
    # Each coordinate's shape and rate are coupled.
    independent_coordinates = False

    def __init__(
        self,
        phi: Optional[Callable],
        dims: int,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize TruncatedGammaUnknownConcentration.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, dims, alpha, hyperparameters=hyperparameters)

    def _calc_emp_model(self):
        """Start at the moment estimate, refined by the warm start."""
        dataset_s = self.train_loader_.dataset.S
        mean, var = dataset_s.mean(0), dataset_s.var(0)
        self.emp_theta = (
            gamma_unknown_concentration.GammaUnknownConcentration.to_natural(
                ch.cat([mean**2 / var, mean / var]).clamp(min=1e-3)
            )
        )
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Keep the shape positive and the negated rate negative."""
        return ch.cat(
            [
                theta[: self.dims].clamp(min=-1.0 + 1e-6),
                theta[self.dims :].clamp(max=-1e-6),
            ]
        )

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated gamma distribution with unknown shape"
