# Author: pstefanou12@
"""Truncated inverse gamma distribution with unknown shape."""

from collections.abc import Callable
from typing import Optional

import torch as ch
import torch.distributions.transforms
import torch.nn

from delphi.configs.distributions.truncated import gamma_unknown_concentration
from delphi.exponential_family import (
    inverse_gamma_unknown_concentration as unknown_shape,
)
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedInverseGammaUnknownConcentration(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated inverse gamma distributions with unknown shape."""

    dist = unknown_shape.InverseGammaUnknownConcentration
    # 1 / x is Gamma, so log x and 1 / x couple as the Gamma's log x and x do.
    defaults = gamma_unknown_concentration.TruncatedGammaUnknownConcentrationConfig
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
        """Initialize TruncatedInverseGammaUnknownConcentration.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, dims, alpha, hyperparameters=hyperparameters)

    def _truncation_set_transform(self) -> torch.distributions.transforms.Transform:
        """Learn the set in log x, where the heavy right tail is tame."""
        return torch.distributions.transforms.ExpTransform().inv

    def _calc_emp_model(self):
        """Start at the Gamma moment estimate of ``1 / x``, then warm start.

        The inverse gamma's own moments need shape above 2 to exist.
        """
        reciprocal = 1.0 / self.train_loader_.dataset.S
        mean, var = reciprocal.mean(0), reciprocal.var(0)
        self.emp_theta = unknown_shape.InverseGammaUnknownConcentration.to_natural(
            ch.cat([mean**2 / var, mean / var]).clamp(min=1e-3)
        )
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Keep the shape positive and the negated rate negative."""
        return ch.cat(
            [
                theta[: self.dims].clamp(max=-1.0 - 1e-6),
                theta[self.dims :].clamp(max=-1e-6),
            ]
        )

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated inverse gamma distribution with unknown shape"
