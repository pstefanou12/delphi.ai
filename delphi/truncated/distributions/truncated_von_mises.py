# Author: pstefanou12@
"""Truncated von Mises distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
import torch.nn

from delphi.configs.distributions.truncated import (
    exponential_family as exponential_family_configs,
)
from delphi.exponential_family import von_mises
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedVonMises(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated von Mises distributions."""

    dist = von_mises.VonMises
    defaults = exponential_family_configs.WarmStartedExponentialFamilyDistributionConfig
    estimate_suffixes = ("loc_", "concentration_")
    # Each coordinate's two natural parameters are coupled.
    independent_coordinates = False

    def __init__(
        self,
        phi: Optional[Callable],
        dims: int,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize TruncatedVonMises.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, dims, alpha, hyperparameters=hyperparameters)

    def _calc_emp_model(self):
        """Start at the circular mean and an approximate concentration.

        The concentration solves ``I1(k) / I0(k) = R``, the mean resultant
        length, which has no closed form. Best and Fisher (1981) approximate
        it piecewise; the warm start then refines both.
        """
        moments = von_mises.VonMises.calc_suff_stat(self.train_loader_.dataset.S).mean(
            0
        )
        cos, sin = moments.chunk(2)
        length = ch.hypot(cos, sin).clamp(max=1 - 1e-3)
        concentration = ch.where(
            length < 0.53,
            2 * length + length**3 + 5 * length**5 / 6,
            ch.where(
                length < 0.85,
                -0.4 + 1.39 * length + 0.43 / (1 - length),
                1 / (length**3 - 4 * length**2 + 3 * length),
            ),
        )
        self.emp_theta = von_mises.VonMises.to_natural(
            ch.cat([ch.atan2(sin, cos), concentration])
        )
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated von mises distribution"
