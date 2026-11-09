# Author: pstefanou12@
"""Truncated Gamma distribution with known shape."""

from collections.abc import Callable
from functools import partial
from typing import Optional

import torch as ch
import torch.nn

from delphi.configs.distributions.truncated import (
    exponential_family as exponential_family_configs,
)
from delphi.exponential_family import gamma
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedGamma(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Gamma distributions with known shape.

    Attributes:
        concentration (ch.Tensor): Known shape, one per coordinate, under
            torch's name for it.
    """

    defaults = exponential_family_configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("rate_",)

    def __init__(
        self,
        phi: Optional[Callable],
        dims: int,
        concentration: ch.Tensor,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedGamma.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            dims: Number of dimensions.
            concentration: Known shape, one per coordinate.
            alpha: Survival probability lower bound.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, dims, alpha, hyperparameters=hyperparameters)
        self.concentration = concentration

    def _build_distribution(self):
        """Bind the known shape into the distribution."""
        return partial(gamma.Gamma, self.concentration), gamma.Gamma.calc_suff_stat

    def _calc_emp_model(self):
        """Initialize theta at the empirical rate, the MLE ``concentration / mean(x)``."""
        dataset_s = self.train_loader_.dataset.S
        self.emp_theta = gamma.Gamma.to_natural(self.concentration / dataset_s.mean(0))
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta to be strictly negative, so the rate is positive."""
        return ch.clamp(theta, max=-1e-6)

    def _canonical(self, theta: ch.Tensor) -> tuple[ch.Tensor, ...]:
        """Return the canonical estimate theta encodes."""
        return (gamma.Gamma.to_canonical(theta),)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated gamma distribution"
