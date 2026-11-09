# Author: pstefanou12@
"""Truncated inverse gamma distribution with known shape."""

from collections.abc import Callable
from functools import partial
from typing import Optional

import torch as ch
import torch.distributions.transforms
import torch.nn

from delphi.configs.distributions.truncated import (
    exponential_family as exponential_family_configs,
)
from delphi.exponential_family import inverse_gamma
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedInverseGamma(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated inverse gamma distributions with known shape.

    Attributes:
        concentration (ch.Tensor): Known shape, one per coordinate.
    """

    defaults = exponential_family_configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("rate_",)

    def __init__(
        self,
        phi: Optional[Callable],
        concentration: ch.Tensor,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedInverseGamma.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            concentration: Known shape, one per coordinate.
            alpha: Survival probability lower bound.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, alpha, hyperparameters=hyperparameters)
        self.concentration = concentration

    def _build_distribution(self):
        """Bind the known shape into the distribution."""
        return (
            partial(inverse_gamma.InverseGamma, self.concentration),
            inverse_gamma.InverseGamma.calc_suff_stat,
        )

    def _truncation_set_transform(self) -> torch.distributions.transforms.Transform:
        """Learn the set in log x, where the heavy right tail is tame."""
        return torch.distributions.transforms.ExpTransform().inv

    def _calc_emp_model(self):
        """Initialize theta at the MLE rate ``concentration / mean(1 / x)``."""
        dataset_s = self.train_loader_.dataset.S
        self.emp_theta = inverse_gamma.InverseGamma.to_natural(
            self.concentration / (1.0 / dataset_s).mean(0)
        )
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta to be strictly negative, so the rate is positive."""
        return ch.clamp(theta, max=-1e-6)

    @property
    def best_rate_(self):
        """Return the best rate estimate."""
        return inverse_gamma.InverseGamma.to_canonical(self.best_params)

    @property
    def final_rate_(self):
        """Return the final rate estimate."""
        return inverse_gamma.InverseGamma.to_canonical(self.final_params)

    @property
    def ema_rate_(self):
        """Return the EMA rate estimate."""
        return inverse_gamma.InverseGamma.to_canonical(self.ema_params)

    @property
    def avg_rate_(self):
        """Return the averaged rate estimate."""
        return inverse_gamma.InverseGamma.to_canonical(self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated inverse gamma distribution"
