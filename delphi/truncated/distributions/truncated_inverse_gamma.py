# Author: pstefanou12@
"""Truncated inverse gamma distribution with known shape."""

from collections.abc import Callable
from functools import partial
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import inverse_gamma
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedInverseGamma(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated inverse gamma distributions with known shape.

    Attributes:
        concentration (ch.Tensor): Known shape, one per coordinate.
    """

    defaults = configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("rate_",)

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        concentration: ch.Tensor,
        alpha: Optional[float] = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedInverseGamma.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            concentration: Known shape, one per coordinate.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, dims, alpha)
        self.concentration = concentration

    def _build_distribution(self):
        """Bind the known shape into the distribution."""
        return (
            partial(inverse_gamma.InverseGamma, self.concentration),
            inverse_gamma.InverseGamma.calc_suff_stat,
        )

    def _calc_emp_model(self):
        """Initialize theta at the MLE rate ``concentration / mean(1 / x)``."""
        dataset_s = self.train_loader_.dataset.S
        self.emp_theta = inverse_gamma.InverseGamma.to_natural(
            self.concentration / (1.0 / dataset_s).mean(0)
        )
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

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
