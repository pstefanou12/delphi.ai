# Author: pstefanou12@
"""Truncated Lévy distribution with known location."""

from collections.abc import Callable
import functools
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import levy
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedLevy(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Lévy distributions with known location.

    Attributes:
        loc (ch.Tensor): Known location, one per coordinate.
    """

    defaults = configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("scale_",)

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        loc: ch.Tensor,
        alpha: Optional[float] = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedLevy.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            loc: Known location, one per coordinate.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, dims, alpha)
        self.loc = loc

    def _build_distribution(self):
        """Bind the known location into the distribution and its statistic."""
        return (
            functools.partial(levy.Levy, self.loc),
            functools.partial(levy.Levy.calc_suff_stat, self.loc),
        )

    def _calc_emp_model(self):
        """Initialize theta at the MLE scale ``1 / mean(1 / (x - loc))``."""
        dataset_s = self.train_loader_.dataset.S
        self.emp_theta = levy.Levy.to_natural(
            1.0 / self.calc_suff_stat(dataset_s).mean(0)
        )
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta to be strictly negative, so the scale is positive."""
        return ch.clamp(theta, max=-1e-6)

    @property
    def best_scale_(self):
        """Return the best scale estimate."""
        return levy.Levy.to_canonical(self.best_params)

    @property
    def final_scale_(self):
        """Return the final scale estimate."""
        return levy.Levy.to_canonical(self.final_params)

    @property
    def ema_scale_(self):
        """Return the EMA scale estimate."""
        return levy.Levy.to_canonical(self.ema_params)

    @property
    def avg_scale_(self):
        """Return the averaged scale estimate."""
        return levy.Levy.to_canonical(self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated levy distribution"
