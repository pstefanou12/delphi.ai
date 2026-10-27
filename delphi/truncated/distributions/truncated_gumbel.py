# Author: pstefanou12@
"""Truncated Gumbel distribution with known scale."""

from collections.abc import Callable
import functools
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import gumbel
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedGumbel(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Gumbel distributions with known scale.

    Attributes:
        scale (ch.Tensor): Known scale, one per coordinate.
    """

    defaults = configs.TruncatedGumbelConfig
    estimate_suffixes = ("loc_",)

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        scale: ch.Tensor,
        alpha: Optional[float] = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedGumbel.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            scale: Known scale, one per coordinate.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, dims, alpha)
        self.scale = scale

    def _build_distribution(self):
        """Bind the known scale into the distribution and its statistic."""
        return (
            functools.partial(gumbel.Gumbel, self.scale),
            functools.partial(gumbel.Gumbel.calc_suff_stat, self.scale),
        )

    def _calc_emp_model(self):
        """Initialize theta at the MLE, ``-1 / mean(exp(-x / scale))``."""
        dataset_s = self.train_loader_.dataset.S
        self.emp_theta = -1.0 / self.calc_suff_stat(dataset_s).mean(0)
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta to be strictly negative, so the loc is real."""
        return ch.clamp(theta, max=-1e-6)

    @property
    def best_loc_(self):
        """Return the best loc estimate."""
        return gumbel.Gumbel.to_canonical(self.scale, self.best_params)

    @property
    def final_loc_(self):
        """Return the final loc estimate."""
        return gumbel.Gumbel.to_canonical(self.scale, self.final_params)

    @property
    def ema_loc_(self):
        """Return the EMA loc estimate."""
        return gumbel.Gumbel.to_canonical(self.scale, self.ema_params)

    @property
    def avg_loc_(self):
        """Return the averaged loc estimate."""
        return gumbel.Gumbel.to_canonical(self.scale, self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated gumbel distribution"
