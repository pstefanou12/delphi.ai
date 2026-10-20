# Author: pstefanou12@
"""Truncated Pareto distribution with known scale."""

from collections.abc import Callable
from functools import partial
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import pareto
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedPareto(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Pareto distributions with known scale.

    Attributes:
        scale (ch.Tensor): Known scale, the lower end of the support, one per
            coordinate.
    """

    defaults = configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("alpha_",)

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        scale: ch.Tensor,
        alpha: Optional[float] = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedPareto.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            scale: Known scale, one per coordinate.
            alpha: Survival probability lower bound. Unrelated to the tail
                index, which the estimates call ``alpha_``.
        """
        super().__init__(hyperparameters, phi, dims, alpha)
        self.scale = scale

    def _build_distribution(self):
        """Bind the known scale into the distribution."""
        return partial(pareto.Pareto, self.scale), pareto.Pareto.calc_suff_stat

    def _calc_emp_model(self):
        """Initialize theta at the MLE tail index ``1 / mean(log(x / scale))``."""
        dataset_s = self.train_loader_.dataset.S
        emp_alpha = 1.0 / (ch.log(dataset_s) - ch.log(self.scale)).mean(0)
        self.emp_theta = pareto.Pareto.to_natural(emp_alpha)
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta below -1, so the tail index is positive."""
        return ch.clamp(theta, max=-1.0 - 1e-6)

    @property
    def best_alpha_(self):
        """Return the best tail index estimate."""
        return pareto.Pareto.to_canonical(self.best_params)

    @property
    def final_alpha_(self):
        """Return the final tail index estimate."""
        return pareto.Pareto.to_canonical(self.final_params)

    @property
    def ema_alpha_(self):
        """Return the EMA tail index estimate."""
        return pareto.Pareto.to_canonical(self.ema_params)

    @property
    def avg_alpha_(self):
        """Return the averaged tail index estimate."""
        return pareto.Pareto.to_canonical(self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated pareto distribution"
