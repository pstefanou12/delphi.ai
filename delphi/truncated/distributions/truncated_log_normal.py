# Author: pstefanou12@
"""Truncated log-normal distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
import torch.distributions.transforms
import torch.nn

from delphi.configs.distributions.truncated import (
    exponential_family as exponential_family_configs,
)
from delphi.exponential_family import log_normal
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedLogNormal(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated log-normal distributions."""

    dist = log_normal.LogNormal
    # Each coordinate's loc and scale are coupled, so the Hessian is
    # block-diagonal and its spectrum needs eigenvalues.
    independent_coordinates = False
    defaults = exponential_family_configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("loc_", "scale_")

    def __init__(
        self,
        phi: Optional[Callable],
        dims: int,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize TruncatedLogNormal.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, dims, alpha, hyperparameters=hyperparameters)

    def _truncation_set_transform(self) -> torch.distributions.transforms.Transform:
        """Learn the set in log x, where the family is normal."""
        return torch.distributions.transforms.ExpTransform().inv

    def _calc_emp_model(self):
        """Initialize theta at the MLE: the mean and deviation of ``log x``."""
        log_s = ch.log(self.train_loader_.dataset.S)
        self.emp_theta = log_normal.LogNormal.to_natural(
            ch.cat([log_s.mean(0), log_s.std(0)])
        )
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Keep the second natural parameter negative, so the scale is real."""
        return ch.cat([theta[: self.dims], theta[self.dims :].clamp(max=-1e-6)])

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated log-normal distribution"
