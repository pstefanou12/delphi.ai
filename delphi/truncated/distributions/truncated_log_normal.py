# Author: pstefanou12@
"""Truncated log-normal distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
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
    defaults = configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("loc_", "scale_")

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedLogNormal.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, dims, alpha)

    def _calc_emp_model(self):
        """Initialize theta at the MLE: the mean and deviation of ``log x``."""
        log_s = ch.log(self.train_loader_.dataset.S)
        self.emp_theta = log_normal.LogNormal.to_natural(
            ch.cat([log_s.mean(0), log_s.std(0)])
        )
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Keep the second natural parameter negative, so the scale is real."""
        return ch.cat([theta[: self.dims], theta[self.dims :].clamp(max=-1e-6)])

    @property
    def best_loc_(self):
        """Return the best loc estimate."""
        return log_normal.LogNormal.to_canonical(self.best_params)[: self.dims]

    @property
    def final_loc_(self):
        """Return the final loc estimate."""
        return log_normal.LogNormal.to_canonical(self.final_params)[: self.dims]

    @property
    def ema_loc_(self):
        """Return the EMA loc estimate."""
        return log_normal.LogNormal.to_canonical(self.ema_params)[: self.dims]

    @property
    def avg_loc_(self):
        """Return the averaged loc estimate."""
        return log_normal.LogNormal.to_canonical(self.avg_params)[: self.dims]

    @property
    def best_scale_(self):
        """Return the best scale estimate."""
        return log_normal.LogNormal.to_canonical(self.best_params)[self.dims :]

    @property
    def final_scale_(self):
        """Return the final scale estimate."""
        return log_normal.LogNormal.to_canonical(self.final_params)[self.dims :]

    @property
    def ema_scale_(self):
        """Return the EMA scale estimate."""
        return log_normal.LogNormal.to_canonical(self.ema_params)[self.dims :]

    @property
    def avg_scale_(self):
        """Return the averaged scale estimate."""
        return log_normal.LogNormal.to_canonical(self.avg_params)[self.dims :]

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated log-normal distribution"
