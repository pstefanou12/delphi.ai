# Author: pstefanou12@
"""Truncated Beta distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import beta
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedBeta(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Beta distributions."""

    dist = beta.Beta
    defaults = configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("a_", "b_")
    # Each coordinate's a and b are coupled, so the Hessian is block-diagonal
    # and its spectrum needs eigenvalues rather than the diagonal.
    independent_coordinates = False

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedBeta.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, dims, alpha)

    def _calc_emp_model(self):
        """Initialize theta by the method of moments; the MLE has no closed form."""
        dataset_s = self.train_loader_.dataset.S
        mean, var = dataset_s.mean(0), dataset_s.var(0)
        common = mean * (1 - mean) / var - 1
        self.emp_theta = beta.Beta.to_natural(
            ch.cat([mean * common, (1 - mean) * common]).clamp(min=1e-3)
        )
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta above -1, so both shape parameters are positive."""
        return ch.clamp(theta, min=-1.0 + 1e-6)

    @property
    def best_a_(self):
        """Return the best first shape estimate."""
        return beta.Beta.to_canonical(self.best_params)[: self.dims]

    @property
    def final_a_(self):
        """Return the final first shape estimate."""
        return beta.Beta.to_canonical(self.final_params)[: self.dims]

    @property
    def ema_a_(self):
        """Return the EMA first shape estimate."""
        return beta.Beta.to_canonical(self.ema_params)[: self.dims]

    @property
    def avg_a_(self):
        """Return the averaged first shape estimate."""
        return beta.Beta.to_canonical(self.avg_params)[: self.dims]

    @property
    def best_b_(self):
        """Return the best second shape estimate."""
        return beta.Beta.to_canonical(self.best_params)[self.dims :]

    @property
    def final_b_(self):
        """Return the final second shape estimate."""
        return beta.Beta.to_canonical(self.final_params)[self.dims :]

    @property
    def ema_b_(self):
        """Return the EMA second shape estimate."""
        return beta.Beta.to_canonical(self.ema_params)[self.dims :]

    @property
    def avg_b_(self):
        """Return the averaged second shape estimate."""
        return beta.Beta.to_canonical(self.avg_params)[self.dims :]

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated beta distribution"
