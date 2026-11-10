# Author: pstefanou12@
"""Truncated inverse Gaussian (Wald) distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
import torch.distributions.transforms
import torch.nn

from delphi.configs.distributions.truncated import (
    inverse_gaussian as inverse_gaussian_configs,
)
from delphi.exponential_family import inverse_gaussian
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedInverseGaussian(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated inverse Gaussian distributions."""

    dist = inverse_gaussian.InverseGaussian
    defaults = inverse_gaussian_configs.TruncatedInverseGaussianConfig
    estimate_suffixes = ("loc_", "concentration_")
    # Each coordinate's x and 1 / x are correlated, so the Hessian is
    # block-diagonal and its spectrum needs eigenvalues.
    independent_coordinates = False

    def __init__(
        self,
        phi: Optional[Callable],
        dims: int,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize TruncatedInverseGaussian.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, dims, alpha, hyperparameters=hyperparameters)

    def _truncation_set_transform(self) -> torch.distributions.transforms.Transform:
        """Learn the set in log x, where the heavy right tail is tame."""
        return torch.distributions.transforms.ExpTransform().inv

    def _calc_emp_model(self):
        """Initialize theta at the MLE, ``mean(x)`` and ``1 / mean(1/x - 1/loc)``."""
        dataset_s = self.train_loader_.dataset.S
        loc = dataset_s.mean(0)
        concentration = 1.0 / (1.0 / dataset_s - 1.0 / loc).mean(0)
        self.emp_theta = inverse_gaussian.InverseGaussian.to_natural(
            ch.cat([loc, concentration])
        )
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta to be strictly negative, so both parameters are positive."""
        return ch.clamp(theta, max=-1e-6)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated inverse gaussian distribution"
