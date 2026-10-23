# Author: pstefanou12@
"""Truncated inverse Gaussian (Wald) distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import inverse_gaussian
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedInverseGaussian(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated inverse Gaussian distributions."""

    dist = inverse_gaussian.InverseGaussian
    defaults = configs.TruncatedInverseGaussianConfig
    estimate_suffixes = ("loc_", "concentration_")
    # Each coordinate's x and 1 / x are correlated, so the Hessian is
    # block-diagonal and its spectrum needs eigenvalues.
    independent_coordinates = False

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedInverseGaussian.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, dims, alpha)

    def _calc_emp_model(self):
        """Initialize theta at the MLE, ``mean(x)`` and ``1 / mean(1/x - 1/loc)``."""
        dataset_s = self.train_loader_.dataset.S
        loc = dataset_s.mean(0)
        concentration = 1.0 / (1.0 / dataset_s - 1.0 / loc).mean(0)
        self.emp_theta = inverse_gaussian.InverseGaussian.to_natural(
            ch.cat([loc, concentration])
        )
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta to be strictly negative, so both parameters are positive."""
        return ch.clamp(theta, max=-1e-6)

    def _canonical(self, theta: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
        """Split theta's canonical form into the mean and the shape."""
        loc, concentration = inverse_gaussian.InverseGaussian.to_canonical(theta).split(
            self.dims
        )
        return loc, concentration

    @property
    def best_loc_(self):
        """Return the best mean estimate."""
        return self._canonical(self.best_params)[0]

    @property
    def final_loc_(self):
        """Return the final mean estimate."""
        return self._canonical(self.final_params)[0]

    @property
    def ema_loc_(self):
        """Return the EMA mean estimate."""
        return self._canonical(self.ema_params)[0]

    @property
    def avg_loc_(self):
        """Return the averaged mean estimate."""
        return self._canonical(self.avg_params)[0]

    @property
    def best_concentration_(self):
        """Return the best shape estimate."""
        return self._canonical(self.best_params)[1]

    @property
    def final_concentration_(self):
        """Return the final shape estimate."""
        return self._canonical(self.final_params)[1]

    @property
    def ema_concentration_(self):
        """Return the EMA shape estimate."""
        return self._canonical(self.ema_params)[1]

    @property
    def avg_concentration_(self):
        """Return the averaged shape estimate."""
        return self._canonical(self.avg_params)[1]

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated inverse gaussian distribution"
