# Author: pstefanou12@
"""Truncated Gamma distribution with unknown shape."""

from collections.abc import Callable
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import gamma_unknown_concentration
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedGammaUnknownConcentration(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Gamma distributions with unknown shape and rate."""

    dist = gamma_unknown_concentration.GammaUnknownConcentration
    defaults = configs.TruncatedGammaUnknownConcentrationConfig
    estimate_suffixes = ("concentration_", "rate_")
    # Each coordinate's shape and rate are coupled.
    independent_coordinates = False

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedGammaUnknownConcentration.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, dims, alpha)

    def _calc_emp_model(self):
        """Start at the moment estimate, refined by the warm start."""
        dataset_s = self.train_loader_.dataset.S
        mean, var = dataset_s.mean(0), dataset_s.var(0)
        self.emp_theta = (
            gamma_unknown_concentration.GammaUnknownConcentration.to_natural(
                ch.cat([mean**2 / var, mean / var]).clamp(min=1e-3)
            )
        )
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Keep the shape positive and the negated rate negative."""
        return ch.cat(
            [
                theta[: self.dims].clamp(min=-1.0 + 1e-6),
                theta[self.dims :].clamp(max=-1e-6),
            ]
        )

    def _canonical(self, theta: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
        """Split theta's canonical form into the shape and the rate."""
        concentration, rate = (
            gamma_unknown_concentration.GammaUnknownConcentration.to_canonical(
                theta
            ).split(self.dims)
        )
        return concentration, rate

    @property
    def best_concentration_(self):
        """Return the best shape estimate."""
        return self._canonical(self.best_params)[0]

    @property
    def final_concentration_(self):
        """Return the final shape estimate."""
        return self._canonical(self.final_params)[0]

    @property
    def ema_concentration_(self):
        """Return the EMA shape estimate."""
        return self._canonical(self.ema_params)[0]

    @property
    def avg_concentration_(self):
        """Return the averaged shape estimate."""
        return self._canonical(self.avg_params)[0]

    @property
    def best_rate_(self):
        """Return the best rate estimate."""
        return self._canonical(self.best_params)[1]

    @property
    def final_rate_(self):
        """Return the final rate estimate."""
        return self._canonical(self.final_params)[1]

    @property
    def ema_rate_(self):
        """Return the EMA rate estimate."""
        return self._canonical(self.ema_params)[1]

    @property
    def avg_rate_(self):
        """Return the averaged rate estimate."""
        return self._canonical(self.avg_params)[1]

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated gamma distribution with unknown shape"
