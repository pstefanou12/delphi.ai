# Author: pstefanou12@
"""Truncated beta prime distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import beta_prime
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedBetaPrime(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated beta prime distributions."""

    dist = beta_prime.BetaPrime
    # log x and log(1 + x) are nearly collinear, which preconditioning undoes.
    defaults = configs.TruncatedGammaUnknownConcentrationConfig
    estimate_suffixes = ("concentration1_", "concentration0_")
    # Each coordinate's log x and log(1 + x) are correlated.
    independent_coordinates = False

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedBetaPrime.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, dims, alpha)

    def _calc_emp_model(self):
        """Start at the Beta moments of ``x / (1 + x)``, then warm start.

        The beta prime's own moments need ``concentration0`` above 2 to exist.
        """
        dataset_s = self.train_loader_.dataset.S
        ratio = dataset_s / (1.0 + dataset_s)
        mean, var = ratio.mean(0), ratio.var(0)
        common = mean * (1.0 - mean) / var - 1.0
        self.emp_theta = beta_prime.BetaPrime.to_natural(
            ch.cat([mean * common, (1.0 - mean) * common]).clamp(min=1e-3)
        )
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Keep both shapes positive."""
        first = theta[: self.dims].clamp(min=-1.0 + 1e-6)
        second = ch.minimum(theta[self.dims :], -first - 1.0 - 1e-6)
        return ch.cat([first, second])

    def _canonical(self, theta: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
        """Split theta's canonical form into the two shapes."""
        concentration1, concentration0 = beta_prime.BetaPrime.to_canonical(theta).split(
            self.dims
        )
        return concentration1, concentration0

    @property
    def best_concentration1_(self):
        """Return the best first shape estimate."""
        return self._canonical(self.best_params)[0]

    @property
    def final_concentration1_(self):
        """Return the final first shape estimate."""
        return self._canonical(self.final_params)[0]

    @property
    def ema_concentration1_(self):
        """Return the EMA first shape estimate."""
        return self._canonical(self.ema_params)[0]

    @property
    def avg_concentration1_(self):
        """Return the averaged first shape estimate."""
        return self._canonical(self.avg_params)[0]

    @property
    def best_concentration0_(self):
        """Return the best second shape estimate."""
        return self._canonical(self.best_params)[1]

    @property
    def final_concentration0_(self):
        """Return the final second shape estimate."""
        return self._canonical(self.final_params)[1]

    @property
    def ema_concentration0_(self):
        """Return the EMA second shape estimate."""
        return self._canonical(self.ema_params)[1]

    @property
    def avg_concentration0_(self):
        """Return the averaged second shape estimate."""
        return self._canonical(self.avg_params)[1]

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated beta prime distribution"
