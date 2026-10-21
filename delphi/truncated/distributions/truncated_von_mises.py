# Author: pstefanou12@
"""Truncated von Mises distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import von_mises
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedVonMises(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated von Mises distributions."""

    dist = von_mises.VonMises
    defaults = configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("loc_", "concentration_")
    needs_warm_start = True
    # Each coordinate's two natural parameters are coupled.
    independent_coordinates = False

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedVonMises.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, dims, alpha)

    def _calc_emp_model(self):
        """Start at the circular mean and an approximate concentration.

        The concentration solves ``I1(k) / I0(k) = R``, the mean resultant
        length, which has no closed form. Best and Fisher (1981) approximate
        it piecewise; the warm start then refines both.
        """
        moments = von_mises.VonMises.calc_suff_stat(self.train_loader_.dataset.S).mean(
            0
        )
        cos, sin = moments.chunk(2)
        length = ch.hypot(cos, sin).clamp(max=1 - 1e-3)
        concentration = ch.where(
            length < 0.53,
            2 * length + length**3 + 5 * length**5 / 6,
            ch.where(
                length < 0.85,
                -0.4 + 1.39 * length + 0.43 / (1 - length),
                1 / (length**3 - 4 * length**2 + 3 * length),
            ),
        )
        self.emp_theta = von_mises.VonMises.to_natural(
            ch.cat([ch.atan2(sin, cos), concentration])
        )
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    @property
    def best_loc_(self):
        """Return the best location estimate."""
        return von_mises.VonMises.to_canonical(self.best_params)[: self.dims]

    @property
    def final_loc_(self):
        """Return the final location estimate."""
        return von_mises.VonMises.to_canonical(self.final_params)[: self.dims]

    @property
    def ema_loc_(self):
        """Return the EMA location estimate."""
        return von_mises.VonMises.to_canonical(self.ema_params)[: self.dims]

    @property
    def avg_loc_(self):
        """Return the averaged location estimate."""
        return von_mises.VonMises.to_canonical(self.avg_params)[: self.dims]

    @property
    def best_concentration_(self):
        """Return the best concentration estimate."""
        return von_mises.VonMises.to_canonical(self.best_params)[self.dims :]

    @property
    def final_concentration_(self):
        """Return the final concentration estimate."""
        return von_mises.VonMises.to_canonical(self.final_params)[self.dims :]

    @property
    def ema_concentration_(self):
        """Return the EMA concentration estimate."""
        return von_mises.VonMises.to_canonical(self.ema_params)[self.dims :]

    @property
    def avg_concentration_(self):
        """Return the averaged concentration estimate."""
        return von_mises.VonMises.to_canonical(self.avg_params)[self.dims :]

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated von mises distribution"
