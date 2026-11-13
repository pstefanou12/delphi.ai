# Author: pstefanou12@
"""Truncated beta prime distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
import torch.distributions.transforms
import torch.nn

from delphi.configs.distributions.truncated import gamma_unknown_concentration
from delphi.exponential_family import beta_prime
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedBetaPrime(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated beta prime distributions."""

    family = beta_prime.BetaPrime
    # log x and log(1 + x) are nearly collinear, which preconditioning undoes.
    defaults = gamma_unknown_concentration.TruncatedGammaUnknownConcentrationConfig
    estimate_suffixes = ("concentration1_", "concentration0_")
    # Each coordinate's log x and log(1 + x) are correlated.
    independent_coordinates = False

    def __init__(
        self,
        phi: Optional[Callable],
        dims: int,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize TruncatedBetaPrime.

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
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Keep both shapes positive."""
        first = theta[: self.dims].clamp(min=-1.0 + 1e-6)
        second = ch.minimum(theta[self.dims :], -first - 1.0 - 1e-6)
        return ch.cat([first, second])

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated beta prime distribution"
