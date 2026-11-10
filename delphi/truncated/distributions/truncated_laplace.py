# Author: pstefanou12@
"""Truncated Laplace distribution with known location."""

from collections.abc import Callable
from functools import partial
from typing import Optional

import torch as ch
import torch.nn

from delphi.configs.distributions.truncated import (
    exponential_family as exponential_family_configs,
)
from delphi.exponential_family import laplace
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedLaplace(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Laplace distributions with known location.

    Attributes:
        loc (ch.Tensor): Known location, one per coordinate.
    """

    defaults = exponential_family_configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("scale_",)

    def __init__(
        self,
        phi: Optional[Callable],
        dims: int,
        loc: ch.Tensor,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedLaplace.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            dims: Number of dimensions.
            loc: Known location, one per coordinate.
            alpha: Survival probability lower bound.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, dims, alpha, hyperparameters=hyperparameters)
        self.loc = loc

    def _build_distribution(self):
        """Bind the known location into the distribution and its statistic."""
        return (
            partial(laplace.Laplace, self.loc),
            partial(laplace.Laplace.calc_suff_stat, self.loc),
        )

    def _calc_emp_model(self):
        """Initialize theta at the MLE scale ``mean(|x - loc|)``."""
        emp = self.calc_suff_stat(self.train_loader_.dataset.S).mean(0)
        self.emp_theta = laplace.Laplace.to_natural(emp)
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta to be strictly negative, so the scale is positive."""
        return ch.clamp(theta, max=-1e-6)

    def _canonical(self, theta: ch.Tensor) -> tuple[ch.Tensor, ...]:
        """Return the canonical estimate theta encodes."""
        return (laplace.Laplace.to_canonical(theta),)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated laplace distribution"
