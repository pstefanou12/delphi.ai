# Author: pstefanou12@
"""Truncated negative binomial distribution with known failures."""

from collections.abc import Callable
from functools import partial
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import negative_binomial
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedNegativeBinomial(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated negative binomial distributions with known failures.

    Attributes:
        total_count (ch.Tensor): Known number of failures, one per coordinate.
    """

    defaults = configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("probs_",)

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        total_count: ch.Tensor,
        alpha: Optional[float] = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedNegativeBinomial.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            total_count: Known number of failures, one per coordinate.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, dims, alpha)
        self.total_count = total_count

    def _build_distribution(self):
        """Bind the known failures into the distribution."""
        return (
            partial(negative_binomial.NegativeBinomial, self.total_count),
            negative_binomial.NegativeBinomial.calc_suff_stat,
        )

    def _calc_emp_model(self):
        """Initialize theta at the MLE ``probs = mean / (total_count + mean)``."""
        mean = self.train_loader_.dataset.S.mean(0)
        emp = (mean / (self.total_count + mean)).clamp(1e-6, 1 - 1e-6)
        self.emp_theta = negative_binomial.NegativeBinomial.to_natural(emp)
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta to be strictly negative, so probs lies in (0, 1)."""
        return ch.clamp(theta, max=-1e-6)

    @property
    def best_probs_(self):
        """Return the best success probability estimate."""
        return negative_binomial.NegativeBinomial.to_canonical(self.best_params)

    @property
    def final_probs_(self):
        """Return the final success probability estimate."""
        return negative_binomial.NegativeBinomial.to_canonical(self.final_params)

    @property
    def ema_probs_(self):
        """Return the EMA success probability estimate."""
        return negative_binomial.NegativeBinomial.to_canonical(self.ema_params)

    @property
    def avg_probs_(self):
        """Return the averaged success probability estimate."""
        return negative_binomial.NegativeBinomial.to_canonical(self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated negative binomial distribution"
