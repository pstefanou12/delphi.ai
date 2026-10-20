# Author: pstefanou12@
"""Truncated binomial distribution with known trials."""

from collections.abc import Callable
from functools import partial
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import binomial
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedBinomial(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated binomial distributions with known trials.

    Attributes:
        total_count (ch.Tensor): Known number of trials, one per coordinate.
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
        """Initialize TruncatedBinomial.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            total_count: Known number of trials, one per coordinate.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, dims, alpha)
        self.total_count = total_count

    def _build_distribution(self):
        """Bind the known trials into the distribution."""
        return (
            partial(binomial.Binomial, self.total_count),
            binomial.Binomial.calc_suff_stat,
        )

    def _calc_emp_model(self):
        """Initialize theta at the MLE ``logit(mean / total_count)``."""
        mean = self.train_loader_.dataset.S.mean(0)
        # A coordinate pinned to 0 or total_count would give an infinite logit.
        emp = (mean / self.total_count).clamp(1e-6, 1 - 1e-6)
        self.emp_theta = binomial.Binomial.to_natural(emp)
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    @property
    def best_probs_(self):
        """Return the best success probability estimate."""
        return binomial.Binomial.to_canonical(self.best_params)

    @property
    def final_probs_(self):
        """Return the final success probability estimate."""
        return binomial.Binomial.to_canonical(self.final_params)

    @property
    def ema_probs_(self):
        """Return the EMA success probability estimate."""
        return binomial.Binomial.to_canonical(self.ema_params)

    @property
    def avg_probs_(self):
        """Return the averaged success probability estimate."""
        return binomial.Binomial.to_canonical(self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated binomial distribution"
