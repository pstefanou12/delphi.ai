# Author: pstefanou12@
"""Truncated multinomial distribution with known trials."""

from collections.abc import Callable
from functools import partial
from typing import Optional

from torch import nn

from delphi import configs
from delphi.exponential_family import multinomial
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedMultinomial(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated multinomial distributions with known trials.

    Attributes:
        total_count (int): Known number of trials.
    """

    # The categories share one normalizer, so they are coupled.
    independent_coordinates = False
    defaults = configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("probs_",)

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        total_count: int,
        alpha: Optional[float] = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedMultinomial.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of categories.
            total_count: Known number of trials.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, dims, alpha)
        self.total_count = total_count

    def _build_distribution(self):
        """Bind the known trials into the distribution."""
        return partial(
            multinomial.Multinomial, self.total_count
        ), multinomial.Multinomial.calc_suff_stat

    def _calc_emp_model(self):
        """Initialize theta at the MLE ``probs = mean(counts) / total_count``."""
        dataset_s = self.train_loader_.dataset.S
        # A category never observed would give an infinite log-ratio.
        probs = (dataset_s.mean(0) / self.total_count).clamp(min=1e-6)
        self.emp_theta = multinomial.Multinomial.to_natural(probs / probs.sum())
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    @property
    def best_probs_(self):
        """Return the best probs estimate."""
        return multinomial.Multinomial.to_canonical(self.best_params)

    @property
    def final_probs_(self):
        """Return the final probs estimate."""
        return multinomial.Multinomial.to_canonical(self.final_params)

    @property
    def ema_probs_(self):
        """Return the EMA probs estimate."""
        return multinomial.Multinomial.to_canonical(self.ema_params)

    @property
    def avg_probs_(self):
        """Return the averaged probs estimate."""
        return multinomial.Multinomial.to_canonical(self.avg_params)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated multinomial distribution"
