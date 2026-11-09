# Author: pstefanou12@
"""Truncated multinomial distribution with known trials."""

from collections.abc import Callable
from functools import partial
from typing import Optional

import torch as ch
import torch.nn

from delphi.configs.distributions.truncated import (
    exponential_family as exponential_family_configs,
)
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
    defaults = exponential_family_configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("probs_",)

    def __init__(
        self,
        phi: Optional[Callable],
        dims: int,
        total_count: int,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedMultinomial.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            dims: Number of categories.
            total_count: Known number of trials.
            alpha: Survival probability lower bound.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, dims, alpha, hyperparameters=hyperparameters)
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
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    def _canonical(self, theta: ch.Tensor) -> tuple[ch.Tensor, ...]:
        """Return the canonical estimate theta encodes."""
        return (multinomial.Multinomial.to_canonical(theta),)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated multinomial distribution"
