# Author: pstefanou12@
"""Truncated Pareto distribution with known scale."""

from collections.abc import Callable
from functools import partial
from typing import Optional

import torch as ch
import torch.distributions.transforms
import torch.nn

from delphi.configs.distributions.truncated import (
    exponential_family as exponential_family_configs,
)
from delphi.exponential_family import pareto
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedPareto(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Pareto distributions with known scale.

    Attributes:
        scale (ch.Tensor): Known scale, the lower end of the support, one per
            coordinate.
    """

    defaults = exponential_family_configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("alpha_",)

    def __init__(
        self,
        phi: Optional[Callable],
        dims: int,
        scale: ch.Tensor,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedPareto.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            dims: Number of dimensions.
            scale: Known scale, one per coordinate.
            alpha: Survival probability lower bound. Unrelated to the tail
                index, which the estimates call ``alpha_``.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, dims, alpha, hyperparameters=hyperparameters)
        self.scale = scale

    def _build_distribution(self):
        """Bind the known scale into the distribution."""
        return partial(pareto.Pareto, self.scale), pareto.Pareto.calc_suff_stat

    def _truncation_set_transform(self) -> torch.distributions.transforms.Transform:
        """Learn the set in log x, where the heavy right tail is tame."""
        return torch.distributions.transforms.ExpTransform().inv

    def _calc_emp_model(self):
        """Initialize theta at the MLE tail index ``1 / mean(log(x / scale))``."""
        dataset_s = self.train_loader_.dataset.S
        emp_alpha = 1.0 / (ch.log(dataset_s) - ch.log(self.scale)).mean(0)
        self.emp_theta = pareto.Pareto.to_natural(emp_alpha)
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta below -1, so the tail index is positive."""
        return ch.clamp(theta, max=-1.0 - 1e-6)

    def _canonical(self, theta: ch.Tensor) -> tuple[ch.Tensor, ...]:
        """Return the canonical estimate theta encodes."""
        return (pareto.Pareto.to_canonical(theta),)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated pareto distribution"
