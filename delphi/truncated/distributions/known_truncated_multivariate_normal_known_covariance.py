# Author: pstefanou12@
"""Truncated multivariate normal with known covariance."""

from collections.abc import Callable
from functools import partial
from typing import Optional

import torch as ch
import torch.nn

from delphi.configs.distributions.truncated import multivariate_normal
from delphi.exponential_family import multivariate_normal_known_covariance
from delphi.truncated.distributions import truncated_exponential_family_distributions


class KnownTruncationMultivariateNormalKnownCovariance(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Truncated multivariate normal distribution with known covariance.

    Attributes:
        covariance_matrix: Known covariance matrix supplied at construction.
    """

    defaults = multivariate_normal.TruncatedMultivariateNormalConfig
    estimate_suffixes = ("loc_",)

    def __init__(
        self,
        phi: Optional[Callable],
        dims: int,
        covariance_matrix: ch.Tensor,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize KnownTruncationMultivariateNormalKnownCovariance.

        Args:
            phi: Truncation set oracle; None learns it from the sample.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
            covariance_matrix: Known covariance matrix.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(phi, dims, alpha, hyperparameters=hyperparameters)
        self.covariance_matrix = covariance_matrix

    def _build_distribution(self):
        """Bind the known covariance into the distribution.

        Returns:
            The factory closed over the covariance, and the sufficient
            statistic of the family.
        """
        return (
            partial(
                multivariate_normal_known_covariance.MultivariateNormalKnownCovariance,
                self.covariance_matrix,
            ),
            multivariate_normal_known_covariance.MultivariateNormalKnownCovariance.calc_suff_stat,
        )

    def _calc_emp_model(self):
        """Calculate empirical natural parameters and register theta as an torch.nn.Parameter."""
        dataset_s = self.train_loader_.dataset.S  # pylint: disable=invalid-name
        emp_mean = multivariate_normal_known_covariance.MultivariateNormalKnownCovariance.calc_suff_stat(
            dataset_s
        ).mean(0)
        self.emp_theta = multivariate_normal_known_covariance.MultivariateNormalKnownCovariance.to_natural(
            emp_mean, self.covariance_matrix
        )
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))
        with ch.no_grad():
            self.nll_init = self._compute_nll(self.emp_theta)

    def _canonical(self, theta: ch.Tensor) -> tuple[ch.Tensor, ...]:
        """Return the mean, which the known covariance enters."""
        return (
            multivariate_normal_known_covariance.MultivariateNormalKnownCovariance.to_canonical(
                theta, self.covariance_matrix
            ),
        )

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated multivariate normal distribution known covariance"
