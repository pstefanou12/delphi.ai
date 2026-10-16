# Author: pstefanou12@
"""Truncated multivariate normal with known covariance."""

from collections.abc import Callable
from functools import partial
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import multivariate_normal_known_covariance
from delphi.truncated.distributions import truncated_exponential_family_distributions


class TruncatedMultivariateNormalKnownCovariance(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Truncated multivariate normal distribution with known covariance.

    Attributes:
        covariance_matrix: Known covariance matrix supplied at construction.
    """

    defaults = configs.TruncatedMultivariateNormalConfig

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        covariance_matrix: ch.Tensor,
        alpha: Optional[float] = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedMultivariateNormalKnownCovariance.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
            covariance_matrix: Known covariance matrix.
        """
        super().__init__(hyperparameters, phi, dims, alpha)
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
        """Calculate empirical natural parameters and register theta as an nn.Parameter."""
        dataset_s = self.train_loader_.dataset.S  # pylint: disable=invalid-name
        emp_mean = multivariate_normal_known_covariance.MultivariateNormalKnownCovariance.calc_suff_stat(
            dataset_s
        ).mean(0)
        self.emp_theta = multivariate_normal_known_covariance.MultivariateNormalKnownCovariance.to_natural(
            emp_mean, self.covariance_matrix
        )
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))
        with ch.no_grad():
            self.nll_init = self._compute_nll(self.emp_theta)

    @property
    def best_loc_(self):
        """Best mean vector estimate based on lowest training loss."""
        return multivariate_normal_known_covariance.MultivariateNormalKnownCovariance.to_canonical(
            self.best_params, self.covariance_matrix
        )

    @property
    def final_loc_(self):
        """Final mean vector estimate at the end of training."""
        return multivariate_normal_known_covariance.MultivariateNormalKnownCovariance.to_canonical(
            self.final_params, self.covariance_matrix
        )

    @property
    def ema_loc_(self):
        """Exponential moving-average mean vector estimate."""
        return multivariate_normal_known_covariance.MultivariateNormalKnownCovariance.to_canonical(
            self.ema_params, self.covariance_matrix
        )

    @property
    def avg_loc_(self):
        """Running-average mean vector estimate."""
        return multivariate_normal_known_covariance.MultivariateNormalKnownCovariance.to_canonical(
            self.avg_params, self.covariance_matrix
        )

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated multivariate normal distribution known covariance"
