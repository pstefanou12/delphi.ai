# Author: pstefanou12@
"""Truncated multivariate normal with known covariance."""

from collections.abc import Callable
from functools import partial
import logging

import torch as ch
from torch import nn

from delphi import delphi_logger
from delphi.exponential_family import multivariate_normal_known_covariance
from delphi.truncated.distributions import truncated_exponential_family_distributions
from delphi.utils import configs


class TruncatedMultivariateNormalKnownCovariance(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Truncated multivariate normal distribution with known covariance.

    Attributes:
        covariance_matrix: Known covariance matrix supplied at construction.
    """

    def __init__(
        self,
        args: dict | configs.TruncatedMultivariateNormalConfig,
        phi: Callable,
        alpha: float,
        dims: int,
        covariance_matrix: ch.Tensor | None,
        sampler: Callable = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedMultivariateNormalKnownCovariance.

        Args:
            args: Hyperparameter dict or Pydantic config.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
            covariance_matrix: Known covariance matrix.
            sampler: Optional sampler override.
        """
        args = configs.make_config(args, configs.TruncatedMultivariateNormalConfig)
        logger = (
            delphi_logger.delphiLogger()
            if args.verbose
            else delphi_logger.delphiLogger(level=logging.CRITICAL)
        )
        self.covariance_matrix = covariance_matrix
        self.dist = partial(
            multivariate_normal_known_covariance.MultivariateNormalKnownCovariance,
            covariance_matrix,
        )
        self.dist.calc_suff_stat = multivariate_normal_known_covariance.MultivariateNormalKnownCovariance.calc_suff_stat
        super().__init__(
            args,
            phi,
            alpha,
            dims,
            logger,
        )
        self._sampler = sampler

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
