# Author: pstefanou12@
"""Truncated multivariate normal with known covariance."""

import logging
from collections.abc import Callable
from functools import partial

import torch as ch
from torch import nn, Tensor

from delphi.truncated.distributions.truncated_exponential_family_distributions import (
    TruncatedExponentialFamilyDistribution,
)
from delphi.truncated.distributions.truncated_multivariate_normal import (
    TruncatedMultivariateNormalConfig,
)
from delphi.utils.configs import make_config
from delphi.delphi_logger import delphiLogger
from delphi.distributions.multivariate_normal import (
    ExponentialFamilyMultivariateNormalKnownCovariance,
)


class TruncatedMultivariateNormalKnownCovariance(
    TruncatedExponentialFamilyDistribution
):
    """Truncated multivariate normal distribution with known covariance.

    Attributes:
        covariance_matrix: Known covariance matrix supplied at construction.
    """

    def __init__(
        self,
        args: dict | TruncatedMultivariateNormalConfig,
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
        args = make_config(args, TruncatedMultivariateNormalConfig)
        logger = (
            delphiLogger() if args.verbose else delphiLogger(level=logging.CRITICAL)
        )
        super().__init__(
            args,
            phi,
            alpha,
            dims,
            partial(
                ExponentialFamilyMultivariateNormalKnownCovariance, covariance_matrix
            ),
            logger,
        )
        self.covariance_matrix = covariance_matrix
        self._sampler = sampler

    @staticmethod
    def _calc_suff_stat(x: Tensor) -> Tensor:
        """Compute sufficient statistics for multivariate normal with known covariance."""
        return ExponentialFamilyMultivariateNormalKnownCovariance.calc_suff_stat(x)

    def _calc_emp_model(self):
        """Calculate empirical natural parameters and register theta as an nn.Parameter."""
        dataset_s = self.train_loader_.dataset.S  # pylint: disable=invalid-name
        emp_mean = self._calc_suff_stat(dataset_s).mean(0)
        self.emp_theta = ExponentialFamilyMultivariateNormalKnownCovariance.to_natural(
            emp_mean, self.covariance_matrix
        )
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))
        with ch.no_grad():
            self.nll_init = self._compute_nll(self.emp_theta)

    @property
    def best_loc_(self):
        """Best mean vector estimate based on lowest training loss."""
        return ExponentialFamilyMultivariateNormalKnownCovariance.to_canonical(
            self.best_params, self.covariance_matrix
        )

    @property
    def final_loc_(self):
        """Final mean vector estimate at the end of training."""
        return ExponentialFamilyMultivariateNormalKnownCovariance.to_canonical(
            self.final_params, self.covariance_matrix
        )

    @property
    def ema_loc_(self):
        """Exponential moving-average mean vector estimate."""
        return ExponentialFamilyMultivariateNormalKnownCovariance.to_canonical(
            self.ema_params, self.covariance_matrix
        )

    @property
    def avg_loc_(self):
        """Running-average mean vector estimate."""
        return ExponentialFamilyMultivariateNormalKnownCovariance.to_canonical(
            self.avg_params, self.covariance_matrix
        )

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated multivariate normal distribution known covariance"
