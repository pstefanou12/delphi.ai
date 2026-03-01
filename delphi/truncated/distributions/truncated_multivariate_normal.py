# Author: pstefanou12@
"""Truncated multivariate normal distribution with oracle access (known truncation set)."""

import logging
from collections.abc import Callable

import torch as ch
from torch import nn, Tensor

from delphi.truncated.distributions.truncated_exponential_family_distributions import (
    TruncatedExponentialFamilyDistribution,
)
from delphi.truncated.distributions.truncated_multivariate_normal_known_covariance import (
    TruncatedMultivariateNormalConfig,
)
from delphi.utils.configs import make_config
from delphi.delphi_logger import delphiLogger
from delphi.distributions.multivariate_normal import ExponentialFamilyMultivariateNormal


class TruncatedMultivariateNormal(TruncatedExponentialFamilyDistribution):
    """Truncated multivariate normal distribution with unknown covariance.

    Attributes:
        eigenvalue_lower_bound: Minimum eigenvalue for covariance projection.
        emp_theta: Empirical natural parameters.
        emp_T: Empirical precision matrix component.
        emp_v: Empirical natural mean component.
    """

    def __init__(
        self,
        args: dict | TruncatedMultivariateNormalConfig,
        phi: Callable,
        alpha: float,
        dims: int,
        sampler: Callable = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedMultivariateNormal.

        Args:
            args: Hyperparameter dict or Pydantic config.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
            sampler: Optional sampler override.
        """
        args = make_config(args, TruncatedMultivariateNormalConfig)
        self.eigenvalue_lower_bound = args.eigenvalue_lower_bound

        logger = (
            delphiLogger() if args.verbose else delphiLogger(level=logging.CRITICAL)
        )
        super().__init__(
            args,
            phi,
            alpha,
            dims,
            ExponentialFamilyMultivariateNormal,
            logger,
        )
        self._sampler = sampler

        # Attributes set during _calc_emp_model.
        self.emp_theta = None
        self.emp_T = None  # pylint: disable=invalid-name
        self.emp_v = None

    @staticmethod
    def _calc_suff_stat(x: Tensor) -> Tensor:
        """Compute sufficient statistics for multivariate normal."""
        return ExponentialFamilyMultivariateNormal.calc_suff_stat(x)

    def _calc_emp_model(self):
        """Calculate empirical natural parameters and register T and v as nn.Parameters."""
        dataset_s = self.train_loader_.dataset.S  # pylint: disable=invalid-name
        suff_stats = self._calc_suff_stat(dataset_s).mean(0)
        second_moment = suff_stats[: self.dims**2].view(self.dims, self.dims)
        loc = suff_stats[self.dims**2 :]
        # Center the second moment to get the empirical covariance: Σ = E[xx^T] - μμ^T.
        cov_matrix = second_moment - ch.outer(loc, loc)
        emp_canon_params = ch.cat([cov_matrix.flatten(), loc])
        self.emp_theta = ExponentialFamilyMultivariateNormal.to_natural(
            emp_canon_params, self.dims
        )
        self.emp_T = self.emp_theta[: self.dims**2].view(  # pylint: disable=invalid-name
            self.dims, self.dims
        )
        self.emp_v = self.emp_theta[self.dims**2 :]
        # Clone so that SGD in-place updates to T and v do not corrupt
        # emp_theta, which is used as the fixed anchor in the sublevel-set
        # projection bisection.
        self.register_parameter("T", nn.Parameter(self.emp_T.clone()))
        self.register_parameter("v", nn.Parameter(self.emp_v.clone()))
        with ch.no_grad():
            self.nll_init = self._compute_nll(self.emp_theta)

    def _project_to_neg_definite(self, M, eps=1e-6):  # pylint: disable=invalid-name
        """Projects a symmetric matrix M onto the negative semi-definite cone."""
        L, Q = ch.linalg.eigh(M)  # pylint: disable=invalid-name,not-callable
        L_clipped = ch.clamp(L, max=-eps)  # pylint: disable=invalid-name
        return Q @ ch.diag_embed(L_clipped) @ Q.T  # pylint: disable=invalid-name

    def step_post_hook(self, optimizer, args, kwargs) -> None:
        """Project T onto the negative-definite cone, then delegate to the base class.

        The neg-definite projection must precede the sublevel-set projection
        because the NLL is undefined when the covariance is not positive
        definite.  After correcting T, the base-class hook handles the
        sublevel-set projection (when args.project is True) or returns
        immediately (when args.project is False).
        """
        with ch.no_grad():
            mat_t = self.T.clone().view(self.dims, self.dims)  # pylint: disable=invalid-name
            mat_t = 0.5 * (mat_t + mat_t.T)
            mat_t = self._project_to_neg_definite(
                mat_t, eps=self.eigenvalue_lower_bound
            )
            mat_t = 0.5 * (mat_t + mat_t.T)
            self.T.copy_(mat_t)
        super().step_post_hook(optimizer, args, kwargs)

    def _write_theta(self, value: Tensor) -> None:
        """Split the projected flat theta into T and v and copy to their Parameters."""
        mat_t = value[: self.dims**2].view(self.dims, self.dims)  # pylint: disable=invalid-name
        mat_t = 0.5 * (mat_t + mat_t.T)
        self.T.copy_(mat_t)
        self.v.copy_(value[self.dims**2 :])

    def parameter_groups(self):
        """Return parameter groups, optionally with separate LR for covariance."""
        if self.args.covariance_matrix_lr is not None:
            return [
                {"params": [self.T], "lr": self.args.covariance_matrix_lr},
                {"params": [self.v], "lr": self.args.lr},
            ]
        return self.parameters()

    @property
    def theta(self):
        """Return the current natural parameters as a flat tensor."""
        return ch.cat([self.T.flatten(), self.v])

    @property
    def best_loc_(self):
        """Best mean vector estimate based on lowest training loss."""
        canon = ExponentialFamilyMultivariateNormal.to_canonical(
            self.best_params, self.dims
        )
        return canon[self.dims**2 :]

    @property
    def best_covariance_matrix_(self):
        """Best covariance matrix estimate based on lowest training loss."""
        canon = ExponentialFamilyMultivariateNormal.to_canonical(
            self.best_params, self.dims
        )
        return canon[: self.dims**2].view(self.dims, self.dims)

    @property
    def final_loc_(self):
        """Final mean vector estimate at the end of training."""
        canon = ExponentialFamilyMultivariateNormal.to_canonical(
            self.final_params, self.dims
        )
        return canon[self.dims**2 :]

    @property
    def final_covariance_matrix_(self):
        """Final covariance matrix estimate at the end of training."""
        canon = ExponentialFamilyMultivariateNormal.to_canonical(
            self.final_params, self.dims
        )
        return canon[: self.dims**2].view(self.dims, self.dims)

    @property
    def ema_loc_(self):
        """Exponential moving-average mean vector estimate."""
        canon = ExponentialFamilyMultivariateNormal.to_canonical(
            self.ema_params, self.dims
        )
        return canon[self.dims**2 :]

    @property
    def ema_covariance_matrix_(self):
        """Exponential moving-average covariance matrix estimate."""
        canon = ExponentialFamilyMultivariateNormal.to_canonical(
            self.ema_params, self.dims
        )
        return canon[: self.dims**2].view(self.dims, self.dims)

    @property
    def avg_loc_(self):
        """Running-average mean vector estimate."""
        canon = ExponentialFamilyMultivariateNormal.to_canonical(
            self.avg_params, self.dims
        )
        return canon[self.dims**2 :]

    @property
    def avg_covariance_matrix_(self):
        """Running-average covariance matrix estimate."""
        canon = ExponentialFamilyMultivariateNormal.to_canonical(
            self.avg_params, self.dims
        )
        return canon[: self.dims**2].view(self.dims, self.dims)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated multivariate normal distribution"
