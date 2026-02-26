# Author: pstefanou12@
"""Truncated multivariate normal distribution with oracle access (known truncation set)."""

import logging
from collections.abc import Callable
from functools import partial

import torch as ch
from torch import nn

from delphi.truncated.distributions.truncated_exponential_family_distributions import (
    TruncatedExponentialFamilyDistribution,
)
from delphi.delphi_logger import delphiLogger
from delphi.distributions.multivariate_normal import (
    ExponentialFamilyMultivariateNormal,
    ExponentialFamilyMultivariateNormalKnownCovariance,
    calc_multi_norm_suff_stat_known_cov,
    calc_multi_norm_suff_stat,
)
from delphi.utils.helpers import Parameters
from delphi.utils.defaults import check_and_fill_args, TRUNC_MULTI_NORM_DEFAULTS


class TruncatedMultivariateNormalKnownCovariance(
    TruncatedExponentialFamilyDistribution
):
    """Truncated multivariate normal distribution with known covariance.

    Attributes:
        covariance_matrix: Known covariance matrix supplied at construction.
    """

    def __init__(
        self,
        args: Parameters,
        phi: Callable,
        alpha: float,
        dims: int,
        covariance_matrix: ch.Tensor | None,
        sampler: Callable = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedMultivariateNormalKnownCovariance.

        Args:
            args: Hyperparameter object.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
            covariance_matrix: Known covariance matrix.
            sampler: Optional sampler override.

        Raises:
            TypeError: If args is not a Parameters instance.
        """
        if not isinstance(args, Parameters):
            raise TypeError(f"args is type {type(args).__name__}; expected Parameters.")
        args = check_and_fill_args(args, TRUNC_MULTI_NORM_DEFAULTS)

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
            calc_multi_norm_suff_stat_known_cov,
            logger,
        )
        self.covariance_matrix = covariance_matrix
        self._sampler = sampler

    def _reparameterize_nat_form(self, theta):
        """Convert canonical mean to natural parameters."""
        inv_cov = self.covariance_matrix.inverse()
        v = theta @ inv_cov  # pylint: disable=invalid-name
        return v.flatten()

    def _reparameterize_canon_form(self, theta):
        """Convert natural parameters to canonical mean."""
        loc = theta @ self.covariance_matrix
        return loc.flatten()

    @property
    def best_loc_(self):
        """Best mean vector estimate based on lowest training loss."""
        return self.best_params

    @property
    def final_loc_(self):
        """Final mean vector estimate at the end of training."""
        return self.final_params

    @property
    def ema_loc_(self):
        """Exponential moving-average mean vector estimate."""
        return self.ema_params

    @property
    def avg_loc_(self):
        """Running-average mean vector estimate."""
        return self.avg_params

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated multivariate normal distribution known covariance"


class TruncatedMultivariateNormalUnknownCovariance(
    TruncatedExponentialFamilyDistribution
):
    """Truncated multivariate normal distribution with unknown covariance.

    Attributes:
        eigenvalue_lower_bound: Minimum eigenvalue for covariance projection.
        emp_canon_params: Empirical canonical parameters computed during fit.
        emp_theta: Empirical natural parameters.
        emp_T: Empirical precision matrix component.
        emp_v: Empirical natural mean component.
    """

    def __init__(
        self,
        args: Parameters,
        phi: Callable,
        alpha: float,
        dims: int,
        sampler: Callable = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedMultivariateNormalUnknownCovariance.

        Args:
            args: Hyperparameter object.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
            sampler: Optional sampler override.

        Raises:
            TypeError: If args is not a Parameters instance.
        """
        if not isinstance(args, Parameters):
            raise TypeError(f"args is type {type(args).__name__}; expected Parameters.")
        args = check_and_fill_args(args, TRUNC_MULTI_NORM_DEFAULTS)
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
            calc_multi_norm_suff_stat,
            logger,
        )
        self._sampler = sampler

        # Attributes set during _calc_emp_model
        self.emp_canon_params = None
        self.emp_theta = None
        self.emp_T = None  # pylint: disable=invalid-name
        self.emp_v = None

    def _calc_emp_model(self):
        """Calculate empirical model parameters and register T and v as nn.Parameters."""
        dataset_s = self.train_loader_.dataset.S  # pylint: disable=invalid-name
        self.emp_canon_params = self.calc_suff_stat(dataset_s).mean(0)
        self.emp_theta = self._reparameterize_nat_form(self.emp_canon_params)
        self.emp_T = self.emp_theta[: self.dims**2].view(  # pylint: disable=invalid-name
            self.dims, self.dims
        )
        self.emp_v = self.emp_theta[self.dims**2 :]
        self.register_parameter("T", nn.Parameter(self.emp_T))
        self.register_parameter("v", nn.Parameter(self.emp_v))
        with ch.no_grad():
            self.nll_init = self._compute_nll(self.emp_theta)

    def project_to_neg_definite(self, M, eps=1e-6):  # pylint: disable=invalid-name
        """Projects a symmetric matrix M onto the Positive Semi-Definite (PSD) cone."""
        L, Q = ch.linalg.eigh(M)  # pylint: disable=invalid-name,not-callable
        L_clipped = ch.clamp(L, max=-eps)  # pylint: disable=invalid-name
        return Q @ ch.diag_embed(L_clipped) @ Q.T  # pylint: disable=invalid-name

    def step_post_hook(self, optimizer, args, kwargs) -> None:
        """Project parameters onto the feasible set after each update.

        Projects T onto the negative-definite cone first (so that the NLL is
        well-defined), then projects the combined (T, v) onto the NLL sublevel
        set via the base-class bisection method.  When args.project is False
        only the cone projection is applied.
        """
        with ch.no_grad():
            # Project T onto the negative-definite cone so that the NLL is
            # computable before calling the sublevel-set projection.
            mat_t = self.T.clone().view(self.dims, self.dims)  # pylint: disable=invalid-name
            mat_t = 0.5 * (mat_t + mat_t.T)
            mat_t = self.project_to_neg_definite(mat_t, eps=self.eigenvalue_lower_bound)
            mat_t = 0.5 * (mat_t + mat_t.T)

            if not self.args.project:
                self.T.copy_(mat_t)
                return

            theta_psd = ch.cat([mat_t.flatten(), self.v.clone()])
            theta_proj = self._project_onto_sublevel_set(theta_psd)

            mat_t_proj = theta_proj[: self.dims**2].view(self.dims, self.dims)
            mat_t_proj = 0.5 * (mat_t_proj + mat_t_proj.T)
            self.T.copy_(mat_t_proj)
            self.v.copy_(theta_proj[self.dims**2 :])

    def parameter_groups(self):
        """Return parameter groups, optionally with separate LR for covariance."""
        if self.args.covariance_matrix_lr is not None:
            return [
                {"params": [self.T], "lr": self.args.covariance_matrix_lr},
                {"params": [self.v], "lr": self.args.lr},
            ]
        return self.parameters()

    def _reparameterize_nat_form(self, theta):
        """Convert canonical parameters to natural form."""
        cov_matrix = theta[: self.dims**2].view(self.dims, self.dims)
        loc = theta[self.dims**2 :]

        mat_t = cov_matrix.inverse()  # pylint: disable=invalid-name
        v = loc @ mat_t  # pylint: disable=invalid-name

        return ch.cat([-0.5 * mat_t.flatten(), v.flatten()])

    def _reparameterize_canon_form(self, theta):
        """Convert natural parameters to canonical form."""
        mat_t = theta[: self.dims**2].view(self.dims, self.dims)  # pylint: disable=invalid-name
        v = theta[self.dims**2 :]

        covariance_matrix = (-2 * mat_t).inverse()
        loc = v @ covariance_matrix

        return ch.cat([covariance_matrix.flatten(), loc.flatten()])

    @property
    def theta(self):
        """Return the current natural parameters as a flat tensor."""
        return ch.cat([self.T.flatten(), self.v])

    @property
    def best_loc_(self):
        """Best mean vector estimate based on lowest training loss."""
        return self.best_params[self.dims**2 :]

    @property
    def best_covariance_matrix_(self):
        """Best covariance matrix estimate based on lowest training loss."""
        return self.best_params[: self.dims**2].view(self.dims, self.dims)

    @property
    def final_loc_(self):
        """Final mean vector estimate at the end of training."""
        return self.final_params[self.dims**2 :]

    @property
    def final_covariance_matrix_(self):
        """Final covariance matrix estimate at the end of training."""
        return self.final_params[: self.dims**2].view(self.dims, self.dims)

    @property
    def ema_loc_(self):
        """Exponential moving-average mean vector estimate."""
        return self.ema_params[self.dims**2 :]

    @property
    def ema_covariance_matrix_(self):
        """Exponential moving-average covariance matrix estimate."""
        return self.ema_params[: self.dims**2].view(self.dims, self.dims)

    @property
    def avg_loc_(self):
        """Running-average mean vector estimate."""
        return self.avg_params[self.dims**2 :]

    @property
    def avg_covariance_matrix_(self):
        """Running-average covariance matrix estimate."""
        return self.avg_params[: self.dims**2].view(self.dims, self.dims)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated multivariate normal distribution"


def TruncatedMultivariateNormal(  # pylint: disable=invalid-name
    args: Parameters,
    phi: Callable,
    alpha: float,
    dims: int,
    covariance_matrix: ch.Tensor | None = None,
    sampler: Callable = None,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Factory function for truncated multivariate normal distributions.

    Returns a known-covariance model if covariance_matrix is provided,
    otherwise returns an unknown-covariance model.

    Args:
        args: Hyperparameter object.
        phi: Truncation set oracle.
        alpha: Survival probability lower bound.
        dims: Number of dimensions.
        covariance_matrix: Known covariance; if None, it is estimated.
        sampler: Optional sampler override.

    Returns:
        TruncatedMultivariateNormalKnownCovariance if covariance_matrix is provided,
        else TruncatedMultivariateNormalUnknownCovariance.

    Raises:
        TypeError: If args is not a Parameters instance.
    """
    if not isinstance(args, Parameters):
        raise TypeError(f"args is type {type(args).__name__}; expected Parameters.")
    args = check_and_fill_args(args, TRUNC_MULTI_NORM_DEFAULTS)
    if covariance_matrix is not None:
        return TruncatedMultivariateNormalKnownCovariance(
            args, phi, alpha, dims, covariance_matrix, sampler
        )
    return TruncatedMultivariateNormalUnknownCovariance(args, phi, alpha, dims, sampler)
