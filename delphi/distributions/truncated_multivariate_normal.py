"""
Truncated multivariate normal distribution with oracle access (ie. known truncation set).
"""

import logging
from functools import partial
from typing import Callable, Optional

import torch as ch
from torch import nn

from delphi.distributions.truncated_exponential_family_distributions import (
    TruncatedExponentialFamilyDistribution,
)
from delphi.delphi_logger import delphiLogger
from delphi.grad import (
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
    """
    Truncated multivariate normal distribution class with known truncation set.
    """

    def __init__(
        self,
        args: Parameters,
        phi: Callable,
        alpha: float,
        dims: int,
        covariance_matrix: Optional[ch.Tensor],
        sampler: Callable = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """
        Initialize TruncatedMultivariateNormalKnownCovariance.

        Args:
            args (Parameters): hyperparameter object
            phi (Callable): truncation set oracle
            alpha (float): survival probability lower bound
            dims (int): number of dimensions
            covariance_matrix (Optional[Tensor]): known covariance matrix
            sampler (Callable): optional sampler override
        """
        assert isinstance(args, Parameters), (
            "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
        )
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
        """
        Returns the best mean vector estimate for the multivariate normal
        distribution based off of the loss function.
        """
        return self.best_params

    @property
    def final_loc_(self):
        """
        Returns the final mean vector estimate for the multivariate normal
        distribution based off of the loss function.
        """
        return self.final_params

    @property
    def ema_loc_(self):
        """
        Returns the ema mean vector estimate for the multivariate normal
        distribution based off of the loss function.
        """
        return self.ema_params

    @property
    def avg_loc_(self):
        """
        Returns the avg mean vector estimate for the multivariate normal
        distribution based off of the loss function.
        """
        return self.avg_params

    def calc_suff_stat(self, S):  # pylint: disable=invalid-name,method-hidden
        """Compute sufficient statistics for the known-covariance case."""
        return calc_multi_norm_suff_stat_known_cov(S)

    def calculate_loss(self, S):  # pylint: disable=invalid-name
        """Compute the loss given a batch of samples S."""
        return self.criterion(
            self.theta, self.calc_suff_stat(S), *self.criterion_params
        )

    def __str__(self):
        return "truncated multivariate normal distribution known covariance"


class TruncatedMultivariateNormalUnknownCovariance(
    TruncatedExponentialFamilyDistribution
):
    """
    Truncated multivariate normal distribution class with unknown covariance.
    """

    def __init__(
        self,
        args: Parameters,
        phi: Callable,
        alpha: float,
        dims: int,
        sampler: Callable = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """
        Initialize TruncatedMultivariateNormalUnknownCovariance.

        Args:
            args (Parameters): hyperparameter object
            phi (Callable): truncation set oracle
            alpha (float): survival probability lower bound
            dims (int): number of dimensions
            sampler (Callable): optional sampler override
        """
        assert isinstance(args, Parameters), (
            "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
        )
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

    def project_to_neg_definite(self, M, eps=1e-6):  # pylint: disable=invalid-name
        """Projects a symmetric matrix M onto the Positive Semi-Definite (PSD) cone."""
        L, Q = ch.linalg.eigh(M)  # pylint: disable=invalid-name,not-callable
        L_clipped = ch.clamp(L, max=-eps)  # pylint: disable=invalid-name
        return Q @ ch.diag_embed(L_clipped) @ Q.T  # pylint: disable=invalid-name

    def step_post_hook(self, optimizer, args, kwargs) -> None:
        """
        Iteration hook called after each training update.
        Projects parameters back into the feasible set.
        """
        with ch.no_grad():
            mat_t = self.T.clone().view(self.dims, self.dims)  # pylint: disable=invalid-name
            v = self.v.clone()

            # --- 1) Project mean into L2 ball ---
            loc_diff = self.emp_v - v
            dist = ch.norm(loc_diff)

            if dist > self.radius:
                v = self.emp_v - loc_diff / dist * self.radius

            # --- 2) Frobenius ball projection for T ---
            cov_diff = mat_t - self.emp_T
            frob_norm = ch.linalg.norm(cov_diff, ord="fro")  # pylint: disable=not-callable

            if frob_norm > self.radius:
                mat_t = self.emp_T + cov_diff * (self.radius / frob_norm)

            # Symmetrize after projection (important!)
            mat_t = 0.5 * (mat_t + mat_t.T)
            # --- 3) Final PSD projection ---
            mat_t = self.project_to_neg_definite(mat_t, eps=self.eigenvalue_lower_bound)
            # Symmetrize again after PSD projection
            mat_t = 0.5 * (mat_t + mat_t.T)

            self.T.copy_(mat_t)
            self.v.copy_(v)

    def parameters_(self):
        """Return parameter groups, optionally with separate LR for covariance."""
        if self.args.covariance_matrix_lr is not None:
            return [
                {"params": self.T, "lr": self.args.covariance_matrix_lr},
                {"params": self.v, "lr": self.args.lr},
            ]
        return self.parameters()

    def _reparameterize_nat_form(self, theta):
        """Convert canonical parameters to natural form."""
        cov_matrix = theta[: self.dims**2].resize(self.dims, self.dims)
        loc = theta[self.dims**2 :]

        mat_t = cov_matrix.inverse()  # pylint: disable=invalid-name
        v = loc @ mat_t  # pylint: disable=invalid-name

        return ch.cat([-0.5 * mat_t.flatten(), v.flatten()])

    def _reparameterize_canon_form(self, theta):
        """Convert natural parameters to canonical form."""
        mat_t = theta[: self.dims**2].resize(self.dims, self.dims)  # pylint: disable=invalid-name
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
        """
        Returns the best mean vector estimate for the multivariate normal
        distribution based off of the loss function.
        """
        return self.best_params[self.dims**2 :]

    @property
    def best_covariance_matrix_(self):
        """
        Returns the best covariance matrix estimate for the multivariate normal
        distribution based off of the loss function.
        """
        return self.best_params[: self.dims**2].view(self.dims, self.dims)

    @property
    def final_loc_(self):
        """
        Returns the final mean vector estimate for the multivariate normal
        distribution based off of the loss function.
        """
        return self.final_params[self.dims**2 :]

    @property
    def final_covariance_matrix_(self):
        """
        Returns the final covariance matrix estimate for the multivariate normal
        distribution based off of the loss function.
        """
        self.final_params[: self.dims**2].view(self.dims, self.dims)

    @property
    def ema_loc_(self):
        """
        Returns the ema mean vector estimate for the multivariate normal
        distribution based off of the loss function.
        """
        return self.ema_params[self.dims**2 :]

    @property
    def ema_covariance_matrix_(self):
        """
        Returns the ema covariance matrix estimate for the multivariate normal
        distribution based off of the loss function.
        """
        return self.ema_params[: self.dims**2].view(self.dims, self.dims)

    @property
    def avg_loc_(self):
        """
        Returns the avg mean vector estimate for the multivariate normal
        distribution based off of the loss function.
        """
        return self.avg_params[self.dims**2 :]

    @property
    def avg_covariance_matrix_(self):
        """
        Returns the avg covariance matrix estimate for the multivariate normal
        distribution based off of the loss function.
        """
        return self.avg_params[: self.dims**2].view(self.dims, self.dims)

    def calc_suff_stat(self, S):  # pylint: disable=invalid-name,method-hidden
        """Compute sufficient statistics for the unknown-covariance case."""
        return calc_multi_norm_suff_stat(S)

    def calculate_loss(self, S):  # pylint: disable=invalid-name
        """Compute the loss given a batch of samples S."""
        return self.criterion(
            self.theta, self.calc_suff_stat(S), *self.criterion_params
        )

    def __str__(self):
        return "truncated multivariate normal distribution"


def TruncatedMultivariateNormal(  # pylint: disable=invalid-name
    args: Parameters,
    phi: Callable,
    alpha: float,
    dims: int,
    covariance_matrix: Optional[ch.Tensor] = None,
    sampler: Callable = None,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """
    Factory function for truncated multivariate normal distributions.

    Returns a known-covariance model if covariance_matrix is provided,
    otherwise returns an unknown-covariance model.
    """
    assert isinstance(args, Parameters), (
        "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
    )
    args = check_and_fill_args(args, TRUNC_MULTI_NORM_DEFAULTS)
    if covariance_matrix is not None:
        return TruncatedMultivariateNormalKnownCovariance(
            args, phi, alpha, dims, covariance_matrix, sampler
        )
    return TruncatedMultivariateNormalUnknownCovariance(args, phi, alpha, dims, sampler)
