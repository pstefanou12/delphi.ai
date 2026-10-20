# Author: pstefanou12@
"""Truncated multivariate normal distribution with oracle access (known truncation set)."""

from collections.abc import Callable
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import multivariate_normal
from delphi.truncated.distributions import truncated_exponential_family_distributions
from delphi.utils import helpers


class TruncatedMultivariateNormal(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Truncated multivariate normal distribution with unknown covariance.

    Attributes:
        eigenvalue_lower_bound: Minimum eigenvalue for covariance projection.
        emp_T: Empirical precision component, as a lower triangle.
        emp_v: Empirical natural mean component.
    """

    # Created by register_parameter in _calc_emp_model, which mypy cannot see.
    # Holds the lower triangle, so descent moves dims * (dims + 1) / 2 numbers
    # rather than repeating every off-diagonal.
    T: nn.Parameter
    v: nn.Parameter

    dist = multivariate_normal.MultivariateNormal
    # T(x) mixes coordinates through xx^T, so the Hessian is dense and its
    # spectrum needs eigenvalues rather than the diagonal.
    independent_coordinates = False

    defaults = configs.TruncatedMultivariateNormalConfig

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedMultivariateNormal.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
        """
        super().__init__(hyperparameters, phi, dims, alpha)
        self.eigenvalue_lower_bound = self.hyperparameters.eigenvalue_lower_bound

        # Attributes set during _calc_emp_model.
        self.emp_T: Optional[ch.Tensor] = None
        self.emp_v: Optional[ch.Tensor] = None

    def _calc_emp_model(self):
        """Calculate empirical natural parameters and register T and v as nn.Parameters."""
        dataset_s = self.train_loader_.dataset.S
        loc = dataset_s.mean(0)
        second_moment = (dataset_s.T @ dataset_s) / dataset_s.size(0)
        # Center the second moment to get the empirical covariance: Σ = E[xx^T] - μμ^T.
        cov_matrix = second_moment - ch.outer(loc, loc)
        emp_canon_params = ch.cat([cov_matrix.flatten(), loc])
        self.emp_theta = multivariate_normal.MultivariateNormal.to_natural(
            emp_canon_params, self.dims
        )
        size = helpers.vech_size(self.dims)
        self.emp_T = self.emp_theta[:size]
        self.emp_v = self.emp_theta[size:]
        # Clone so that SGD in-place updates to T and v do not corrupt
        # emp_theta, which is used as the fixed anchor in the sublevel-set
        # projection bisection.
        self.register_parameter("T", nn.Parameter(self.emp_T.clone()))
        self.register_parameter("v", nn.Parameter(self.emp_v.clone()))
        with ch.no_grad():
            self.nll_init = self._compute_nll(self.emp_theta)

    def _project_to_neg_definite(self, M, eps=1e-6):
        """Projects a symmetric matrix M onto the negative semi-definite cone."""
        L, Q = ch.linalg.eigh(M)  # pylint: disable=invalid-name,not-callable
        L_clipped = ch.clamp(L, max=-eps)
        return Q @ ch.diag_embed(L_clipped) @ Q.T

    def step_post_hook(self, optimizer, args, kwargs) -> None:
        """Project T onto the negative-definite cone, then delegate to the base class.

        The neg-definite projection must precede the sublevel-set projection
        because the NLL is undefined when the covariance is not positive
        definite.  After correcting T, the base-class hook handles the
        sublevel-set projection (when args.project is True) or returns
        immediately (when args.project is False).
        """
        with ch.no_grad():
            mat_t = helpers.unvech(self.T.detach(), self.dims)
            mat_t = self._project_to_neg_definite(
                mat_t, eps=self.eigenvalue_lower_bound
            )
            self.T.copy_(helpers.vech(mat_t))
        super().step_post_hook(optimizer, args, kwargs)

    def _write_theta(self, value: ch.Tensor) -> None:
        """Split the projected flat theta into T and v and copy to their Parameters."""
        size = helpers.vech_size(self.dims)
        self.T.copy_(value[:size])
        self.v.copy_(value[size:])

    def parameter_groups(self):
        """Return parameter groups, optionally with separate LR for covariance."""
        if self.hyperparameters.covariance_matrix_lr is not None:
            return [
                {"params": [self.T], "lr": self.hyperparameters.covariance_matrix_lr},
                {"params": [self.v], "lr": self.hyperparameters.lr},
            ]
        return self.parameters()

    @property
    def theta(self):
        """Return the current natural parameters as a flat tensor."""
        return ch.cat([self.T, self.v])

    @property
    def best_loc_(self):
        """Best mean vector estimate based on lowest training loss."""
        canon = multivariate_normal.MultivariateNormal.to_canonical(
            self.best_params, self.dims
        )
        return canon[self.dims**2 :]

    @property
    def best_covariance_matrix_(self):
        """Best covariance matrix estimate based on lowest training loss."""
        canon = multivariate_normal.MultivariateNormal.to_canonical(
            self.best_params, self.dims
        )
        return canon[: self.dims**2].view(self.dims, self.dims)

    @property
    def final_loc_(self):
        """Final mean vector estimate at the end of training."""
        canon = multivariate_normal.MultivariateNormal.to_canonical(
            self.final_params, self.dims
        )
        return canon[self.dims**2 :]

    @property
    def final_covariance_matrix_(self):
        """Final covariance matrix estimate at the end of training."""
        canon = multivariate_normal.MultivariateNormal.to_canonical(
            self.final_params, self.dims
        )
        return canon[: self.dims**2].view(self.dims, self.dims)

    @property
    def ema_loc_(self):
        """Exponential moving-average mean vector estimate."""
        canon = multivariate_normal.MultivariateNormal.to_canonical(
            self.ema_params, self.dims
        )
        return canon[self.dims**2 :]

    @property
    def ema_covariance_matrix_(self):
        """Exponential moving-average covariance matrix estimate."""
        canon = multivariate_normal.MultivariateNormal.to_canonical(
            self.ema_params, self.dims
        )
        return canon[: self.dims**2].view(self.dims, self.dims)

    @property
    def avg_loc_(self):
        """Running-average mean vector estimate."""
        canon = multivariate_normal.MultivariateNormal.to_canonical(
            self.avg_params, self.dims
        )
        return canon[self.dims**2 :]

    @property
    def avg_covariance_matrix_(self):
        """Running-average covariance matrix estimate."""
        canon = multivariate_normal.MultivariateNormal.to_canonical(
            self.avg_params, self.dims
        )
        return canon[: self.dims**2].view(self.dims, self.dims)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated multivariate normal distribution"
