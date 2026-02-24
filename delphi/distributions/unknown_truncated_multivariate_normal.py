"""
Truncated multivariate normal distribution without oracle access (ie. unknown truncation set)
"""

# pylint: disable=duplicate-code

from functools import partial
from typing import Optional

import torch as ch
from torch import Tensor
from torch import nn

from .truncated_multivariate_normal import (
    TruncatedMultivariateNormalUnknownCovariance,
    TruncatedMultivariateNormalKnownCovariance,
)
from ..oracle import UnknownGaussian
from ..trainer import Trainer
from ..grad import UnknownTruncationMultivariateNormalNLL
from ..utils.datasets import UnknownTruncationNormalDataset, make_train_and_val_distr
from ..utils.helpers import Parameters, cov
from ..utils.defaults import check_and_fill_args, UNKNOWN_TRUNC_MULTI_NORM_DEFAULTS


class UnknownTruncationMultivariateNormalKnownCovariance(  # pylint: disable=too-many-instance-attributes
    TruncatedMultivariateNormalKnownCovariance
):
    """
    Truncated multivariate normal distribution class with known covariance
    and unknown truncation set.
    """

    def __init__(
        self,
        args: Parameters,
        k: int,
        alpha: float,
        dims: int,
        covariance_matrix: Optional[ch.Tensor],
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """
        Initialize UnknownTruncationMultivariateNormalKnownCovariance.

        Args:
            args (Parameters): hyperparameter object
            k (int): number of nearest neighbors for oracle
            alpha (float): survival probability lower bound
            dims (int): number of dimensions
            covariance_matrix (Optional[Tensor]): known covariance matrix
        """
        assert isinstance(args, Parameters), (
            f"args is type {type(args)}. expecting type delphi.utils.helper.Parameters."
        )
        # algorithm hyperparameters
        self.k = k
        self.args = check_and_fill_args(args, UNKNOWN_TRUNC_MULTI_NORM_DEFAULTS)
        super().__init__(
            args,
            partial(UnknownGaussian, k),
            alpha,
            dims,
            covariance_matrix=covariance_matrix,
        )

        self.emp_loc, self.emp_covariance_matrix = None, None
        self.criterion = UnknownTruncationMultivariateNormalNLL.apply

        # Attributes initialized during fit
        self.train_loader_ = None
        self.val_loader_ = None
        self.exp_h = None
        self.prev_best_loss = None
        self.radius_history = None
        self.loss_history = None
        self.radius = None
        self.trainer = None
        self.prev_theta = None
        self.prev_loss = None

    def fit(self, S: Tensor):
        """
        Fit the model to the observed (truncated) samples S.

        Args:
            S (Tensor): observed samples of shape (num_samples, dims)
        """
        assert isinstance(S, Tensor), (
            f"S is type: {type(S)}. expected type torch.Tensor."
        )
        assert S.size(0) > S.size(1), (
            "input expected to be shape num samples by dimenions, "
            f"current input is size {S.size()}."
        )
        self.train_loader_, self.val_loader_ = make_train_and_val_distr(
            self.args, S, UnknownTruncationNormalDataset
        )

        # verify that the S is whitened to N(0, I)
        emp_loc = S.mean(0)
        emp_cov = cov(S)
        if (
            ch.norm(emp_loc - ch.zeros(S.size(1))) >= 1e-3
            or ch.norm(emp_cov - ch.eye(S.size(1))) >= 1e-3
        ):
            raise ValueError(
                f"input dataset must be whitened (eg. N(O, I)). \n "
                f"dataset mean: {emp_loc}, covariance matrix: {emp_cov}"
            )
        self.phi = self.phi(S)
        self.exp_h = ExpH(emp_loc, emp_cov)
        self.criterion_params = [self.phi, self.exp_h, self.dims]

        # Initialize tracking
        self.prev_best_loss = None
        self.radius_history = []
        self.loss_history = []
        # Initialize radius and parameters
        self._calc_emp_model()
        self.radius = self.args.min_radius

        phase = 0
        while phase < self.args.max_phases:
            phase += 1
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"phase {phase}: training with radius={self.radius:.4f}")
            self.logger.info(f"\n{'=' * 60}")

            self.trainer = Trainer(self, self.args, self.logger)
            self.trainer.train_model(self.train_loader_, self.val_loader_)

            # Update tracking
            current_loss = self.trainer.best_loss
            self.radius_history.append(self.radius)
            self.loss_history.append(current_loss)

            self.best_params, self.best_loss = (
                self._reparameterize_canon_form(self.trainer.best_params),
                self.trainer.best_loss,
            )
            self.prev_theta, self.prev_loss = self.best_params.clone(), self.best_loss
            self.final_params, self.final_loss = (
                self._reparameterize_canon_form(self.trainer.final_params),
                self.trainer.final_loss,
            )
            self.ema_params = self._reparameterize_canon_form(self.trainer.ema_params)
            self.avg_params = self._reparameterize_canon_form(self.trainer.avg_params)

            should_stop, reason = self._check_convergence()

            if should_stop:
                self.logger.info(f"\n{'=' * 60}")
                self.logger.info(f"procedure converged: {reason}")
                self.logger.info(
                    f"final radius: {self.radius:.4f}, final loss: {current_loss:.6f}"
                )
                self.logger.info(f"total phases: {phase}")
                self.logger.info(f"\n{'=' * 60}")
                break

            # Expand radius for next phase
            self.prev_best_loss = current_loss
            old_radius = self.radius
            self.radius = min(self.radius * self.args.rate, self.args.max_radius)

            self.logger.info(
                f"expanding radius: {old_radius:.4f} -> {self.radius:.4f}, "
                f"loss improved by: {self.prev_best_loss - current_loss:.6e}"
            )

        return self

    def forward(self, x):
        """Return concatenated inverse covariance and natural mean parameters."""
        return ch.cat([self.covariance_matrix.inverse().flatten(), self.theta])


class UnknownTruncationMultivariateNormalUnknownCovariance(  # pylint: disable=too-many-instance-attributes
    TruncatedMultivariateNormalUnknownCovariance
):
    """
    Truncated multivariate normal distribution class with unknown covariance
    and unknown truncation set.
    """

    def __init__(self, args: Parameters, k: int, alpha: float, dims: int):
        """
        Initialize UnknownTruncationMultivariateNormalUnknownCovariance.

        Args:
            args (Parameters): hyperparameter object
            k (int): number of nearest neighbors for oracle
            alpha (float): survival probability lower bound
            dims (int): number of dimensions
        """
        assert isinstance(args, Parameters), (
            f"args is type {type(args)}. expecting type delphi.utils.helper.Parameters."
        )
        # algorithm hyperparameters
        self.k = k
        self.args = check_and_fill_args(args, UNKNOWN_TRUNC_MULTI_NORM_DEFAULTS)
        super().__init__(args, partial(UnknownGaussian, k), alpha, dims)

        self.emp_loc, self.emp_covariance_matrix = None, None
        self.criterion = UnknownTruncationMultivariateNormalNLL.apply

        # Attributes initialized during fit
        self.train_loader_ = None
        self.val_loader_ = None
        self.exp_h = None
        self.prev_best_loss = None
        self.radius_history = None
        self.loss_history = None
        self.radius = None
        self.trainer = None
        self.prev_theta = None
        self.prev_loss = None
        self.emp_T = None  # pylint: disable=invalid-name
        self.emp_v = None

    def fit(self, S: Tensor):
        """
        Fit the model to the observed (truncated) samples S.

        Args:
            S (Tensor): observed samples of shape (num_samples, dims)
        """
        assert isinstance(S, Tensor), (
            f"S is type: {type(S)}. expected type torch.Tensor."
        )
        assert S.size(0) > S.size(1), (
            "input expected to be shape num samples by dimenions, "
            f"current input is size {S.size()}."
        )
        self.train_loader_, self.val_loader_ = make_train_and_val_distr(
            self.args, S, UnknownTruncationNormalDataset
        )

        # verify that the S is whitened to N(0, I)
        emp_loc = S.mean(0)
        emp_cov = cov(S)
        if (
            ch.norm(emp_loc - ch.zeros(S.size(1))) >= 1e-3
            or ch.norm(emp_cov - ch.eye(S.size(1))) >= 1e-3
        ):
            raise ValueError(
                f"input dataset must be whitened (eg. N(O, I)). \n "
                f"dataset mean: {emp_loc}, covariance matrix: {emp_cov}"
            )
        self.phi = self.phi(S)
        self.exp_h = ExpH(emp_loc, emp_cov)
        self.criterion_params = [self.phi, self.exp_h, self.dims]

        # Initialize tracking
        self.prev_best_loss = None
        self.radius_history = []
        self.loss_history = []
        # Initialize radius and parameters
        self._calc_emp_model()
        self.radius = self.args.min_radius

        phase = 0
        while phase < self.args.max_phases:
            phase += 1
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"phase {phase}: training with radius={self.radius:.4f}")
            self.logger.info(f"\n{'=' * 60}")

            self.trainer = Trainer(self, self.args, self.logger)
            self.trainer.train_model(self.train_loader_, self.val_loader_)

            # Update tracking
            current_loss = self.trainer.best_loss
            self.radius_history.append(self.radius)
            self.loss_history.append(current_loss)

            self.best_params, self.best_loss = (
                self._reparameterize_canon_form(self.trainer.best_params),
                self.trainer.best_loss,
            )
            self.prev_theta, self.prev_loss = self.best_params.clone(), self.best_loss
            self.final_params, self.final_loss = (
                self._reparameterize_canon_form(self.trainer.final_params),
                self.trainer.final_loss,
            )
            self.ema_params = self._reparameterize_canon_form(self.trainer.ema_params)
            self.avg_params = self._reparameterize_canon_form(self.trainer.avg_params)

            should_stop, reason = self._check_convergence()

            if should_stop:
                self.logger.info(f"\n{'=' * 60}")
                self.logger.info(f"procedure converged: {reason}")
                self.logger.info(
                    f"final radius: {self.radius:.4f}, final loss: {current_loss:.6f}"
                )
                self.logger.info(f"total phases: {phase}")
                self.logger.info(f"\n{'=' * 60}")
                break

            # Expand radius for next phase
            self.prev_best_loss = current_loss
            old_radius = self.radius
            self.radius = min(self.radius * self.args.rate, self.args.max_radius)

            self.logger.info(
                f"expanding radius: {old_radius:.4f} -> {self.radius:.4f}, "
                f"loss improved by: {self.prev_best_loss - current_loss:.6e}"
            )

        return self

    def _calc_emp_model(self):
        """Calculate empirical natural parameters and register T and v."""
        dataset_s = self.train_loader_.dataset.S  # pylint: disable=invalid-name

        self.emp_T = cov(dataset_s).inverse()  # pylint: disable=invalid-name
        self.emp_v = self.emp_T @ dataset_s.mean(0)
        self.register_parameter("T", nn.Parameter(self.emp_T.clone()))
        self.register_parameter("v", nn.Parameter(self.emp_v.clone()))

    def forward(self, x):
        """Return concatenated precision matrix and natural mean parameters."""
        return ch.cat([self.T.flatten(), self.v])

    def project_to_pos_definite(self, M, eps=1e-6):  # pylint: disable=invalid-name
        """Projects a symmetric matrix M onto the Positive Semi-Definite (PSD) cone."""
        L, Q = ch.linalg.eigh(M)  # pylint: disable=invalid-name,not-callable
        L_clipped = ch.clamp(L, min=eps)  # pylint: disable=invalid-name
        return Q @ ch.diag_embed(L_clipped) @ Q.T  # pylint: disable=invalid-name

    def step_post_hook(self, optimizer, args, kwargs):
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
                v = self.emp_v + loc_diff / dist * self.radius

            # --- 2) Frobenius ball projection for T ---
            cov_diff = mat_t - self.emp_T
            frob_norm = ch.linalg.norm(cov_diff, ord="fro")  # pylint: disable=not-callable

            if frob_norm > self.radius:
                mat_t = self.emp_T + cov_diff * (self.radius / frob_norm)

            # Symmetrize after projection (important!)
            mat_t = 0.5 * (mat_t + mat_t.T)
            # --- 3) Final PSD projection ---
            mat_t = self.project_to_pos_definite(mat_t, eps=self.eigenvalue_lower_bound)
            # Symmetrize again after PSD projection
            mat_t = 0.5 * (mat_t + mat_t.T)

            self.T.copy_(mat_t)
            self.v.copy_(v)

    def _reparameterize_canon_form(self, theta):
        """Convert natural parameters to canonical (mean/covariance) form."""
        mat_t = theta[: self.dims**2].resize(self.dims, self.dims)  # pylint: disable=invalid-name
        v = theta[self.dims**2 :]

        covariance_matrix = mat_t.inverse()
        loc = v @ covariance_matrix

        return ch.cat([covariance_matrix.flatten(), loc.flatten()])


def UnknownTruncationMultivariateNormal(  # pylint: disable=invalid-name
    args: Parameters,
    k: int,
    alpha: float,
    dims: int,
    covariance_matrix: Optional[ch.Tensor] = None,
):
    """
    Factory function for unknown-truncation multivariate normal distributions.

    Returns a known-covariance model if covariance_matrix is provided,
    otherwise returns an unknown-covariance model.
    """
    assert isinstance(args, Parameters), (
        "args is type: {}. expecting args to be type delphi.utils.helpers.Parameters"
    )
    args = check_and_fill_args(args, UNKNOWN_TRUNC_MULTI_NORM_DEFAULTS)

    if covariance_matrix is not None:
        return UnknownTruncationMultivariateNormalKnownCovariance(
            args, k, alpha, dims, covariance_matrix=covariance_matrix
        )
    return UnknownTruncationMultivariateNormalUnknownCovariance(args, k, alpha, dims)


# HELPER FUNCTIONS
class ExpH:  # pylint: disable=too-few-public-methods
    """Helper class computing the exponential h function for unknown truncation."""

    def __init__(self, emp_loc, emp_cov):
        """
        Initialize ExpH with empirical location and covariance.

        Args:
            emp_loc: empirical mean vector
            emp_cov: empirical covariance matrix
        """
        self.emp_loc = emp_loc
        self.emp_cov = emp_cov
        self.pi_const = (self.emp_loc.size(0) / 2.0) * ch.log(
            ch.Tensor([2.0 * ch.pi])
        ).unsqueeze(0)

    def __call__(self, u, B, x):  # pylint: disable=invalid-name
        """Return the evaluated exponential h function."""
        quad_term = 0.5 * ch.sum(x @ B * x, dim=1, keepdim=True)
        trace_term = (
            ch.trace(
                (B - ch.eye(u.size(0)))
                @ (self.emp_cov + self.emp_loc[..., None] @ self.emp_loc[None, ...])
            ).unsqueeze(0)
            / 2.0
        )
        lin_term = ((x - self.emp_loc) @ u)[..., None]
        h = quad_term - trace_term - lin_term + self.pi_const  # pylint: disable=invalid-name
        return ch.exp(h).double()


# Keep old name as alias for backwards compatibility
Exp_h = ExpH  # pylint: disable=invalid-name
