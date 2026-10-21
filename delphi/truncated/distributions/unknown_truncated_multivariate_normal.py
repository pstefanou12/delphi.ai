# Author: pstefanou12@
"""Truncated multivariate normal distribution without oracle access (unknown truncation set)."""

# pylint: disable=duplicate-code

from functools import partial
from typing import Optional

import torch as ch
from torch import nn

from delphi import oracle, trainer
from delphi.truncated.distributions import (
    losses,
    truncated_multivariate_normal,
    truncated_multivariate_normal_known_covariance,
)
from delphi.utils import datasets, helpers


class UnknownTruncationMultivariateNormalKnownCovariance(  # pylint: disable=too-many-instance-attributes
    truncated_multivariate_normal_known_covariance.TruncatedMultivariateNormalKnownCovariance
):
    """Truncated multivariate normal with known covariance and unknown truncation.

    Attributes:
        k (int): Number of nearest neighbors used by the oracle.
        exp_h: Exponential h helper object, set during fit.
    """

    def __init__(
        self,
        args: helpers.Parameters,
        k: int,
        alpha: float,
        dims: int,
        covariance_matrix: ch.Tensor,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize UnknownTruncationMultivariateNormalKnownCovariance.

        Args:
            args: Hyperparameter object.
            k: Number of nearest neighbors for oracle.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
            covariance_matrix: Known covariance matrix.

        Raises:
            TypeError: If args is not a Parameters instance.
        """
        self.k = k
        # TODO: pstefanou12@ - args is a helpers.Parameters, but __init__ has
        # been dict-only since e3c37f7; port this call with delphi/stats/.
        super().__init__(
            args,  # type: ignore[arg-type]
            partial(oracle.UnknownGaussian, k),
            dims,
            covariance_matrix,
            alpha,
        )

        self.emp_loc, self.emp_covariance_matrix = None, None
        self.criterion = losses.UnknownTruncationMultivariateNormalNLL.apply

        # Attributes initialized during fit.
        self.train_loader_ = None
        self.val_loader_ = None
        self.exp_h: Optional["ExpH"] = None
        self.radius_history = None
        self.loss_history = None
        self.radius = None
        self.trainer = None

    def fit(self, S: ch.Tensor):
        """Fit the model to the observed (truncated) samples S.

        Args:
            S: Observed samples of shape (num_samples, dims).
        """
        assert isinstance(S, ch.Tensor), (
            f"S is type: {type(S)}. expected type torch.Tensor."
        )
        assert S.size(0) > S.size(1), (
            "input expected to be shape num samples by dimenions, "
            f"current input is size {S.size()}."
        )
        self.train_loader_, self.val_loader_ = datasets.make_train_and_val_distr(
            self.hyperparameters,
            S,
            datasets.UnknownTruncationNormalDataset,
        )

        # Verify that S is whitened to N(0, I).
        emp_loc = S.mean(0)
        emp_cov = helpers.cov(S)
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

        self._calc_emp_model()
        self.trainer = trainer.Trainer(self, self.hyperparameters)
        self.trainer.fit(self.train_loader_, self.val_loader_)
        self._adopt_results(self.trainer)
        return self

    def pretrain_procedure_hook(self) -> None:
        """Start the projection ball at min_radius."""
        self.radius = self.hyperparameters.min_radius
        self.radius_history = [self.radius]

    def _adopt_results(self, procedure: trainer.Trainer) -> None:
        """Take the last phase's estimates in canonical form."""
        super()._adopt_results(procedure)
        # TODO: pstefanou12@ - _reparameterize_canon_form is defined only on
        # UnknownTruncationMultivariateNormalUnknownCovariance, so these calls
        # raise AttributeError at runtime. Its body also assumes theta packs
        # [T.flatten(), v], whereas this class's theta is just v, so the
        # known-covariance conversion is probably loc = covariance_matrix @ theta.
        # Needs a decision before it can be written; see the summary.
        for name in ("best_params", "final_params", "ema_params", "avg_params"):
            setattr(self, name, self._reparameterize_canon_form(getattr(self, name)))  # type: ignore[operator]

    def forward(self, x):
        """Return concatenated inverse covariance and natural mean parameters."""
        return ch.cat([self.covariance_matrix.inverse().flatten(), self.theta])


class UnknownTruncationMultivariateNormalUnknownCovariance(  # pylint: disable=too-many-instance-attributes
    truncated_multivariate_normal.TruncatedMultivariateNormal
):
    """Truncated multivariate normal with unknown covariance and unknown truncation.

    Attributes:
        k (int): Number of nearest neighbors used by the oracle.
        exp_h: Exponential h helper object, set during fit.
    """

    def __init__(
        self,
        args: helpers.Parameters,
        k: int,
        alpha: float,
        dims: int,
    ):
        """Initialize UnknownTruncationMultivariateNormalUnknownCovariance.

        Args:
            args: Hyperparameter object.
            k: Number of nearest neighbors for oracle.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.

        Raises:
            TypeError: If args is not a Parameters instance.
        """
        self.k = k
        # TODO: pstefanou12@ - args is a helpers.Parameters, but __init__ has
        # been dict-only since e3c37f7; port this call with delphi/stats/.
        super().__init__(
            args,  # type: ignore[arg-type]
            partial(oracle.UnknownGaussian, k),
            dims,
            alpha,
        )

        self.emp_loc, self.emp_covariance_matrix = None, None
        self.criterion = losses.UnknownTruncationMultivariateNormalNLL.apply

        # Attributes initialized during fit.
        self.train_loader_ = None
        self.val_loader_ = None
        self.exp_h: Optional["ExpH"] = None
        self.radius_history = None
        self.loss_history = None
        self.radius = None
        self.trainer = None
        self.emp_T: Optional[ch.Tensor] = None  # pylint: disable=invalid-name
        self.emp_v: Optional[ch.Tensor] = None

    def fit(self, S: ch.Tensor):
        """Fit the model to the observed (truncated) samples S.

        Args:
            S: Observed samples of shape (num_samples, dims).
        """
        assert isinstance(S, ch.Tensor), (
            f"S is type: {type(S)}. expected type torch.Tensor."
        )
        assert S.size(0) > S.size(1), (
            "input expected to be shape num samples by dimenions, "
            f"current input is size {S.size()}."
        )
        self.train_loader_, self.val_loader_ = datasets.make_train_and_val_distr(
            self.hyperparameters,
            S,
            datasets.UnknownTruncationNormalDataset,
        )

        # Verify that S is whitened to N(0, I).
        emp_loc = S.mean(0)
        emp_cov = helpers.cov(S)
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

        self._calc_emp_model()
        self.trainer = trainer.Trainer(self, self.hyperparameters)
        self.trainer.fit(self.train_loader_, self.val_loader_)
        self._adopt_results(self.trainer)
        return self

    def pretrain_procedure_hook(self) -> None:
        """Start the projection ball at min_radius."""
        self.radius = self.hyperparameters.min_radius
        self.radius_history = [self.radius]

    def _adopt_results(self, procedure: trainer.Trainer) -> None:
        """Take the last phase's estimates in canonical form."""
        super()._adopt_results(procedure)
        for name in ("best_params", "final_params", "ema_params", "avg_params"):
            setattr(self, name, self._reparameterize_canon_form(getattr(self, name)))

    def _calc_emp_model(self):
        """Calculate empirical natural parameters and register T and v."""
        dataset_s = self.train_loader_.dataset.S  # pylint: disable=invalid-name

        self.emp_T = (  # pylint: disable=invalid-name
            helpers.cov(dataset_s).inverse()
        )
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
        """Project parameters back into the feasible set after each update."""
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
    args: helpers.Parameters,
    k: int,
    alpha: float,
    dims: int,
    covariance_matrix: Optional[ch.Tensor] = None,
):
    """Factory function for unknown-truncation multivariate normal distributions.

    Returns a known-covariance model if covariance_matrix is provided,
    otherwise returns an unknown-covariance model.

    Args:
        args: Hyperparameter object.
        k: Number of nearest neighbors for the oracle.
        alpha: Survival probability lower bound.
        dims: Number of dimensions.
        covariance_matrix: Known covariance; if None, it is estimated.

    Returns:
        UnknownTruncationMultivariateNormalKnownCovariance if covariance_matrix is
        provided, else UnknownTruncationMultivariateNormalUnknownCovariance.

    Raises:
        TypeError: If args is not a Parameters instance.
    """
    if covariance_matrix is not None:
        return UnknownTruncationMultivariateNormalKnownCovariance(
            args, k, alpha, dims, covariance_matrix=covariance_matrix
        )
    return UnknownTruncationMultivariateNormalUnknownCovariance(args, k, alpha, dims)


class ExpH:  # pylint: disable=too-few-public-methods
    """Helper class computing the exponential h function for unknown truncation.

    Attributes:
        emp_loc: Empirical mean vector.
        emp_cov: Empirical covariance matrix.
        pi_const: Pre-computed log-normalisation constant.
    """

    def __init__(self, emp_loc, emp_cov):
        """Initialize ExpH with empirical location and covariance.

        Args:
            emp_loc: Empirical mean vector.
            emp_cov: Empirical covariance matrix.
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


# Keep old name as alias for backwards compatibility.
Exp_h = ExpH  # pylint: disable=invalid-name
