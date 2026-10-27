# Author: pstefanou12@
"""Truncated linear regression: known variance, unknown variance, dependent data.

- ``noise_var`` given: Daskalakis, Gouleakis, Tzamos and Zampetakis (2020),
  projected SGD without replacement on the truncated NLL in D_{r,B}, then
  Newton steps to the empirical MLE.
- ``noise_var=None``: the variance is learned with the weight, in the natural
  parameters ``(v, lambda) = (w / sigma^2, 1 / sigma^2)``.
- ``dependent=True``: a linear dynamical system, by the switching gradient
  with ``Sigma^-1`` preconditioning.
"""

from collections.abc import Callable
import math
from typing import Optional, Union, cast

import torch as ch
import torch.nn

from delphi import configs
import delphi.projection_sets
from delphi.stats import linear_model
from delphi.stats.truncated import losses
from delphi.stats.truncated.truncated_linear_model import TruncatedLinearModel
from delphi.truncated import sampling
from delphi.utils import helpers


class TruncatedLinearRegression(  # pylint: disable=too-many-instance-attributes
    TruncatedLinearModel[configs.TruncatedRegressionConfig]
):
    """Truncated linear regression ``y = w^T x + b + e``, ``e ~ N(0, sigma^2)``.

    Observed only where ``phi(y)`` admits it. The variant follows the
    arguments, as an OLS package's would: a known ``noise_var``, an unknown
    one, or dependent data.

    Attributes:
        noise_var: The known noise variance, or None when it is learned.
        dependent: Whether the samples come from one trajectory.
        variance_bounds: The interval the learned variance is kept in.
        survival_threshold: The threshold s of dependent data; 1 otherwise.
    """

    # Created by register_parameter and register_buffer during fit.
    v: torch.nn.Parameter
    lambda_: torch.nn.Parameter
    Sigma: ch.Tensor  # pylint: disable=invalid-name

    defaults = configs.TruncatedRegressionConfig

    def __init__(  # pylint: disable=too-many-arguments
        self,
        hyperparameters: dict,
        phi: Callable,
        alpha: Optional[float] = None,
        *,
        noise_var: Optional[Union[float, ch.Tensor]] = None,
        dependent: bool = False,
        sampler: Optional[sampling.GaussianSampler] = None,
        fit_intercept: bool = True,
        emp_weight: Optional[ch.Tensor] = None,
        rand_seed: int = 0,
    ):
        """Initialize TruncatedLinearRegression.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation oracle over responses shaped (n, m, k).
            alpha: Survival probability in (0, 1]; None if unknown, which
                only known variance supports.
            noise_var: The known noise variance; None learns it.
            dependent: Whether the samples form one trajectory, x_{t+1} =
                A x_t + e; needs noise_var, its noise covariance.
            sampler: Estimates the known-variance loss's conditional
                moments; rejection sampling at ``num_samples`` and
                ``min_accepted`` by default.
            fit_intercept: Whether to fit an intercept.
            emp_weight: Start in place of OLS; see ``TruncatedLinearModel``.
            rand_seed: Seed applied at the start of every run.

        Raises:
            ValueError: If dependent data lacks noise_var, or alpha is
                missing where the variant needs it.
        """
        if dependent and noise_var is None:
            raise ValueError("dependent data needs the noise covariance noise_var.")
        if alpha is None and (dependent or noise_var is None):
            raise ValueError("unknown variance and dependent data need alpha.")
        if dependent:
            self.defaults = configs.TruncatedLinearDynamicalSystemConfig
        elif noise_var is None:
            self.defaults = configs.TruncatedUnknownVarianceRegressionConfig
        super().__init__(
            hyperparameters, phi, alpha, fit_intercept, emp_weight, rand_seed
        )
        self.noise_var = None if noise_var is None else ch.as_tensor(noise_var)
        self.dependent = dependent
        self.variance_bounds: Optional[helpers.Bounds] = None
        self.survival_threshold = 1.0
        self.sampler = sampler or sampling.GaussianRejectionSampler(
            self.hyperparameters.num_samples, self.hyperparameters.min_accepted
        )
        if dependent:
            lds = cast(
                configs.TruncatedLinearDynamicalSystemConfig, self.hyperparameters
            )
            known = cast(float, alpha)
            self.hyperparameters.lr = (2 / known) ** lds.c_eta
            self.survival_threshold = lds.c_s * (math.sqrt(math.log(1 / known)) + 1)
            self.criterion = losses.SwitchGrad.apply
        elif noise_var is None:
            self.criterion = losses.TruncatedGaussianNLLUnknownVariance.apply
        else:
            self.criterion = losses.TruncatedGaussianNLL.apply

    def criterion_parameters(self, X: ch.Tensor, y: ch.Tensor) -> list:  # pylint: disable=invalid-name
        hp = self.hyperparameters
        if self.dependent:
            lds = cast(configs.TruncatedLinearDynamicalSystemConfig, hp)
            return [
                self.phi,
                lds.c_gamma,
                self.alpha,
                X.size(0),
                self.noise_var,
                hp.num_samples,
                hp.eps,
            ]
        if self.noise_var is None:
            return [self.lambda_, self.phi, hp.num_samples, hp.eps]
        return [self.phi, self.noise_var, self.sampler, hp.eps]

    def register_weight(self, weight: ch.Tensor) -> None:
        X, y = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
        if self.noise_var is None:
            self._register_natural_parameters(X, y, weight)
            return
        super().register_weight(weight)
        if self.dependent:
            sigma_0 = X.T @ X / (self.survival_threshold * X.size(0))
            self.register_buffer("Sigma", sigma_0)
            # A tensor hook runs after backward, inside optimizer.step's
            # closure; a step pre-hook runs before the closure's zero_grad
            # and backward, which discard what it writes.
            self.weight.register_hook(self.precondition)

    def _register_natural_parameters(self, X, y, weight) -> None:  # pylint: disable=invalid-name
        """Register (v, lambda) from a start weight, with the variance bounds.

        The bounds are Daskalakis, Stefanou, Yao and Zampetakis (2022),
        Definition 3.1, around the residual variance at the start.
        """
        hp = cast(
            configs.TruncatedUnknownVarianceRegressionConfig, self.hyperparameters
        )
        residual_var = ch.var(X @ weight - y, dim=0)[..., None]
        lower = residual_var / (8 * (5 + 2 * math.log(1 / cast(float, self.alpha))))
        upper = residual_var + hp.r * self.base_radius
        self.variance_bounds = helpers.Bounds(lower, upper)
        lambda_ = residual_var.inverse()
        self.register_parameter("v", torch.nn.Parameter(weight * lambda_))
        self.register_parameter("lambda_", torch.nn.Parameter(lambda_.clone()))

    def precondition(self, grad: ch.Tensor) -> ch.Tensor:
        """Return Sigma^-1 grad, the SON-SG step direction for dependent data."""
        return ch.linalg.solve(self.Sigma, grad)  # pylint: disable=not-callable,no-member

    def compute_loss(self, batch, criterion=None) -> ch.Tensor:
        """Average the known-variance NLL, which is per sample, over the batch."""
        loss = super().compute_loss(batch, criterion)
        if self.noise_var is None or self.dependent:
            return loss
        return loss.mean()

    def forward(self, X: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        if self.noise_var is None:
            return X @ self.v / self.lambda_
        if self.dependent:
            self.Sigma += X.T @ X / X.size(0)  # pylint: disable=invalid-name,no-member
        return X @ self.weight

    def step_post_hook(self, optimizer, args, kwargs) -> None:
        """Keep the learned variance in its bounds, or project the weight."""
        if self.noise_var is None:
            bounds = linear_model.require_fitted(self.variance_bounds, "bounds")
            with ch.no_grad():
                variance = ch.clamp(1.0 / self.lambda_, bounds.lower, bounds.upper)
                self.lambda_.copy_(1.0 / variance)
            return
        super().step_post_hook(optimizer, args, kwargs)

    def parameter_groups(self) -> list[dict]:
        """Train lambda at var_lr when it is set."""
        var_lr = getattr(self.hyperparameters, "var_lr", None)
        if self.noise_var is None and var_lr is not None:
            return [{"params": [self.v]}, {"params": [self.lambda_], "lr": var_lr}]
        return super().parameter_groups()

    def build_projection_set(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        y: ch.Tensor,
        radius: float,
    ) -> delphi.projection_sets.ProjectionSet:
        """Return D_{r,B} anchored at emp_weight, its r grown until it holds it.

        Raises:
            ValueError: If emp_weight lies outside the ball of radius
                weight_bound.
        """
        if self.dependent:
            return super().build_projection_set(X, y, radius)
        hp = self.hyperparameters
        anchor = self.emp_start
        if hp.weight_bound is not None and float(anchor.norm()) > hp.weight_bound:
            raise ValueError(
                f"emp_weight has norm {float(anchor.norm()):.4f} in the scaled "
                f"coordinates, outside weight_bound {hp.weight_bound}."
            )
        variance = float(linear_model.require_fitted(self.noise_var, "noise_var"))
        projection = delphi.projection_sets.ResidualSpectralSet(
            X,
            y,
            variance,
            radius,
            anchor,
            hp.weight_bound,
            bisection_steps=hp.bisection_steps,
            max_quadratic_entries=hp.max_quadratic_entries,
        )
        while not projection.contains(anchor):
            projection.radius = cast(float, projection.radius) * hp.rate
            self.logger.warning(
                f"emp_weight lies outside D_{{r,B}}; r -> {projection.radius:.4f}."
            )
        return projection

    def hessian(self, theta: Optional[ch.Tensor] = None) -> ch.Tensor:
        """Return the Hessian of the known-variance NLL, by default at OLS."""
        if self.noise_var is None or self.dependent:
            return super().hessian(theta)
        X, _ = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
        weight = self.weight.detach() if theta is None else theta
        return losses.TruncatedGaussianNLL.hessian(
            X @ weight.reshape(-1, linear_model.require_fitted(self.k, "k")),
            X,
            self.phi,
            self.noise_var,
            self.sampler,
        )

    def coefficients(self, params):
        """Split a flat iterate; for learned variance, ``(v, lambda)``."""
        if params is not None and self.noise_var is None:
            params = params[:-1] / params[-1]
        return super().coefficients(params)

    def _variance(self, name: str) -> Optional[ch.Tensor]:
        """Return the noise variance at one of the estimates."""
        if self.noise_var is not None:
            return self.noise_var
        params = self._estimate(name)
        return None if params is None else 1.0 / params[-1:]

    @property
    def avg_variance_(self) -> Optional[ch.Tensor]:
        """Noise variance at the averaged iterate."""
        return self._variance("avg")

    @property
    def ema_variance_(self) -> Optional[ch.Tensor]:
        """Noise variance at the moving average of the iterates."""
        return self._variance("ema")

    @property
    def final_variance_(self) -> Optional[ch.Tensor]:
        """Noise variance at the last iterate."""
        return self._variance("final")

    @property
    def best_variance_(self) -> Optional[ch.Tensor]:
        """Noise variance at the iterate with the lowest validation loss."""
        return self._variance("best")

    @property
    def variance_(self) -> Optional[ch.Tensor]:
        """Noise variance at the averaged iterate."""
        return self.avg_variance_

    def __str__(self):
        """Return a human-readable name for this model."""
        if self.dependent:
            return "truncated linear dynamical system"
        if self.noise_var is None:
            return "truncated linear regression, unknown variance"
        return "truncated linear regression, known variance"
