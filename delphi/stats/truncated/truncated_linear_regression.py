# Author: pstefanou12@
"""Truncated linear regression: known variance, unknown variance, dependent data.

- ``noise_var`` given: Daskalakis, Gouleakis, Tzamos and Zampetakis (2020),
  projected SGD without replacement on the truncated NLL in D_{r,B}, then
  Newton steps to the empirical MLE.
- ``noise_var=None``: Daskalakis, Stefanou, Yao and Zampetakis (2022),
  projected SGD on the truncated NLL in the natural parameters
  ``(v, lambda) = (w / sigma^2, 1 / sigma^2)``, kept in their set D, then
  Newton steps to the empirical MLE.
- ``dependent=True``: a linear dynamical system, by the switching gradient
  with ``Sigma^-1`` preconditioning.
"""

from collections.abc import Callable
import math
from typing import Optional, Union, cast

import torch as ch

from delphi import configs
import delphi.projection_sets
from delphi.stats import linear_model
from delphi.stats.truncated import losses
from delphi.stats.truncated.truncated_linear_model import TruncatedLinearModel
from delphi.truncated import sampling
from delphi.truncated.distributions import summary


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
        survival_threshold: The threshold s of dependent data; 1 otherwise.
    """

    # Created by register_buffer during fit.
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
                dependent data does not support.
            noise_var: The known noise variance; None learns it, training
                ``weight`` as ``(v, lambda)``: v's rows, then lambda's.
            dependent: Whether the samples form one trajectory, x_{t+1} =
                A x_t + e; needs noise_var, its noise covariance.
            sampler: Estimates the loss's conditional moments; rejection
                sampling at ``num_samples`` and ``min_accepted`` by default.
            fit_intercept: Whether to fit an intercept.
            emp_weight: Start in place of OLS; see ``TruncatedLinearModel``.
            rand_seed: Seed applied at the start of every run.

        Raises:
            ValueError: If dependent data lacks noise_var or alpha.
        """
        if dependent and (noise_var is None or alpha is None):
            raise ValueError("dependent data needs noise_var and alpha.")
        if dependent:
            self.defaults = configs.TruncatedLinearDynamicalSystemConfig
        elif noise_var is None:
            self.defaults = configs.TruncatedUnknownVarianceRegressionConfig
        super().__init__(
            hyperparameters, phi, alpha, fit_intercept, emp_weight, rand_seed
        )
        self.noise_var = None if noise_var is None else ch.as_tensor(noise_var)
        self.dependent = dependent
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
            return [self.phi, self.sampler, hp.eps]
        return [self.phi, self.noise_var, self.sampler, hp.eps]

    def register_weight(self, weight: ch.Tensor) -> None:
        if self.noise_var is None:
            precision = 1.0 / self._residual_var(weight)
            super().register_weight(ch.cat([weight * precision, precision[None]]))
            return
        super().register_weight(weight)
        if self.dependent:
            X, _ = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
            sigma_0 = X.T @ X / (self.survival_threshold * X.size(0))
            self.register_buffer("Sigma", sigma_0)
            # A tensor hook runs after backward, inside optimizer.step's
            # closure; a step pre-hook runs before the closure's zero_grad
            # and backward, which discard what it writes.
            self.weight.register_hook(self.precondition)

    def _residual_var(self, weight: ch.Tensor) -> ch.Tensor:
        """Return each output's mean squared residual at weight, shaped (k,)."""
        X, y = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
        return (X @ weight - y).pow(2).mean(0)

    def precondition(self, grad: ch.Tensor) -> ch.Tensor:
        """Return Sigma^-1 grad, the SON-SG step direction for dependent data."""
        return ch.linalg.solve(self.Sigma, grad)  # pylint: disable=not-callable,no-member

    def compute_loss(self, batch, criterion=None) -> ch.Tensor:
        """Average the NLL, which is per sample, over the batch."""
        loss = super().compute_loss(batch, criterion)
        return loss if self.dependent else loss.mean()

    def forward(self, X: ch.Tensor):  # pylint: disable=invalid-name
        """Return ``h_W(x)``, or ``(v^T x, lambda)`` when lambda is learned."""
        if self.noise_var is None:
            return X @ self.weight[:-1], self.weight[-1]
        if self.dependent:
            self.Sigma += X.T @ X / X.size(0)  # pylint: disable=invalid-name,no-member
        return X @ self.weight

    def default_radius(self) -> float:
        """Return 1 / alpha, the radius of D, for learned variance."""
        if self.noise_var is not None:
            return super().default_radius()
        return 1.0 if self.alpha is None else 1.0 / self.alpha

    def build_projection_set(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        y: ch.Tensor,
        radius: float,
    ) -> delphi.projection_sets.ProjectionSet:
        """Return the set the descent is kept in, anchored at emp_weight.

        D_{r,B} with its r grown until it holds emp_weight, or for learned
        variance D around emp_weight's residual variance.

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
        if self.noise_var is None:
            residual_var = self._residual_var(anchor)
            return delphi.projection_sets.NaturalParameterSet(
                residual_var,
                radius,
                ch.cat([anchor / residual_var, 1.0 / residual_var[None]]),
                hp.weight_bound,
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
        """Return the Hessian of the NLL, by default at the current weight."""
        if self.dependent:
            return super().hessian(theta)
        X, _ = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
        weight = self.weight.detach() if theta is None else theta
        weight = weight.reshape(-1, linear_model.require_fitted(self.k, "k"))
        if self.noise_var is None:
            return losses.TruncatedGaussianNLLUnknownVariance.hessian(
                X @ weight[:-1], weight[-1], X, self.phi, self.sampler
            )
        return losses.TruncatedGaussianNLL.hessian(
            X @ weight,
            X,
            self.phi,
            self.noise_var,
            self.sampler,
        )

    def coefficients(self, params):
        """Split a flat iterate; for learned variance, ``(v, lambda)``."""
        if params is not None and self.noise_var is None:
            natural = params.reshape(-1, linear_model.require_fitted(self.k, "k"))
            params = (natural[:-1] / natural[-1]).flatten()
        return super().coefficients(params)

    def _variance(self, name: str) -> Optional[ch.Tensor]:
        """Return the noise variance at one of the estimates."""
        if self.noise_var is not None:
            return self.noise_var
        params = self._estimate(name)
        if params is None:
            return None
        return 1.0 / params.reshape(-1, linear_model.require_fitted(self.k, "k"))[-1]

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

    def _summary_estimates(self) -> list[tuple[str, str]]:
        rows = super()._summary_estimates()
        if self.noise_var is None:
            rows += [
                (f"{name}_variance_", summary.tensor(self._variance(name)))
                for name in linear_model.ESTIMATORS
            ]
        return rows

    def __str__(self):
        """Return a human-readable name for this model."""
        if self.dependent:
            return "truncated linear dynamical system"
        if self.noise_var is None:
            return "truncated linear regression, unknown variance"
        return "truncated linear regression, known variance"
