# Author: pstefanou12@
"""Weighted quantile regression, ``min_c sum_i w_i rho_{tau_i}(y_i - a_i^T c)``.

``rho_tau(r) = max(tau r, (tau - 1) r)`` is the check loss, and each row has
its own level ``tau_i``. At the default ``tau = 1/2`` it is half of ``|r|``,
so the problem is least absolute deviations; at ``tau = 1`` it is the hinge
``max(r, 0)``.

Solved as a linear program by the Frisch-Newton interior point method
(Portnoy and Koenker, 1997), with Mehrotra's predictor-corrector. Weights
fold into the rows, as ``w rho_tau(r) = rho_tau(w r)`` for ``w >= 0``. The
primal is ``min sum_i tau_i u_i + (1 - tau_i) v_i`` over ``u, v >= 0`` with
``X c + u - v = y``. Its dual, shifted by ``1 - tau`` into the unit box, is

    max_a  y^T a   s.t.  X^T a = sum_i (1 - tau_i) x_i,   0 <= a <= 1,

whose equality multiplier is c; at ``tau = 1/2`` the right-hand side is
``X^T 1 / 2``. Every Newton step solves one p x p system ``X^T D X``, so the
cost is ``O(n p^2)`` per step over a few tens of steps, and the duality gap
certifies the solution. The trainer runs the steps: each is one of
``L1Regression``'s own updates, with no optimizer.
"""

from typing import Optional

import torch as ch

from delphi.configs.learning import l1_regression as l1_regression_configs
from delphi.delphi import DelphiModel
import delphi.trainer

# The fraction of the distance to the boundary a step may cover.
STEP_FRACTION = 0.99995


def _step_length(value: ch.Tensor, direction: ch.Tensor) -> float:
    """Return the largest step in (0, 1] keeping ``value + t direction > 0``."""
    shrinking = direction < 0
    if not bool(shrinking.any()):
        return 1.0
    limit = float((-value[shrinking] / direction[shrinking]).min())
    return min(1.0, STEP_FRACTION * limit)


class L1Regression(  # pylint: disable=too-many-instance-attributes
    DelphiModel[l1_regression_configs.L1RegressionConfig]
):
    """Weighted least absolute deviations, by an interior point method.

    Given per-row quantile levels, it fits weighted quantile regression.

    Usage::

        regression = L1Regression().fit(X, y, weights)
        regression.coef_, regression.gap_

    Attributes:
        coef_: The minimizer c, shaped (p,), set by fit.
        objective_: ``sum_i 2 w_i rho_{tau_i}(y_i - a_i^T c)`` at the current
            iterate, which is ``sum_i w_i |y_i - a_i^T c|`` at ``tau = 1/2``.
        gap_: The duality gap, relative to ``1 + objective_``.
        procedure: The trainer's record of the fit.
    """

    defaults = l1_regression_configs.L1RegressionConfig
    owns_updates = True

    def __init__(self, *, hyperparameters: Optional[dict] = None):
        """Initialize an unfit solver.

        Args:
            hyperparameters: Overrides of ``L1RegressionConfig``.
        """
        super().__init__(hyperparameters=hyperparameters)
        self.coef_: Optional[ch.Tensor] = None
        self.objective_ = float("inf")
        self.gap_ = float("inf")
        # The rows with the weights folded in and their levels, the dual's
        # right-hand side, and the iterate (a, c, z, v): the rows set by fit,
        # the rest by pre_phase_hook.
        self._X = self._y = self._tau = self._b = ch.zeros(0)  # pylint: disable=invalid-name
        self._a = self._c = self._z = self._v = ch.zeros(0)

    def fit(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        y: ch.Tensor,
        weights: Optional[ch.Tensor] = None,
        quantiles: Optional[ch.Tensor] = None,
    ) -> "L1Regression":
        """Fit c to minimize the weighted check loss of the residuals.

        Args:
            X: Features, shaped (n, p), of full column rank.
            y: Targets, shaped (n,).
            weights: Nonnegative row weights, shaped (n,); ones if None.
            quantiles: Each row's level tau in [0, 1], shaped (n,); one half,
                least absolute deviations, if None.

        Returns:
            self.

        Raises:
            ValueError: If the shapes disagree, a weight is negative, or a
                level lies outside [0, 1].
        """
        n = X.size(0)
        if y.shape != (n,):
            raise ValueError(f"y must be shaped ({n},); got {tuple(y.shape)}.")
        features = X.to(self.dtype)
        targets = y.to(device=features.device, dtype=self.dtype)
        w = ch.ones_like(targets) if weights is None else weights.to(targets)
        if w.shape != (n,) or bool((w < 0).any()):
            raise ValueError(f"weights must be nonnegative and shaped ({n},).")
        tau = ch.full_like(targets, 0.5) if quantiles is None else quantiles.to(targets)
        if tau.shape != (n,) or bool(((tau < 0) | (tau > 1)).any()):
            raise ValueError(f"quantiles must lie in [0, 1] and be shaped ({n},).")
        kept = w > 0
        self._X = features[kept] * w[kept, None]  # pylint: disable=invalid-name
        self._y = targets[kept] * w[kept]
        self._tau = tau[kept]
        trainer = delphi.trainer.Trainer(self, self.hyperparameters)
        self.procedure = trainer.train_model(None, None)
        self.coef_ = self._c.to(X.dtype)
        return self

    def pre_phase_hook(self) -> None:
        """Start the iterate: a = 1/2, c by least squares.

        The start is dual feasible only at ``tau = 1/2``; elsewhere the
        Newton steps also close the primal residual.
        """
        X, y = self._X, self._y  # pylint: disable=invalid-name
        self._b = (1.0 - self._tau) @ X
        self._a = ch.full_like(y, 0.5)
        self._c = ch.linalg.lstsq(X, y).solution  # pylint: disable=not-callable
        residual = y - X @ self._c
        shift = float(residual.abs().mean()) + 1.0
        self._z = (-residual).clamp_min(0.0) + shift
        self._v = residual.clamp_min(0.0) + shift
        self._measure()

    def _residuals(self) -> tuple[ch.Tensor, ch.Tensor]:
        """Return the primal and dual residuals at the iterate."""
        X, y = self._X, self._y  # pylint: disable=invalid-name
        primal = self._b - X.T @ self._a
        dual = y - X @ self._c + self._z - self._v
        return primal, dual

    def _measure(self) -> None:
        """Set the objective and the relative duality gap at the iterate."""
        residual = self._y - self._X @ self._c
        self.objective_ = float(
            (2.0 * self._tau * residual.clamp_min(0.0)).sum()
            + (2.0 * (1.0 - self._tau) * (-residual).clamp_min(0.0)).sum()
        )
        complementarity = float(self._a @ self._z + (1.0 - self._a) @ self._v)
        self.gap_ = complementarity / (1.0 + self.objective_)

    def _newton(
        self,
        residuals: tuple[ch.Tensor, ch.Tensor],
        r_lower: ch.Tensor,
        r_upper: ch.Tensor,
        scaling: ch.Tensor,
    ) -> tuple[ch.Tensor, ch.Tensor, ch.Tensor, ch.Tensor]:
        """Solve the Newton system for (da, dc, dz, dv)."""
        r_primal, r_dual = residuals
        X, a = self._X, self._a  # pylint: disable=invalid-name
        s = 1.0 - a
        q = r_dual + r_lower / a - r_upper / s
        system = X.T @ (scaling[:, None] * X)
        dc = ch.linalg.solve(system, X.T @ (scaling * q) - r_primal)  # pylint: disable=not-callable
        da = scaling * (q - X @ dc)
        return da, dc, (r_lower - self._z * da) / a, (r_upper + self._v * da) / s

    def _steps(
        self, da: ch.Tensor, dz: ch.Tensor, dv: ch.Tensor
    ) -> tuple[float, float]:
        """Return the primal and dual step lengths that keep the iterate interior."""
        a, s = self._a, 1.0 - self._a
        primal = min(_step_length(a, da), _step_length(s, -da))
        dual = min(_step_length(self._z, dz), _step_length(self._v, dv))
        return primal, dual

    def train_batch(self, batch: object) -> dict:
        """Take one predictor-corrector Newton step.

        Args:
            batch: Unused; the solver holds its own data.

        Returns:
            The objective after the step, as ``{"loss": ...}``.
        """
        del batch
        a, z, v = self._a, self._z, self._v
        s = 1.0 - a
        residuals = self._residuals()
        mu = float(a @ z + s @ v) / (2 * a.numel())
        scaling = 1.0 / (z / a + v / s)

        # Predictor: the affine-scaling step, aiming at zero complementarity.
        da, dc, dz, dv = self._newton(residuals, -a * z, -s * v, scaling)
        primal, dual = self._steps(da, dz, dv)
        mu_affine = float(
            (a + primal * da) @ (z + dual * dz) + (s - primal * da) @ (v + dual * dv)
        ) / (2 * a.numel())
        sigma = (mu_affine / mu) ** 3

        # Corrector: center, and cancel the predictor's second-order terms.
        r_lower = sigma * mu - a * z - da * dz
        r_upper = sigma * mu - s * v + da * dv
        da, dc, dz, dv = self._newton(residuals, r_lower, r_upper, scaling)
        primal, dual = self._steps(da, dz, dv)
        self._a = a + primal * da
        self._c, self._z, self._v = (
            self._c + dual * dc,
            z + dual * dz,
            v + dual * dv,
        )
        self._measure()
        return {"loss": ch.tensor(self.objective_)}

    def should_stop(self) -> tuple[bool, Optional[str]]:
        """Stop once the duality gap and both residuals meet the tolerance."""
        tolerance = self.hyperparameters.tolerance
        primal, dual = self._residuals()
        done = (
            self.gap_ < tolerance
            and float(primal.norm()) <= tolerance * (1.0 + float(self._b.norm()))
            and float(dual.norm()) <= tolerance * (1.0 + float(self._y.norm()))
        )
        return done, "duality_gap" if done else None

    def evaluate(self) -> dict:
        """Return the objective at the iterate."""
        return {"loss": ch.tensor(self.objective_)}

    def run_metrics(self) -> dict:
        """Return the objective and the duality gap the run ended at."""
        return {"objective": self.objective_, "gap": self.gap_}
