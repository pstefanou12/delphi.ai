# Author: pstefanou12@
"""Least absolute deviations, ``min_c sum_i w_i |y_i - a_i^T c|``.

Solved as a linear program by the Frisch-Newton interior point method
(Portnoy and Koenker, 1997), with Mehrotra's predictor-corrector. Weights
fold into the rows, as ``w |r| = |w r|`` for ``w >= 0``. The method runs on
the dual,

    max_a  y^T a   s.t.  X^T a = X^T 1 / 2,   0 <= a <= 1,

whose equality multiplier is c. Every Newton step solves one p x p system
``X^T D X``, so the cost is ``O(n p^2)`` per step over a few tens of steps,
and the duality gap certifies the solution. The trainer runs the steps:
each is one of ``L1Regression``'s own updates, with no optimizer.
"""

from typing import Optional

import torch as ch

from delphi.configs.learning import l1_regression as l1_regression_configs
from delphi.delphi import DelphiModel
import delphi.trainer
from delphi.utils import constants

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

    Usage::

        regression = L1Regression().fit(design, targets, weights)
        regression.coef_, regression.gap_

    Attributes:
        coef_: The minimizer c, shaped (p,), set by fit.
        objective_: ``sum_i w_i |y_i - a_i^T c|`` at the current iterate.
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
        self.procedure: Optional[delphi.trainer.ProcedureResult] = None
        # The rows with the weights folded in, the dual's right-hand side,
        # and the iterate (a, c, z, v), all set by fit.
        self._x = self._y = self._b = ch.zeros(0)
        self._a = self._c = self._z = self._v = ch.zeros(0)

    def fit(
        self,
        design: ch.Tensor,
        targets: ch.Tensor,
        weights: Optional[ch.Tensor] = None,
    ) -> "L1Regression":
        """Fit c to minimize the weighted absolute residuals.

        Args:
            design: The matrix X, shaped (n, p), of full column rank.
            targets: y, shaped (n,).
            weights: Nonnegative row weights, shaped (n,); ones if None.

        Returns:
            self.

        Raises:
            ValueError: If the shapes disagree or a weight is negative.
        """
        n = design.size(0)
        if targets.shape != (n,):
            raise ValueError(
                f"targets must be shaped ({n},); got {tuple(targets.shape)}."
            )
        x = design.to(ch.float64)
        y = targets.to(device=x.device, dtype=ch.float64)
        w = ch.ones_like(y) if weights is None else weights.to(y)
        if w.shape != (n,) or bool((w < 0).any()):
            raise ValueError(f"weights must be nonnegative and shaped ({n},).")
        kept = w > 0
        self._x, self._y = x[kept] * w[kept, None], y[kept] * w[kept]
        self._start()
        trainer = delphi.trainer.Trainer(self, self.hyperparameters)
        self.procedure = trainer.train_model(None, None)
        self.coef_ = self._c.to(design.dtype)
        return self

    @property
    def converged_(self) -> bool:
        """Whether the fit stopped on its tolerance rather than its budget."""
        return (
            self.procedure is not None
            and self.procedure.final.stop_reason == constants.StopReason.MODEL_STOP
        )

    def _start(self) -> None:
        """Start from a = 1/2, feasible for the dual, and c from least squares."""
        x, y = self._x, self._y
        self._b = 0.5 * x.sum(0)
        self._a = ch.full_like(y, 0.5)
        self._c = ch.linalg.lstsq(x, y).solution  # pylint: disable=not-callable
        residual = y - x @ self._c
        shift = float(residual.abs().mean()) + 1.0
        self._z = (-residual).clamp_min(0.0) + shift
        self._v = residual.clamp_min(0.0) + shift
        self._measure()

    def _residuals(self) -> tuple[ch.Tensor, ch.Tensor]:
        """Return the primal and dual residuals at the iterate."""
        x, y = self._x, self._y
        primal = self._b - x.T @ self._a
        dual = y - x @ self._c + self._z - self._v
        return primal, dual

    def _measure(self) -> None:
        """Set the objective and the relative duality gap at the iterate."""
        self.objective_ = float((self._y - self._x @ self._c).abs().sum())
        complementarity = float(self._a @ self._z + (1.0 - self._a) @ self._v)
        self.gap_ = complementarity / (1.0 + self.objective_)

    def _newton(self, residuals, r_lower, r_upper, scaling):
        """Solve the Newton system for (da, dc, dz, dv)."""
        r_primal, r_dual = residuals
        x, a = self._x, self._a
        s = 1.0 - a
        q = r_dual + r_lower / a - r_upper / s
        system = x.T @ (scaling[:, None] * x)
        dc = ch.linalg.solve(system, x.T @ (scaling * q) - r_primal)  # pylint: disable=not-callable
        da = scaling * (q - x @ dc)
        return da, dc, (r_lower - self._z * da) / a, (r_upper + self._v * da) / s

    def _steps(self, da, dz, dv) -> tuple[float, float]:
        """Return the primal and dual step lengths that keep the iterate interior."""
        a, s = self._a, 1.0 - self._a
        primal = min(_step_length(a, da), _step_length(s, -da))
        dual = min(_step_length(self._z, dz), _step_length(self._v, dv))
        return primal, dual

    def train_batch(self, batch) -> dict:
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
