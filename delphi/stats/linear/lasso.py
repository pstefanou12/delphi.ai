# Author: pstefanou12@
"""The lasso, ``min_c ||y - X c - b||^2 / (2n) + l1 ||c||_1``.

Solved by FISTA (Beck and Teboulle, 2009), accelerated proximal gradient
steps of size ``1 / L`` for L the largest eigenvalue of ``X^T X / n``, with
the gradient restart of O'Donoghue and Candes (2015). An unpenalized
intercept b is fit by centering X and y. The duality gap at the dual point
``s r``, for r the residual scaled by s into the dual's feasible set
``||X^T u||_inf <= n l1``, certifies the solution. The trainer runs the
steps: each is one of ``Lasso``'s own updates, with no optimizer.

``fit_constrained`` solves the constrained form,
``min ||c||_1 s.t. ||y - X c - b||_2 <= bound``, as the lasso at the
penalty whose residual norm meets the bound, which is nondecreasing in the
penalty.
"""

import math
from typing import Optional

import torch as ch

from delphi.configs.stats import linear
from delphi.delphi import DelphiModel
import delphi.trainer

# The constrained form's search starts this far below the smallest penalty
# that zeroes every coefficient.
_PENALTY_RANGE = 1e-4


class Lasso(  # pylint: disable=too-many-instance-attributes
    DelphiModel[linear.LassoConfig]
):
    """The lasso, by accelerated proximal gradient steps.

    Usage::

        lasso = Lasso(0.1).fit(X, y)
        lasso.coef_, lasso.intercept_, lasso.gap_

    Attributes:
        l1: The penalty on the coefficients' L1 norm.
        fit_intercept: Whether to fit an unpenalized intercept.
        coef_: The minimizer c, shaped (p,), set by fit.
        intercept_: The intercept b, a scalar tensor, or None without one.
        objective_: The objective at the current iterate.
        gap_: The duality gap at the current iterate.
        procedure: The trainer's record of the last fit.
    """

    defaults = linear.LassoConfig
    owns_updates = True

    def __init__(
        self,
        l1: float = 1.0,
        fit_intercept: bool = True,
        *,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize an unfit solver.

        Args:
            l1: The penalty, positive.
            fit_intercept: Whether to fit an unpenalized intercept.
            hyperparameters: Overrides of ``LassoConfig``.

        Raises:
            ValueError: If l1 is not positive.
        """
        if l1 <= 0.0:
            raise ValueError(f"l1 must be positive; got {l1}.")
        super().__init__(hyperparameters=hyperparameters)
        self.l1 = l1
        self.fit_intercept = fit_intercept
        self.coef_: Optional[ch.Tensor] = None
        self.intercept_: Optional[ch.Tensor] = None
        self.objective_ = float("inf")
        self.gap_ = float("inf")
        # The centered data and the means it was centered by, set by fit;
        # the step size 1 / L; the iterate c, the extrapolated point and
        # FISTA's t_k, set by pre_phase_hook.
        self._X = self._y = self._x_mean = ch.zeros(0)  # pylint: disable=invalid-name
        self._y_mean = 0.0
        self._lipschitz = 1.0
        self._start: Optional[ch.Tensor] = None
        self._c = self._z = ch.zeros(0)
        self._fista_t = 1.0

    def fit(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        y: ch.Tensor,
        start: Optional[ch.Tensor] = None,
    ) -> "Lasso":
        """Fit c and b to minimize the penalized squared residuals.

        Args:
            X: Features, shaped (n, p).
            y: Targets, shaped (n,).
            start: The first iterate for c, shaped (p,); zeros if None.

        Returns:
            self.

        Raises:
            ValueError: If the shapes disagree.
        """
        self._set_data(X, y)
        return self._optimize(start, X.dtype)

    def _optimize(self, start: Optional[ch.Tensor], dtype: ch.dtype) -> "Lasso":
        """Run the steps on the data _set_data holds, from start."""
        self._start = None if start is None else start.to(self._X)
        trainer = delphi.trainer.Trainer(self, self.hyperparameters)
        self.procedure = trainer.train_model(None, None)
        self.coef_ = self._c.to(dtype)
        self.intercept_ = None
        if self.fit_intercept:
            intercept = self._y_mean - self._x_mean @ self._c
            self.intercept_ = intercept.to(dtype)
        return self

    def fit_constrained(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        y: ch.Tensor,
        bound: float,
    ) -> "Lasso":
        """Fit the smallest-L1 c whose residual norm is at most bound.

        Bisects the log penalty, warm-starting each fit at the last, and
        keeps the largest penalty found whose residual meets the bound;
        ``l1`` is set to it.

        Args:
            X: Features, shaped (n, p).
            y: Targets, shaped (n,).
            bound: The bound on ``||y - X c - b||_2``.

        Returns:
            self.

        Raises:
            ValueError: If the shapes disagree, or no coefficients within
                the search's range meet the bound.
        """
        self._set_data(X, y)
        largest = float((self._X.T @ self._y).abs().max()) / X.size(0)
        if largest == 0.0 or float(self._y.norm()) <= bound:
            self.l1 = max(largest, 1e-12)
            return self._optimize(None, X.dtype)
        low, high = math.log(_PENALTY_RANGE * largest), math.log(largest)
        feasible: Optional[tuple[float, ch.Tensor]] = None
        start: Optional[ch.Tensor] = None
        for _ in range(self.hyperparameters.bisection_steps):
            self.l1 = math.exp((low + high) / 2.0)
            start = self._optimize(start, X.dtype).coef_
            if self._residual_norm() <= bound:
                low, feasible = (low + high) / 2.0, (self.l1, self._c.clone())
            else:
                high = (low + high) / 2.0
        if feasible is None:
            self.l1 = math.exp(low)
            self._optimize(start, X.dtype)
            if self._residual_norm() > bound:
                raise ValueError(
                    f"no lasso fit with l1 >= {self.l1:.3e} has a residual "
                    f"norm within {bound:.4f}; its is {self._residual_norm():.4f}."
                )
            return self
        self.l1 = feasible[0]
        return self._optimize(feasible[1], X.dtype)

    def _set_data(self, X: ch.Tensor, y: ch.Tensor) -> None:  # pylint: disable=invalid-name
        """Center the data when fitting an intercept, and size L."""
        n = X.size(0)
        if X.dim() != 2 or y.shape != (n,):
            raise ValueError(
                f"expected X shaped (n, p) and y shaped (n,); got "
                f"{tuple(X.shape)} and {tuple(y.shape)}."
            )
        features = X.to(self.dtype)
        targets = y.to(device=features.device, dtype=self.dtype)
        self._x_mean = ch.zeros(X.size(1), dtype=self.dtype)
        self._y_mean = 0.0
        if self.fit_intercept:
            self._x_mean = features.mean(0)
            self._y_mean = float(targets.mean())
        self._X = features - self._x_mean  # pylint: disable=invalid-name
        self._y = targets - self._y_mean
        spectral = float(ch.linalg.matrix_norm(self._X, ord=2))  # pylint: disable=not-callable
        self._lipschitz = max(spectral**2 / n, 1e-12)

    def _residual_norm(self) -> float:
        """Return ``||y - X c - b||_2`` at the iterate."""
        return float((self._y - self._X @ self._c).norm())

    def pre_phase_hook(self) -> None:
        """Start the iterate at the given start, or at zero."""
        p = self._X.size(1)
        start = self._start
        self._c = ch.zeros(p, dtype=self.dtype) if start is None else start.clone()
        self._z = self._c.clone()
        self._fista_t = 1.0
        self._measure()

    def _shrink(self, point: ch.Tensor) -> ch.Tensor:
        """Return the proximal gradient step from point."""
        n = self._X.size(0)
        gradient = self._X.T @ (self._X @ point - self._y) / n
        moved = point - gradient / self._lipschitz
        threshold = self.l1 / self._lipschitz
        return moved.sign() * (moved.abs() - threshold).clamp_min(0.0)

    def _measure(self) -> None:
        """Set the objective and the duality gap at the iterate."""
        n = self._X.size(0)
        residual = self._y - self._X @ self._c
        self.objective_ = float(
            residual.pow(2).sum() / (2 * n) + self.l1 * self._c.abs().sum()
        )
        correlation = float((self._X.T @ residual).abs().max()) / n
        scale = 1.0 if correlation == 0.0 else min(1.0, self.l1 / correlation)
        dual_point = scale * residual
        dual = float(self._y @ dual_point - dual_point.pow(2).sum() / 2) / n
        self.gap_ = max(self.objective_ - dual, 0.0)

    def train_batch(self, batch: object) -> dict:
        """Take one FISTA step, restarting its extrapolation when it goes uphill.

        Args:
            batch: Unused; the solver holds its own data.

        Returns:
            The objective after the step, as ``{"loss": ...}``.
        """
        del batch
        previous = self._c
        self._c = self._shrink(self._z)
        if float((self._z - self._c) @ (self._c - previous)) > 0.0:
            self._fista_t = 1.0
            self._z = self._c
        else:
            t_next = (1.0 + math.sqrt(1.0 + 4.0 * self._fista_t**2)) / 2.0
            push = (self._fista_t - 1.0) / t_next
            self._z = self._c + push * (self._c - previous)
            self._fista_t = t_next
        self._measure()
        return {"loss": ch.tensor(self.objective_)}

    def should_stop(self) -> tuple[bool, Optional[str]]:
        """Stop once the gap is below tolerance times the objective at zero."""
        scale = float(self._y.pow(2).sum()) / (2 * self._X.size(0))
        done = self.gap_ <= self.hyperparameters.tolerance * max(scale, 1e-12)
        return done, "duality_gap" if done else None

    def evaluate(self) -> dict:
        """Return the objective at the iterate."""
        return {"loss": ch.tensor(self.objective_)}

    def run_metrics(self) -> dict:
        """Return the objective and the duality gap the run ended at."""
        return {"objective": self.objective_, "gap": self.gap_}
