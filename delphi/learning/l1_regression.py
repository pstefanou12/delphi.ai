# Author: pstefanou12@
"""Least absolute deviations, ``min_c sum_i w_i |y_i - a_i^T c|``.

Solved as a linear program by the Frisch-Newton interior point method
(Portnoy and Koenker, 1997), with Mehrotra's predictor-corrector. Weights
fold into the rows, as ``w |r| = |w r|`` for ``w >= 0``. The method runs on
the dual,

    max_a  y^T a   s.t.  X^T a = X^T 1 / 2,   0 <= a <= 1,

whose equality multiplier is c. Every Newton step solves one p x p system
``X^T D X``, so the cost is ``O(n p^2)`` per iteration over a few tens of
iterations, and the duality gap certifies the solution.
"""

import dataclasses
from typing import Optional

import torch as ch

# The fraction of the distance to the boundary a step may cover.
STEP_FRACTION = 0.99995


@dataclasses.dataclass(frozen=True)
class L1Fit:
    """The solution of an L1 regression.

    Attributes:
        coefficients: The minimizer c, shaped (p,).
        objective: ``sum_i w_i |y_i - a_i^T c|`` at c.
        gap: The duality gap at exit, relative to ``1 + objective``.
        iterations: Newton iterations run.
        converged: Whether the gap and the residuals met the tolerance.
    """

    coefficients: ch.Tensor
    objective: float
    gap: float
    iterations: int
    converged: bool


def _step_length(value: ch.Tensor, direction: ch.Tensor) -> float:
    """Return the largest step in (0, 1] keeping ``value + t direction > 0``."""
    shrinking = direction < 0
    if not bool(shrinking.any()):
        return 1.0
    limit = float((-value[shrinking] / direction[shrinking]).min())
    return min(1.0, STEP_FRACTION * limit)


def l1_regression(  # pylint: disable=too-many-arguments,too-many-locals
    design: ch.Tensor,
    targets: ch.Tensor,
    weights: Optional[ch.Tensor] = None,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> L1Fit:
    """Return the weighted least-absolute-deviations fit of targets on design.

    Args:
        design: The matrix X, shaped (n, p), of full column rank.
        targets: y, shaped (n,).
        weights: Nonnegative row weights w, shaped (n,); ones if None.
        max_iterations: Newton iterations before giving up.
        tolerance: Bound on the relative duality gap and on the relative
            primal and dual residuals.

    Returns:
        The coefficients, the objective and gap there, the iterations run,
        and whether they converged.

    Raises:
        ValueError: If the shapes disagree or a weight is negative.
    """
    n, p = design.shape
    if targets.shape != (n,):
        raise ValueError(f"targets must be shaped ({n},); got {tuple(targets.shape)}.")
    x = design.to(ch.float64)
    y = targets.to(device=x.device, dtype=ch.float64)
    w = ch.ones_like(y) if weights is None else weights.to(y)
    if w.shape != (n,) or bool((w < 0).any()):
        raise ValueError(f"weights must be nonnegative and shaped ({n},).")
    kept = w > 0
    x, y = x[kept] * w[kept, None], y[kept] * w[kept]

    b = 0.5 * x.sum(0)
    a = ch.full_like(y, 0.5)
    c = ch.linalg.lstsq(x, y).solution  # pylint: disable=not-callable
    residual = y - x @ c
    shift = float(residual.abs().mean()) + 1.0
    z, v = (-residual).clamp_min(0.0) + shift, residual.clamp_min(0.0) + shift

    def newton(r_primal, r_dual, r_lower, r_upper, scaling):
        """Solve the Newton system for (da, dc, dz, dv)."""
        s = 1.0 - a
        q = r_dual + r_lower / a - r_upper / s
        system = x.T @ (scaling[:, None] * x)
        rhs = x.T @ (scaling * q) - r_primal
        dc = ch.linalg.solve(system, rhs)  # pylint: disable=not-callable
        da = scaling * (q - x @ dc)
        dz = (r_lower - z * da) / a
        dv = (r_upper + v * da) / s
        return da, dc, dz, dv

    converged, iterations, gap = False, 0, float("inf")
    while iterations < max_iterations:
        s = 1.0 - a
        r_primal = b - x.T @ a
        r_dual = y - x @ c + z - v
        complementarity = float(a @ z + s @ v)
        objective = float((y - x @ c).abs().sum())
        gap = complementarity / (1.0 + objective)
        if (
            gap < tolerance
            and float(r_primal.norm()) <= tolerance * (1.0 + float(b.norm()))
            and float(r_dual.norm()) <= tolerance * (1.0 + float(y.norm()))
        ):
            converged = True
            break
        iterations += 1
        mu = complementarity / (2 * a.numel())
        scaling = 1.0 / (z / a + v / s)

        # Predictor: the affine-scaling step, aiming at zero complementarity.
        da, dc, dz, dv = newton(r_primal, r_dual, -a * z, -s * v, scaling)
        primal_step = min(_step_length(a, da), _step_length(s, -da))
        dual_step = min(_step_length(z, dz), _step_length(v, dv))
        mu_affine = float(
            (a + primal_step * da) @ (z + dual_step * dz)
            + (s - primal_step * da) @ (v + dual_step * dv)
        ) / (2 * a.numel())
        sigma = (mu_affine / mu) ** 3

        # Corrector: center, and cancel the predictor's second-order terms.
        r_lower = sigma * mu - a * z - da * dz
        r_upper = sigma * mu - s * v + da * dv
        da, dc, dz, dv = newton(r_primal, r_dual, r_lower, r_upper, scaling)
        primal_step = min(_step_length(a, da), _step_length(s, -da))
        dual_step = min(_step_length(z, dz), _step_length(v, dv))
        a = a + primal_step * da
        c, z, v = c + dual_step * dc, z + dual_step * dz, v + dual_step * dv

    objective = float((y - x @ c).abs().sum())
    return L1Fit(c.to(design.dtype), objective, gap, iterations, converged)
