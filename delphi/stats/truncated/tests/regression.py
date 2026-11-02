# Author: pstefanou12@
"""Test harness for the truncated regression models.

Any model fit to ``y = w^T x + b + e`` observed through a set on y, the
known- and unknown-variance regressions and the lasso among them:
``simulate`` draws such a sample with known parameters, ``simulate_dynamics`` the observed pairs of a linear
dynamical system's trajectory, and ``assert_recovers`` fits a model to
either and checks every estimate the model returns against the truth. Models observing something else, such as the
classifiers' labels, get a harness module of their own.
"""

from collections.abc import Callable, Sequence
import dataclasses
import math
from typing import Optional, Union

import torch as ch

from delphi.utils import constants

# Draws per block while collecting survivors.
_BLOCK = 100_000


@dataclasses.dataclass(frozen=True)
class Simulation:  # pylint: disable=too-many-instance-attributes
    """A truncated regression sample and the parameters that produced it.

    Attributes:
        X: Observed covariates, shaped (n, d).
        y: Observed responses, shaped (n, 1).
        coef: True coefficients, shaped (k, d).
        intercept: True intercept, shaped (k,).
        noise_var: True noise variance, or covariance shaped (k, k).
        phi: The truncation oracle the sample passed.
        alpha: Fraction of the draws that survived.
        ols_coef: OLS coefficients on the observed sample.
        ols_intercept: OLS intercept on the observed sample.
    """

    X: ch.Tensor  # pylint: disable=invalid-name
    y: ch.Tensor
    coef: ch.Tensor
    intercept: ch.Tensor
    noise_var: Union[float, ch.Tensor]
    phi: Callable
    alpha: float
    ols_coef: ch.Tensor
    ols_intercept: ch.Tensor

    def error(self, coef: ch.Tensor, intercept: Optional[ch.Tensor]) -> float:
        """Return the L2 distance of (coef, intercept) from the truth."""
        estimate = coef.flatten()
        truth = self.coef.flatten()
        if intercept is not None:
            estimate = ch.cat([estimate, intercept.flatten()])
            truth = ch.cat([truth, self.intercept])
        return float((estimate - truth).norm())


def simulate(  # pylint: disable=too-many-arguments,too-many-locals
    coef: Sequence[float],
    intercept: float,
    phi: Callable,
    n: int,
    *,
    noise_var: float = 1.0,
    covariance: Optional[ch.Tensor] = None,
    seed: int = 0,
) -> Simulation:
    """Draw ``y = coef^T x + intercept + e`` until n samples pass phi.

    Args:
        coef: True coefficients.
        intercept: True intercept.
        phi: Oracle over responses shaped (n, m, 1); a float result is a
            survival probability, applied by a coin flip.
        n: Survivors to collect.
        noise_var: Noise variance.
        covariance: Covariance of x ~ N(0, covariance); identity if None.
        seed: Seed for the draws.

    Returns:
        The first n survivors.
    """
    ch.manual_seed(seed)
    weight = ch.tensor(coef)[:, None]
    covariance = ch.eye(weight.size(0)) if covariance is None else covariance
    sampler = ch.distributions.MultivariateNormal(ch.zeros(weight.size(0)), covariance)
    kept_X: list[ch.Tensor] = []  # pylint: disable=invalid-name
    kept_y: list[ch.Tensor] = []
    survived, drawn = 0, 0
    while survived < n:
        X = sampler.sample([_BLOCK])  # pylint: disable=invalid-name
        y = X @ weight + intercept + math.sqrt(noise_var) * ch.randn(_BLOCK, 1)
        kept = phi(y[:, None, :]).reshape(-1)
        kept = ch.bernoulli(kept).bool() if kept.is_floating_point() else kept
        kept_X.append(X[kept])
        kept_y.append(y[kept])
        survived += int(kept.sum())
        drawn += _BLOCK
    X, y = ch.cat(kept_X)[:n], ch.cat(kept_y)[:n]  # pylint: disable=invalid-name
    design = ch.cat([X, ch.ones(n, 1)], dim=1).double()
    ols = ch.linalg.lstsq(design, y.double()).solution.flatten().float()  # pylint: disable=not-callable
    return Simulation(
        X=X,
        y=y,
        coef=weight.T,
        intercept=ch.tensor([intercept]),
        noise_var=noise_var,
        phi=phi,
        alpha=survived / drawn,
        ols_coef=ols[:-1][None, :],
        ols_intercept=ols[-1:],
    )


def simulate_dynamics(
    transition: ch.Tensor,
    phi: Callable,
    steps: int,
    *,
    drift: Optional[ch.Tensor] = None,
    noise_var: Optional[ch.Tensor] = None,
    fit_intercept: bool = True,
    seed: int = 0,
) -> Simulation:
    """Run ``x_{t+1} = A x_t + b + w_t`` from 0 and keep the pairs phi observes.

    Args:
        transition: The A, shaped (d, d).
        phi: Oracle over states shaped (n, m, d), returning a mask.
        steps: Length of the trajectory.
        drift: The b, shaped (d,); zero if None.
        noise_var: Covariance of w_t, shaped (d, d); identity if None.
        fit_intercept: Whether the OLS fit has an intercept.
        seed: Seed for the draws.

    Returns:
        The pairs ``(x_t, x_{t+1})`` with both states observed, in temporal
        order, as X and y. alpha is the fraction of observed states whose
        successor is observed too.
    """
    ch.manual_seed(seed)
    dims = transition.size(0)
    drift = ch.zeros(dims) if drift is None else drift
    covariance = ch.eye(dims) if noise_var is None else noise_var
    noise = ch.randn(steps, dims) @ ch.linalg.cholesky(covariance).T  # pylint: disable=not-callable
    states = [ch.zeros(dims)]
    for step in range(steps):
        states.append(transition @ states[-1] + drift + noise[step])
    trajectory = ch.stack(states)
    observed = phi(trajectory[:, None, :]).reshape(-1)
    paired = observed[:-1] & observed[1:]
    X, y = trajectory[:-1][paired], trajectory[1:][paired]  # pylint: disable=invalid-name
    design = ch.cat([X, ch.ones(X.size(0), 1)], dim=1) if fit_intercept else X
    ols = ch.linalg.lstsq(design.double(), y.double()).solution.T.float()  # pylint: disable=not-callable
    return Simulation(
        X=X,
        y=y,
        coef=transition,
        intercept=drift,
        noise_var=1.0 if noise_var is None else noise_var,
        phi=phi,
        alpha=int(paired.sum()) / int(observed[:-1].sum()),
        ols_coef=ols[:, :dims],
        ols_intercept=ols[:, dims] if fit_intercept else ch.zeros(dims),
    )


def assert_recovers(
    model,
    simulation: Simulation,
    tolerance: float,
    *,
    variance_tolerance: Optional[float] = None,
    ols_margin: Optional[float] = 0.5,
) -> None:
    """Fit a regression model to a simulation and check each estimate it returns.

    Args:
        model: An unfitted truncated regression model, exposing
            ``{avg,ema,final,best}_coef_`` and ``_intercept_``, and
            ``_variance_`` when its ``noise_var`` is None.
        simulation: The sample, with its truth.
        tolerance: Largest L2 error allowed for (coef, intercept).
        variance_tolerance: Largest error allowed for the variance when the
            model learns it; tolerance if None.
        ols_margin: Each estimate's error must also be below this fraction
            of OLS's; None skips the comparison, as without truncation.
    """
    model.fit(simulation.X, simulation.y)
    if variance_tolerance is None:
        variance_tolerance = tolerance
    ols = simulation.error(simulation.ols_coef, simulation.ols_intercept)
    names = [*constants.Estimator, ""]
    for name in names:
        prefix = f"{name}_" if name else ""
        coef = getattr(model, f"{prefix}coef_")
        intercept = getattr(model, f"{prefix}intercept_")
        error = simulation.error(coef, intercept)
        label = f"{prefix}coef_ and {prefix}intercept_"
        assert error <= tolerance, (
            f"{label}: error {error:.4f} above {tolerance} (OLS {ols:.4f})"
        )
        if ols_margin is not None:
            assert error <= ols_margin * ols, (
                f"{label}: error {error:.4f} not below {ols_margin} of OLS {ols:.4f}"
            )
        if getattr(model, "noise_var", 1.0) is None:
            variance = float(getattr(model, f"{prefix}variance_"))
            truth = float(simulation.noise_var)
            assert abs(variance - truth) <= variance_tolerance, (
                f"{prefix}variance_: {variance:.4f} against "
                f"{simulation.noise_var}, beyond {variance_tolerance}"
            )
