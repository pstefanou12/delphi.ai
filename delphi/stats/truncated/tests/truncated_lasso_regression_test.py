# Author: pstefanou12@
"""Tests for truncated lasso regression.

The recovery cases are sparse and high dimensional: d from 100 to 2000
against fewer samples than dimensions, with five or ten nonzero
coefficients, through intervals, half lines and unions. Each runs on the
shared harness: every estimate against the truth, the averaged iterate's
support, and the averaged iterate against the lasso fit to the same
truncated sample at the same penalty, where the set biases that fit. Each
tolerance is about twice the worst error over seeds 0 to 2, recorded
beside it, unless that would pass the all-zero estimate. The fast tests
cover E_r's projection, the start and the inputs.
"""

from collections.abc import Callable, Sequence
import math

import pytest
import torch as ch

from delphi.stats.linear import lasso
from delphi.stats.truncated import truncated_lasso_regression
from delphi.stats.truncated.tests import regression
from delphi.utils import constants

INF = math.inf


def _sparse(value: float, d: int, k: int = 5) -> list[float]:
    """Return k coefficients of value followed by zeros, d in all."""
    return [value] * k + [0.0] * (d - k)


def _case(
    coef: Sequence[float],
    intercept: float,
    phi: Callable[[ch.Tensor], ch.Tensor],
    n: int,
    **options,
) -> Callable[[int], regression.Simulation]:
    """Return a map from a seed to a sample, as ``regression.simulate``'s."""

    def simulate(seed: int) -> regression.Simulation:
        return regression.simulate(coef, intercept, phi, n, seed=seed, **options)

    return simulate


_INTERVAL = regression.intervals((-4.0, 4.0))
_LEFT = regression.intervals((0.0, INF))
_UNION = regression.intervals((-INF, -2.0), (0.0, 1.0), (2.5, INF))
_CASES: dict[str, Callable[[int], regression.Simulation]] = {
    "100d_interval": _case(_sparse(1.0, 100), 0.0, _INTERVAL, 80),
    "500d_interval": _case(_sparse(1.0, 500), 0.0, _INTERVAL, 250),
    "500d_union_of_intervals": _case(_sparse(1.0, 500), 0.5, _UNION, 250),
    "1000d_interval": _case(_sparse(1.0, 1000), 0.0, _INTERVAL, 400),
    "1000d_left_truncation": _case(_sparse(0.5, 1000), 0.5, _LEFT, 400),
    "1000d_interval_noise_var_4": _case(
        _sparse(1.0, 1000), 0.0, _INTERVAL, 400, noise_var=4.0
    ),
    "1000d_interval_10_sparse": _case(_sparse(1.0, 1000, 10), 0.0, _INTERVAL, 400),
    "2000d_interval": _case(_sparse(1.0, 2000), 0.0, _INTERVAL, 500),
}


def _simulate(case: str, seed: int = 0) -> regression.Simulation:
    """Return one of ``_CASES``' samples."""
    return _CASES[case](seed)


def _fit(
    simulation: regression.Simulation,
    tolerance: float,
    *,
    fit_intercept: bool = True,
    alpha_known: bool = True,
    beats_naive: bool = True,
) -> truncated_lasso_regression.TruncatedLassoRegression:
    """Fit the model to a simulation and check its estimates and support.

    Args:
        simulation: The sample and its truth.
        tolerance: Largest error of any estimate.
        fit_intercept: Whether to fit an intercept.
        alpha_known: Whether the model is given the survival mass.
        beats_naive: Whether coef_ must beat the naive lasso; False where
            the set's selection offsets the naive fit's shrinkage.

    Returns:
        The fitted model.
    """
    model = truncated_lasso_regression.TruncatedLassoRegression(
        simulation.phi,
        simulation.alpha if alpha_known else None,
        noise_var=simulation.noise_var,
        fit_intercept=fit_intercept,
    )
    regression.assert_recovers(model, simulation, tolerance, ols_margin=None)
    assert model.coef_ is not None and model.l1 is not None
    support = simulation.coef.flatten() != 0
    coef = model.coef_.flatten().abs()
    assert float(coef[support].min()) > float(coef[~support].max())
    if not beats_naive:
        return model
    # The model's l1 penalizes the coefficients of y / sigma.
    penalty = model.l1 * math.sqrt(simulation.noise_var)
    naive = lasso.Lasso(penalty, fit_intercept=fit_intercept)
    naive.fit(simulation.X, simulation.y[:, 0])
    assert naive.coef_ is not None
    error = simulation.error(model.coef_, model.intercept_)
    naive_error = simulation.error(naive.coef_[None], naive.intercept_)
    assert error < naive_error, f"coef_ {error:.4f}, naive {naive_error:.4f}"
    return model


# The recovery tests. A left-truncated case below 1000 dimensions is absent:
# at 500 dimensions and 250 samples, as at 100 and 80, the estimates were
# no closer than the zero estimate and missed the support.


def test_100d_interval():
    # Errors 0.82 to 1.15, against the naive fit's 0.88 to 1.18. Twice the
    # worst would pass the zero estimate, at 2.24.
    _fit(_simulate("100d_interval"), 2.0, fit_intercept=False)


@pytest.mark.slow
def test_500d_interval():
    # Errors 0.50 to 0.88, against the naive fit's 0.57 to 0.91.
    _fit(_simulate("500d_interval"), 1.75, fit_intercept=False)


@pytest.mark.slow
def test_500d_union_of_intervals():
    # Errors 0.26 to 0.35. The naive fit's 0.17 to 0.22 is closer: the set
    # keeps large |y|, whose selection inflates the slopes the penalty
    # shrinks.
    _fit(_simulate("500d_union_of_intervals"), 0.7, beats_naive=False)


@pytest.mark.slow
def test_1000d_interval():
    # Errors 0.53 to 0.61, against the naive fit's 0.57 to 0.66.
    _fit(_simulate("1000d_interval"), 1.2, fit_intercept=False)


@pytest.mark.slow
def test_1000d_interval_alpha_unknown():
    model = _fit(
        _simulate("1000d_interval"), 1.2, fit_intercept=False, alpha_known=False
    )
    # E_r never binds, so one phase runs, as with alpha known.
    assert model.procedure.stop_reason == "radius_not_binding"


@pytest.mark.slow
def test_1000d_left_truncation():
    # Errors 0.80 to 1.03, against the naive fit's 1.07 to 1.22: the penalty
    # shrinks the coefficients and the intercept absorbs it. Twice the worst
    # would pass the zero estimate, at 1.22.
    _fit(_simulate("1000d_left_truncation"), 1.15)


@pytest.mark.slow
def test_1000d_interval_noise_var_4():
    # Errors 1.08 to 1.45, against the naive fit's 1.32 to 1.57; the
    # shrinkage scales with sigma. Twice the worst would pass the zero
    # estimate, at 2.24.
    _fit(_simulate("1000d_interval_noise_var_4"), 1.8, fit_intercept=False)


@pytest.mark.slow
def test_1000d_interval_10_sparse():
    # Errors 1.34 to 1.55, level with the naive fit's 1.34 to 1.48: ten
    # coefficients double the shrinkage at 400 samples. Twice the worst
    # would pass the zero estimate, at 3.16.
    _fit(
        _simulate("1000d_interval_10_sparse"),
        2.5,
        fit_intercept=False,
        beats_naive=False,
    )


@pytest.mark.slow
def test_2000d_interval():
    # Errors 0.59 to 0.68, against the naive fit's 0.61 to 0.69.
    _fit(_simulate("2000d_interval"), 1.35, fit_intercept=False)


def _ball(seed: int = 0):
    """Return a wide design, responses and E_r at radius 1 around them."""
    generator = ch.Generator().manual_seed(seed)
    X = ch.randn(30, 60, dtype=ch.float64, generator=generator)
    y = ch.randn(30, 1, dtype=ch.float64, generator=generator)
    anchor = ch.linalg.lstsq(X, y).solution  # pylint: disable=not-callable
    ball = truncated_lasso_regression.ResidualBall(X, y, 1.0, anchor)
    return X, y, ball


def test_projection_lands_on_the_boundary_at_the_nearest_point():
    X, y, ball = _ball()
    outside = 3.0 * ch.randn(60, 1, dtype=ch.float64)
    assert not ball.contains(outside)
    projected = ball.project(outside)
    residual = X @ projected - y
    ch.testing.assert_close(float(residual.norm()), 30**0.5, rtol=1e-6, atol=0.0)
    # Nearest: the step back is along the constraint's gradient X^T r.
    step = outside - projected
    gradient = X.T @ residual
    cosine = float((step.T @ gradient) / (step.norm() * gradient.norm()))
    assert cosine == pytest.approx(1.0, abs=1e-8)


def test_projection_keeps_points_inside():
    X, y, ball = _ball(seed=1)
    inside = ch.linalg.lstsq(X, y).solution  # pylint: disable=not-callable
    inside = inside + 1e-3 * ch.randn(60, 1, dtype=ch.float64)
    assert ball.contains(inside)
    assert ball.project(inside) is inside


def test_an_empty_set_cannot_be_projected_onto():
    generator = ch.Generator().manual_seed(2)
    X = ch.randn(200, 3, dtype=ch.float64, generator=generator)
    y = 5.0 * ch.randn(200, 1, dtype=ch.float64, generator=generator)
    ball = truncated_lasso_regression.ResidualBall(X, y, 1.0, ch.zeros(3, 1))
    with pytest.raises(ValueError, match="empty"):
        ball.project(ch.ones(3, 1, dtype=ch.float64))


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_start_is_the_smallest_l1_point_of_the_projection_set(fit_intercept):
    simulation = regression.simulate(
        _sparse(1.0, 200), 0.5, regression.intervals((0.0, INF)), 100, seed=3
    )
    model = truncated_lasso_regression.TruncatedLassoRegression(
        simulation.phi,
        simulation.alpha,
        fit_intercept=fit_intercept,
        # At the default radius E_r holds zero, which is then the start.
        hyperparameters={"iterations": 1, "radius": 1.0},
    )
    model.fit(simulation.X, simulation.y)
    radius = 1.0
    intercept = 0.0 if model.emp_intercept_ is None else model.emp_intercept_
    residual = simulation.y[:, 0] - simulation.X @ model.emp_coef_[0] - intercept
    assert float(residual.norm()) == pytest.approx(radius * 100**0.5, rel=1e-3)
    # The lasso's optimality conditions at the penalty that meets the bound.
    correlation = simulation.X.T @ residual
    active = model.emp_coef_[0] != 0
    signs = correlation[active].sign() * model.emp_coef_[0][active].sign()
    assert bool((signs > 0).all())
    level = correlation[active].abs()
    assert float(level.max() - level.min()) <= 1e-3 * float(level.max())
    assert float(correlation[~active].abs().max()) <= float(level.max()) * 1.001
    assert model.l1 == pytest.approx(math.sqrt(math.log(200) / 100))


def test_fits_through_the_trainer_on_fewer_samples_than_dimensions():
    simulation = regression.simulate(
        _sparse(1.0, 200), 0.0, regression.intervals((0.0, INF)), 100, seed=4
    )
    model = truncated_lasso_regression.TruncatedLassoRegression(
        simulation.phi, simulation.alpha, hyperparameters={"iterations": 50}
    )
    model.fit(simulation.X, simulation.y)
    assert model.procedure.iterations == 50
    assert model.coef_.shape == (1, 200)
    assert model.predict(simulation.X).shape == (100, 1)
    assert "l1 penalty" in model.summary()
    for name in constants.Estimator:
        assert getattr(model, f"{name}_intercept_").shape == (1,)


@pytest.mark.parametrize(
    "options, match",
    [({"l1": 0.0}, "l1"), ({"noise_var": 0.0}, "noise_var")],
)
def test_rejects_nonpositive_penalties_and_variances(options, match):
    with pytest.raises(ValueError, match=match):
        truncated_lasso_regression.TruncatedLassoRegression(
            regression.intervals((0.0, INF)), 0.5, **options
        )


def test_rejects_more_than_one_response():
    model = truncated_lasso_regression.TruncatedLassoRegression(
        regression.intervals((0.0, INF)), 0.5
    )
    with pytest.raises(ValueError, match="y shaped"):
        model.fit(ch.randn(10, 20), ch.randn(10, 2))
