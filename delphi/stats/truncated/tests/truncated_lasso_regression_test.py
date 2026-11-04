# Author: pstefanou12@
"""Tests for truncated lasso regression.

The fits are high dimensional, d = 1000 against n = 400 samples with five
nonzero coefficients, so they are slow. Each checks every estimate against
the truth, the averaged iterate's support, and that the averaged iterate
beats the lasso fit to the same truncated sample at the same penalty. The
sets attenuate that naive fit, as a set selecting on large |y| would not.
Each tolerance is about twice the worst error over seeds 0 to 2, recorded
beside it. The fast tests cover E_r's projection and the start.
"""

import math

import pytest
import torch as ch

from delphi.learning import lasso
from delphi.stats.truncated import truncated_lasso_regression
from delphi.stats.truncated.tests import regression
from delphi.utils import constants

INF = math.inf

# A fit's largest dimension, and the samples it sees.
_D = 1000
_N = 400


def _sparse(value: float, d: int = _D) -> list[float]:
    """Return five coefficients of value followed by zeros, d in all."""
    return [value] * 5 + [0.0] * (d - 5)


def _fit(simulation, *, fit_intercept, alpha_known=True):
    """Fit the model and the naive lasso at its penalty to a simulation."""
    model = truncated_lasso_regression.TruncatedLassoRegression(
        simulation.phi,
        simulation.alpha if alpha_known else None,
        noise_var=simulation.noise_var,
        fit_intercept=fit_intercept,
    )
    model.fit(simulation.X, simulation.y)
    # The model's l1 penalizes the coefficients of y / sigma.
    penalty = model.l1 * math.sqrt(simulation.noise_var)
    naive = lasso.Lasso(penalty, fit_intercept=fit_intercept)
    naive.fit(simulation.X, simulation.y[:, 0])
    return model, naive


def _assert_recovers(simulation, model, naive, tolerance):
    """Check each estimate's error, coef_'s support, and coef_ against naive.

    Args:
        simulation: The sample and its truth.
        model: The fitted model.
        naive: The lasso fit to the same sample.
        tolerance: Largest error of any estimate.
    """
    for name in [*constants.Estimator, ""]:
        prefix = f"{name}_" if name else ""
        coef = getattr(model, f"{prefix}coef_")
        error = simulation.error(coef, getattr(model, f"{prefix}intercept_"))
        assert error <= tolerance, f"{prefix}coef_: error {error:.4f} > {tolerance}"
    support = simulation.coef.flatten() != 0
    coef = model.coef_.flatten().abs()
    assert float(coef[support].min()) > float(coef[~support].max())
    error = simulation.error(model.coef_, model.intercept_)
    naive_error = simulation.error(naive.coef_[None], naive.intercept_)
    assert error < naive_error, f"coef_ {error:.4f}, naive {naive_error:.4f}"


def _interval(seed: int = 0, **options) -> regression.Simulation:
    """Return the sample kept where |y| < 4, without an intercept."""
    phi = regression.intervals((-4.0, 4.0))
    return regression.simulate(_sparse(1.0), 0.0, phi, _N, seed=seed, **options)


@pytest.mark.slow
def test_1000d_interval_truncation():
    simulation = _interval()
    model, naive = _fit(simulation, fit_intercept=False)
    # Over seeds 0 to 2 every estimate's error lies in 0.48 to 0.61, against
    # the naive fit's 0.57 to 0.66.
    _assert_recovers(simulation, model, naive, 1.2)


@pytest.mark.slow
def test_1000d_interval_truncation_alpha_unknown():
    simulation = _interval()
    model, naive = _fit(simulation, fit_intercept=False, alpha_known=False)
    # E_r never binds, so one phase runs, as with alpha known.
    assert model.procedure.stop_reason == "radius_not_binding"
    _assert_recovers(simulation, model, naive, 1.2)


@pytest.mark.slow
def test_1000d_left_truncation_with_intercept():
    simulation = regression.simulate(
        _sparse(0.5), 0.5, regression.intervals((0.0, INF)), _N, seed=0
    )
    model, naive = _fit(simulation, fit_intercept=True)
    # Over seeds 0 to 2 the errors lie in 0.67 to 1.03, against the naive
    # fit's 1.07 to 1.22: the penalty shrinks the coefficients to about 0.1
    # and the intercept absorbs it. Twice the worst would pass the zero
    # estimate, at 1.22, so the bound sits just below that.
    _assert_recovers(simulation, model, naive, 1.15)


@pytest.mark.slow
def test_1000d_interval_truncation_noise_var_4():
    simulation = _interval(noise_var=4.0)
    model, naive = _fit(simulation, fit_intercept=False)
    # Over seeds 0 to 2 the errors lie in 0.98 to 1.45, against the naive
    # fit's 1.32 to 1.57; the shrinkage scales with sigma. Twice the worst
    # would pass the zero estimate, at 2.24.
    _assert_recovers(simulation, model, naive, 1.8)


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
