# Author: pstefanou12@
"""Tests for the lasso solver."""

import pytest
import torch as ch

from delphi.stats.linear import lasso
from delphi.utils import constants


def _sparse_problem(n: int, p: int, seed: int = 0):
    """Return X shaped (n, p) and y from five unit coefficients plus 0.5."""
    generator = ch.Generator().manual_seed(seed)
    X = ch.randn(n, p, dtype=ch.float64, generator=generator)
    truth = ch.zeros(p, dtype=ch.float64)
    truth[:5] = 1.0
    noise = ch.randn(n, dtype=ch.float64, generator=generator)
    return X, X @ truth + 0.5 + noise


def _assert_kkt(X, y, fit, atol=1e-6):
    """Check the lasso's optimality conditions at the fit.

    ``X^T r / n`` equals ``l1 sign(c)`` where c is nonzero and lies within
    ``[-l1, l1]`` elsewhere, for r the residual; with an intercept, r sums
    to zero.
    """
    intercept = 0.0 if fit.intercept_ is None else fit.intercept_
    residual = y - X @ fit.coef_ - intercept
    correlation = X.T @ residual / X.size(0)
    active = fit.coef_ != 0
    assert bool(active.any())
    ch.testing.assert_close(
        correlation[active],
        fit.l1 * fit.coef_[active].sign(),
        atol=atol,
        rtol=0.0,
    )
    assert float(correlation[~active].abs().max()) <= fit.l1 + atol
    if fit.intercept_ is not None:
        assert abs(float(residual.mean())) <= atol


def test_orthogonal_design_is_soft_thresholding():
    generator = ch.Generator().manual_seed(1)
    basis, _ = ch.linalg.qr(ch.randn(200, 20, dtype=ch.float64, generator=generator))  # pylint: disable=not-callable
    X = basis * 200**0.5
    y = ch.randn(200, dtype=ch.float64, generator=generator)
    fit = lasso.Lasso(0.05, fit_intercept=False).fit(X, y)
    correlation = X.T @ y / 200
    closed_form = correlation.sign() * (correlation.abs() - 0.05).clamp_min(0.0)
    ch.testing.assert_close(fit.coef_, closed_form, atol=1e-7, rtol=0.0)
    assert fit.intercept_ is None


@pytest.mark.parametrize("n, p", [(100, 400), (500, 50)], ids=["wide", "tall"])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_satisfies_the_kkt_conditions(n, p, fit_intercept):
    X, y = _sparse_problem(n, p)
    fit = lasso.Lasso(0.1, fit_intercept=fit_intercept).fit(X, y)
    assert fit.converged_
    _assert_kkt(X, y, fit)


def test_reports_a_small_duality_gap():
    X, y = _sparse_problem(200, 500, seed=2)
    fit = lasso.Lasso(0.1).fit(X, y)
    assert fit.gap_ <= 1e-8 * float((y - y.mean()).pow(2).sum()) / 400
    residual = y - X @ fit.coef_ - fit.intercept_
    objective = residual.pow(2).sum() / 400 + 0.1 * fit.coef_.abs().sum()
    assert abs(float(objective) - fit.objective_) <= 1e-12


def test_a_penalty_above_the_largest_correlation_zeroes_every_coefficient():
    X, y = _sparse_problem(100, 30, seed=3)
    centered = X - X.mean(0)
    largest = float((centered.T @ (y - y.mean())).abs().max()) / 100
    fit = lasso.Lasso(1.01 * largest).fit(X, y)
    assert bool((fit.coef_ == 0).all())
    ch.testing.assert_close(fit.intercept_, y.mean())


def test_a_warm_start_takes_fewer_steps():
    X, y = _sparse_problem(100, 400, seed=4)
    cold = lasso.Lasso(0.05).fit(X, y)
    warm = lasso.Lasso(0.05).fit(X, y, start=cold.coef_)
    assert warm.procedure.iterations < cold.procedure.iterations
    ch.testing.assert_close(warm.coef_, cold.coef_, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_constrained_fit_meets_its_bound_on_the_boundary(fit_intercept):
    X, y = _sparse_problem(100, 400, seed=5)
    bound = 1.5 * 100**0.5
    fit = lasso.Lasso(fit_intercept=fit_intercept).fit_constrained(X, y, bound)
    intercept = 0.0 if fit.intercept_ is None else fit.intercept_
    residual = float((y - X @ fit.coef_ - intercept).norm())
    assert residual <= bound
    assert residual >= bound * (1.0 - 1e-4)
    # The lasso at the penalty found is the constrained form's solution.
    _assert_kkt(X, y, fit)


def test_constrained_fit_is_zero_when_the_bound_admits_it():
    X, y = _sparse_problem(100, 30, seed=6)
    bound = float((y - y.mean()).norm()) + 1.0
    fit = lasso.Lasso().fit_constrained(X, y, bound)
    assert bool((fit.coef_ == 0).all())
    ch.testing.assert_close(fit.intercept_, y.mean())


def test_constrained_fit_rejects_an_unreachable_bound():
    X, y = _sparse_problem(500, 10, seed=7)
    with pytest.raises(ValueError, match="residual"):
        lasso.Lasso().fit_constrained(X, y, 1.0)


def test_returns_the_design_dtype():
    X, y = _sparse_problem(100, 30, seed=8)
    fit = lasso.Lasso(0.1).fit(X.float(), y.float())
    assert fit.coef_.dtype == ch.float32
    assert fit.intercept_.dtype == ch.float32


def test_the_config_sets_the_working_dtype():
    X, y = _sparse_problem(100, 30, seed=8)
    fit = lasso.Lasso(0.1, hyperparameters={"dtype": "float32", "tolerance": 1e-5})
    fit.fit(X, y)
    assert fit.dtype == ch.float32
    assert fit._X.dtype == ch.float32  # pylint: disable=protected-access
    assert fit.converged_
    assert lasso.Lasso().dtype == ch.float64


@pytest.mark.parametrize(
    "l1, y, match", [(0.0, ch.zeros(10), "l1"), (0.1, ch.zeros(5), "y shaped")]
)
def test_rejects_bad_penalties_and_shapes(l1, y, match):
    with pytest.raises(ValueError, match=match):
        lasso.Lasso(l1).fit(ch.ones(10, 2), y)


def test_runs_through_the_trainer_without_an_optimizer():
    X, y = _sparse_problem(100, 50, seed=9)
    fit = lasso.Lasso(0.1).fit(X, y)
    assert fit.optimizer is None
    assert fit.procedure.final.stop_reason == constants.StopReason.MODEL_STOP
    assert fit.procedure.series("gap")[-1] == fit.gap_


def test_an_exhausted_budget_is_not_converged():
    X, y = _sparse_problem(100, 400, seed=10)
    fit = lasso.Lasso(0.01, hyperparameters={"iterations": 3}).fit(X, y)
    assert not fit.converged_
    assert fit.procedure.iterations == 3
