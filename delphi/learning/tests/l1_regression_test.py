# Author: pstefanou12@
"""Tests for the least-absolute-deviations solver."""

import pytest
import torch as ch

from delphi.learning import l1_regression
from delphi.utils import constants


def _fit(design, targets, weights=None) -> l1_regression.L1Regression:
    return l1_regression.L1Regression().fit(design, targets, weights)


def _objective(design, targets, coefficients, weights=None):
    residual = (targets - design @ coefficients).abs()
    return float(residual.sum() if weights is None else (weights * residual).sum())


def _outlier_problem(seed: int = 0):
    """Return a 20-column design whose targets are 20% gross outliers."""
    generator = ch.Generator().manual_seed(seed)
    design = ch.randn(5000, 20, dtype=ch.float64, generator=generator)
    truth = ch.randn(20, dtype=ch.float64, generator=generator)
    targets = design @ truth + 0.01 * ch.randn(
        5000, dtype=ch.float64, generator=generator
    )
    outliers = ch.randperm(5000, generator=generator)[:1000]
    targets[outliers] += 50.0 * ch.randn(1000, dtype=ch.float64, generator=generator)
    return design, targets, truth


def test_intercept_only_fit_is_the_median():
    targets = ch.randn(1001, dtype=ch.float64, generator=ch.Generator().manual_seed(1))
    fit = _fit(ch.ones(1001, 1, dtype=ch.float64), targets)
    assert fit.converged_
    assert abs(float(fit.coef_[0] - targets.median())) < 1e-8


def test_noiseless_targets_are_fit_exactly():
    generator = ch.Generator().manual_seed(2)
    design = ch.randn(500, 10, dtype=ch.float64, generator=generator)
    truth = ch.randn(10, dtype=ch.float64, generator=generator)
    fit = _fit(design, design @ truth)
    ch.testing.assert_close(fit.coef_, truth)
    assert fit.objective_ < 1e-8


def test_gross_outliers_leave_the_fit_near_the_truth():
    design, targets, truth = _outlier_problem()
    fit = _fit(design, targets)
    least_squares = ch.linalg.lstsq(design, targets).solution  # pylint: disable=not-callable
    assert float((fit.coef_ - truth).norm()) < 0.01
    assert float((least_squares - truth).norm()) > 0.5


def test_no_perturbation_lowers_the_objective():
    design, targets, _ = _outlier_problem(seed=3)
    fit = _fit(design, targets)
    best = _objective(design, targets, fit.coef_)
    assert abs(best - fit.objective_) <= 1e-9 * best
    generator = ch.Generator().manual_seed(4)
    for scale in (1e-4, 1e-2, 1.0):
        for _ in range(20):
            step = scale * ch.randn(20, dtype=ch.float64, generator=generator)
            assert _objective(design, targets, fit.coef_ + step) >= best


def test_reports_a_small_duality_gap():
    design, targets, _ = _outlier_problem(seed=5)
    fit = _fit(design, targets)
    assert fit.converged_
    assert fit.gap_ < 1e-8
    assert fit.procedure.iterations < 50


def test_weights_scale_their_rows():
    design, targets, _ = _outlier_problem(seed=6)
    weights = ch.rand(5000, dtype=ch.float64, generator=ch.Generator().manual_seed(7))
    weighted = _fit(design, targets, weights)
    scaled = _fit(weights[:, None] * design, weights * targets)
    ch.testing.assert_close(weighted.coef_, scaled.coef_)


def test_zero_weight_rows_are_ignored():
    design, targets, _ = _outlier_problem(seed=8)
    weights = ch.ones(5000, dtype=ch.float64)
    weights[:2500] = 0.0
    weighted = _fit(design, targets, weights)
    kept = _fit(design[2500:], targets[2500:])
    ch.testing.assert_close(weighted.coef_, kept.coef_)


def test_imbalanced_weights_converge():
    # The positive-unlabeled reduction weighs one class 99 times the other.
    generator = ch.Generator().manual_seed(9)
    design = ch.randn(20000, 200, dtype=ch.float64, generator=generator)
    labels = (design[:, 0] > 0.3).double()
    weights = ch.where(labels > 0, 0.99, 0.01).double()
    fit = _fit(design, labels, weights)
    assert fit.converged_
    assert fit.procedure.iterations < 50


def test_returns_the_design_dtype():
    design = ch.randn(200, 3, generator=ch.Generator().manual_seed(10))
    fit = _fit(design, design @ ch.ones(3))
    assert fit.coef_.dtype == ch.float32


@pytest.mark.parametrize(
    "targets,weights,match",
    [
        (ch.zeros(5), None, "targets"),
        (ch.zeros(10), -ch.ones(10), "weights"),
        (ch.zeros(10), ch.ones(5), "weights"),
    ],
)
def test_rejects_bad_shapes_and_weights(targets, weights, match):
    with pytest.raises(ValueError, match=match):
        _fit(ch.ones(10, 2), targets, weights)


def test_runs_through_the_trainer_without_an_optimizer():
    design, targets, _ = _outlier_problem(seed=11)
    fit = _fit(design, targets)
    assert fit.optimizer is None
    assert fit.procedure.final.stop_reason == constants.StopReason.MODEL_STOP
    assert fit.procedure.series("gap")[-1] == fit.gap_


def test_an_exhausted_budget_is_not_converged():
    design, targets, _ = _outlier_problem(seed=12)
    fit = l1_regression.L1Regression(hyperparameters={"iterations": 2}).fit(
        design, targets
    )
    assert not fit.converged_
    assert fit.procedure.iterations == 2
