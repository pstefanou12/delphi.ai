# Author: pstefanou12@
"""Tests for the least-absolute-deviations solver."""

import pytest
import torch as ch

from delphi.learning import l1_regression


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
    fit = l1_regression.l1_regression(ch.ones(1001, 1, dtype=ch.float64), targets)
    assert fit.converged
    assert abs(float(fit.coefficients[0] - targets.median())) < 1e-8


def test_noiseless_targets_are_fit_exactly():
    generator = ch.Generator().manual_seed(2)
    design = ch.randn(500, 10, dtype=ch.float64, generator=generator)
    truth = ch.randn(10, dtype=ch.float64, generator=generator)
    fit = l1_regression.l1_regression(design, design @ truth)
    ch.testing.assert_close(fit.coefficients, truth)
    assert fit.objective < 1e-8


def test_gross_outliers_leave_the_fit_near_the_truth():
    design, targets, truth = _outlier_problem()
    fit = l1_regression.l1_regression(design, targets)
    least_squares = ch.linalg.lstsq(design, targets).solution  # pylint: disable=not-callable
    assert float((fit.coefficients - truth).norm()) < 0.01
    assert float((least_squares - truth).norm()) > 0.5


def test_no_perturbation_lowers_the_objective():
    design, targets, _ = _outlier_problem(seed=3)
    fit = l1_regression.l1_regression(design, targets)
    best = _objective(design, targets, fit.coefficients)
    assert abs(best - fit.objective) <= 1e-9 * best
    generator = ch.Generator().manual_seed(4)
    for scale in (1e-4, 1e-2, 1.0):
        for _ in range(20):
            step = scale * ch.randn(20, dtype=ch.float64, generator=generator)
            assert _objective(design, targets, fit.coefficients + step) >= best


def test_reports_a_small_duality_gap():
    design, targets, _ = _outlier_problem(seed=5)
    fit = l1_regression.l1_regression(design, targets)
    assert fit.converged
    assert fit.gap < 1e-8
    assert fit.iterations < 50


def test_weights_scale_their_rows():
    design, targets, _ = _outlier_problem(seed=6)
    weights = ch.rand(5000, dtype=ch.float64, generator=ch.Generator().manual_seed(7))
    weighted = l1_regression.l1_regression(design, targets, weights)
    scaled = l1_regression.l1_regression(weights[:, None] * design, weights * targets)
    ch.testing.assert_close(weighted.coefficients, scaled.coefficients)


def test_zero_weight_rows_are_ignored():
    design, targets, _ = _outlier_problem(seed=8)
    weights = ch.ones(5000, dtype=ch.float64)
    weights[:2500] = 0.0
    weighted = l1_regression.l1_regression(design, targets, weights)
    kept = l1_regression.l1_regression(design[2500:], targets[2500:])
    ch.testing.assert_close(weighted.coefficients, kept.coefficients)


def test_imbalanced_weights_converge():
    # The positive-unlabeled reduction weighs one class 99 times the other.
    generator = ch.Generator().manual_seed(9)
    design = ch.randn(20000, 200, dtype=ch.float64, generator=generator)
    labels = (design[:, 0] > 0.3).double()
    weights = ch.where(labels > 0, 0.99, 0.01).double()
    fit = l1_regression.l1_regression(design, labels, weights)
    assert fit.converged
    assert fit.iterations < 50


def test_returns_the_design_dtype():
    design = ch.randn(200, 3, generator=ch.Generator().manual_seed(10))
    fit = l1_regression.l1_regression(design, design @ ch.ones(3))
    assert fit.coefficients.dtype == ch.float32


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
        l1_regression.l1_regression(ch.ones(10, 2), targets, weights)
