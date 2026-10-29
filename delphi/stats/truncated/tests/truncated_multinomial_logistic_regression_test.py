# Author: pstefanou12@
"""Tests for truncated multinomial logistic regression.

One case per truncation set on the shared harness, each checked against the
true weights and the naive fit. phi reads one latent coordinate, which
leaves each label's mass in closed form. Each tolerance is about twice the
error of the exact truncated MLE on the same sample, measured offline with
the closed-form NLL.
"""

import math
from typing import Optional

import pytest
import torch as ch

from delphi import stats
from delphi.stats.truncated.tests import classification

INF = math.inf
_COEF_1D = [[1.0], [-1.0]]
# Kept where the reference class's latent, pure noise, is positive.
_SIMPLE = classification.CoordinateSurvival(0, (0.0, INF, 1.0))
_COMPLEX = classification.CoordinateSurvival(
    0, (-INF, -1.0, 1.0), (0.0, 1.0, 1.0), (2.0, INF, 1.0)
)
_NONE = classification.CoordinateSurvival(1, (-INF, INF, 1.0))


def _coef_10d(classes: int) -> list[list[float]]:
    """Return weights of norm 2 for classes 1 to K - 1, on disjoint features."""
    rows = []
    for label in range(classes - 1):
        row = ch.zeros(10)
        row[label :: classes - 1] = 1.0
        rows.append((2.0 * row / row.norm()).tolist())
    return rows


def _fit(
    simulation: classification.Simulation,
    tolerance: float,
    *,
    alpha_known: bool = True,
    naive_margin: Optional[float] = 0.5,
    **hyperparameters,
) -> stats.TruncatedMultinomialLogisticRegression:
    """Fit the model to a simulation and check its estimates.

    Args:
        simulation: The sample and its truth.
        tolerance: As ``classification.assert_recovers``.
        alpha_known: Whether the model is given the smallest label mass.
        naive_margin: As ``classification.assert_recovers``.
        **hyperparameters: Passed to the model.

    Returns:
        The fitted model.
    """
    model = stats.TruncatedMultinomialLogisticRegression(
        hyperparameters, simulation.phi, simulation.alpha if alpha_known else None
    )
    classification.assert_recovers(
        model, simulation, tolerance, naive_margin=naive_margin
    )
    return model


def _simulate_1d(
    phi: classification.CoordinateSurvival, n: int = 2000
) -> classification.Simulation:
    return classification.simulate_multinomial(_COEF_1D, [0.5, -0.5], phi, n)


def _simulate_10d(
    phi: classification.CoordinateSurvival, classes: int = 3, **kwargs
) -> classification.Simulation:
    """Return 5000 survivors of K classes over 10 features."""
    intercept = [0.5] + [0.0] * (classes - 2)
    return classification.simulate_multinomial(
        _coef_10d(classes), intercept, phi, 5000, **kwargs
    )


def test_1d_no_truncation():
    _fit(_simulate_1d(_NONE), 0.13, alpha_known=False, naive_margin=None)


def test_1d_simple_truncation():
    _fit(_simulate_1d(_SIMPLE), 0.06)


def test_1d_complex_truncation():
    _fit(_simulate_1d(_COMPLEX), 0.32)


def test_2d_complex_truncation():
    phi = classification.CoordinateSurvival(1, (-2.0, -0.5, 1.0), (0.5, INF, 1.0))
    covariance = ch.tensor([[1.0, 0.5], [0.5, 1.0]])
    coef = [[1.0, -1.0], [-0.5, 1.0]]
    simulation = classification.simulate_multinomial(
        coef, [0.5, -0.5], phi, 2000, covariance=covariance
    )
    _fit(simulation, 0.25)


def test_10d_simple_truncation():
    _fit(_simulate_10d(_SIMPLE), 0.38)


def test_10d_complex_truncation():
    # The MLE itself lands at 0.61 of the naive fit's error here.
    _fit(_simulate_10d(_COMPLEX), 0.53, naive_margin=0.7)


def test_10d_no_truncation():
    _fit(_simulate_10d(_NONE), 0.5, alpha_known=False, naive_margin=None)


def test_10d_five_classes():
    _fit(_simulate_10d(_SIMPLE, classes=5), 0.86)


@pytest.mark.slow
def test_100d_complex_truncation():
    coef = [(ch.ones(100) / 5.0).tolist(), (-ch.ones(100) / 5.0).tolist()]
    simulation = classification.simulate_multinomial(coef, [0.5, 0.0], _COMPLEX, 20000)
    # The MLE itself lands at 0.63 of the naive fit's error here.
    _fit(simulation, 0.68, naive_margin=0.7)


def test_10d_randomized_truncation():
    phi = classification.CoordinateSurvival(1, (-INF, 0.0, 0.3), (0.0, INF, 1.0))
    _fit(_simulate_10d(phi), 0.52)


def test_10d_alpha_unknown():
    _fit(_simulate_10d(_SIMPLE), 0.38, alpha_known=False)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_10d_seeds(seed):
    _fit(_simulate_10d(_SIMPLE, seed=seed), 0.51)


def test_summary_reports_the_classes():
    simulation = _simulate_1d(_NONE, 200)
    model = stats.TruncatedMultinomialLogisticRegression(
        {"iterations": 20, "refine_steps": 0}, _NONE
    ).fit(simulation.X, simulation.y)
    assert "classes" in model.summary()
    assert model.predict_proba(simulation.X).shape == (200, 3)


def test_alpha_above_one_over_k_raises():
    simulation = _simulate_1d(_NONE, 200)
    model = stats.TruncatedMultinomialLogisticRegression({}, _NONE, 0.4)
    with pytest.raises(ValueError, match="alpha"):
        model.fit(simulation.X, simulation.y)


def test_missing_label_raises():
    model = stats.TruncatedMultinomialLogisticRegression({}, _NONE)
    with pytest.raises(ValueError, match="every label"):
        model.fit(ch.randn(10, 2), ch.tensor([0, 2] * 5)[:, None])
