# Author: pstefanou12@
"""Tests for truncated multinomial logistic regression.

One case per truncation set on the shared harness, each checked against the
true weights and the naive fit. Most sets read the margin, the gap between
the top two latents: keeping the hard samples shrinks every class's naive
weights, keeping the easy ones inflates them. The others read one latent
coordinate. Either leaves each label's mass in closed form, and each
tolerance is about twice the error of the exact truncated MLE on the same
sample, measured offline with the closed-form NLL. Each fit's predictions
must also match the true weights' accuracy, and every class's precision and
recall, and its log loss, on held-out untruncated data, and be calibrated
where the naive fit is not.
"""

import math
from typing import Optional

import pytest
import torch as ch

from delphi import stats
from delphi.stats.truncated.tests import classification

INF = math.inf
_COEF_1D = [[1.0], [-1.0]]
_NONE = classification.CoordinateSurvival(1, (-INF, INF, 1.0))
# Class 1's latent outside (0, 2), which biases its weights the most.
_CLASS_ONE = classification.CoordinateSurvival(1, (-INF, 0.0, 1.0), (2.0, INF, 1.0))
_COMPLEX = classification.CoordinateSurvival(
    1, (-INF, -1.0, 1.0), (0.0, 1.0, 1.0), (2.0, INF, 1.0)
)
# Each threshold keeps 15 to 40% of the samples, at that K.
_HARD_3 = classification.MarginSurvival((0.0, 0.8, 1.0))
_EASY_3 = classification.MarginSurvival((0.8, INF, 1.0))
_HARD_5 = classification.MarginSurvival((0.0, 0.7, 1.0))
_EASY_5 = classification.MarginSurvival((1.0, INF, 1.0))
_HARD_10 = classification.MarginSurvival((0.0, 0.3, 1.0))
_EASY_10 = classification.MarginSurvival((1.5, INF, 1.0))


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
    """Fit the model to a simulation and check its estimates and predictions.

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
    classification.assert_classifies(
        model, simulation, naive_miscalibrated=naive_margin is not None
    )
    return model


def _simulate_1d(phi, n: int = 2000) -> classification.Simulation:
    return classification.simulate_multinomial(_COEF_1D, [0.5, -0.5], phi, n)


def _simulate_10d(
    phi, classes: int = 3, n: int = 5000, **kwargs
) -> classification.Simulation:
    """Return n survivors of K classes over 10 features."""
    intercept = [0.5] + [0.0] * (classes - 2)
    return classification.simulate_multinomial(
        _coef_10d(classes), intercept, phi, n, **kwargs
    )


def test_1d_no_truncation():
    _fit(_simulate_1d(_NONE), 0.13, alpha_known=False, naive_margin=None)


def test_1d_label_prior_shift():
    # Class 0's latent is pure noise, so a set on it alone moves the label
    # prior, which the naive fit absorbs into the intercepts.
    simulation = _simulate_1d(classification.CoordinateSurvival(0, (0.0, INF, 1.0)))
    model = _fit(simulation, 0.06)
    # Margin sets mostly rescale the weights, which leaves argmax alone; a
    # prior shift is what moves the naive fit's predictions.
    X, labels = simulation.population  # pylint: disable=invalid-name
    naive = classification.multinomial_labels(X, model.emp_coef_, model.emp_intercept_)
    naive_accuracy = float((naive == labels).float().mean())
    accuracy = float((model.predict(X).flatten() == labels).float().mean())
    assert accuracy - naive_accuracy > 0.03, (accuracy, naive_accuracy)


def test_1d_class_one_truncation():
    _fit(_simulate_1d(_CLASS_ONE), 0.13)


def test_1d_complex_truncation():
    _fit(_simulate_1d(_COMPLEX), 0.09)


def test_2d_complex_truncation():
    phi = classification.CoordinateSurvival(1, (-2.0, -0.5, 1.0), (0.5, INF, 1.0))
    covariance = ch.tensor([[1.0, 0.5], [0.5, 1.0]])
    coef = [[1.0, -1.0], [-0.5, 1.0]]
    simulation = classification.simulate_multinomial(
        coef, [0.5, -0.5], phi, 2000, covariance=covariance
    )
    _fit(simulation, 0.25)


def test_10d_hard_margin():
    _fit(_simulate_10d(_HARD_3), 0.62)


def test_10d_easy_margin():
    _fit(_simulate_10d(_EASY_3), 0.41)


@pytest.mark.slow
def test_10d_randomized_margin():
    phi = classification.MarginSurvival((0.0, 0.8, 1.0), (0.8, INF, 0.1))
    _fit(_simulate_10d(phi), 0.53)


@pytest.mark.slow
def test_10d_class_one_truncation():
    _fit(_simulate_10d(_CLASS_ONE), 0.41)


@pytest.mark.slow
def test_10d_no_truncation():
    _fit(_simulate_10d(_NONE), 0.5, alpha_known=False, naive_margin=None)


@pytest.mark.slow
def test_10d_five_classes_hard_margin():
    _fit(_simulate_10d(_HARD_5, classes=5, n=10000), 0.63)


@pytest.mark.slow
def test_10d_five_classes_easy_margin():
    _fit(_simulate_10d(_EASY_5, classes=5, n=10000), 0.66)


@pytest.mark.slow
def test_10d_unequal_classes_hard_margin():
    # Weight norms 3, 2, 1 and 0.5, and a rare last class, so the margin set
    # biases each class differently instead of rescaling them all alike.
    norms = ch.tensor([3.0, 2.0, 1.0, 0.5])[:, None]
    coef = (ch.tensor(_coef_10d(5)) * norms / 2.0).tolist()
    simulation = classification.simulate_multinomial(
        coef, [0.5, 0.0, 0.0, -1.5], _HARD_5, 10000
    )
    _fit(simulation, 0.67)


@pytest.mark.slow
def test_10d_ten_classes_hard_margin():
    # Sampler noise leaves the estimates at 1.6 times the MLE's error.
    _fit(_simulate_10d(_HARD_10, classes=10, n=20000), 0.95, naive_margin=0.6)


@pytest.mark.slow
def test_10d_ten_classes_easy_margin():
    # Sampler noise leaves the estimates at 1.45 times the MLE's error.
    _fit(_simulate_10d(_EASY_10, classes=10, n=20000), 1.05, naive_margin=0.6)


@pytest.mark.slow
def test_100d_hard_margin():
    coef = [(ch.ones(100) / 5.0).tolist(), (-ch.ones(100) / 5.0).tolist()]
    simulation = classification.simulate_multinomial(coef, [0.5, 0.0], _HARD_3, 20000)
    # The gradient noise at the schedule's first step grows with the 202
    # parameters; at 100 samples a step it leaves the truth's region.
    _fit(simulation, 0.8, batch_size=1000, iterations=200)


@pytest.mark.slow
def test_10d_alpha_unknown():
    _fit(_simulate_10d(_HARD_3), 0.62, alpha_known=False)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_10d_seeds(seed):
    _fit(_simulate_10d(_HARD_3, seed=seed), 0.6)


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
