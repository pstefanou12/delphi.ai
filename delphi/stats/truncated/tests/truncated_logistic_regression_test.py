# Author: pstefanou12@
"""Tests for truncated logistic regression.

One case per truncation set on the shared harness; each tolerance is about
twice the error of the exact truncated MLE on the same sample, measured with
the closed-form NLL. The averaged and final estimates land within a
thousandth of that MLE, the EMA and best iterates within a few hundredths.
"""

import math

import pytest
import torch as ch

from delphi import stats
from delphi.stats.truncated.tests import classification
from delphi.truncated import sampling

INF = math.inf
_NOISE = sampling.LOGISTIC_NOISE
_COEF_10D = (2.0 * ch.ones(10) / math.sqrt(10)).tolist()
_SIMPLE = classification.Truncation((-1.0, INF, 1.0))
_COMPLEX = classification.Truncation(
    (-2.0, -1.0, 1.0), (-0.5, 1.0, 1.0), (2.0, INF, 1.0)
)
_NONE = classification.Truncation((-INF, INF, 1.0))


def _fit(dataset, tolerance, alpha="known", naive_margin=0.5, **hyperparameters):
    """Fit truncated logistic regression to dataset and check its estimates."""
    model = stats.TruncatedLogisticRegression(
        hyperparameters, dataset.phi, dataset.alpha if alpha == "known" else alpha
    )
    classification.assert_recovers(model, dataset, tolerance, naive_margin=naive_margin)
    return model


def _dataset(coef, phi, n, **kwargs):
    return classification.make_dataset(coef, 0.5, phi, n, noise=_NOISE, **kwargs)


def test_1d_no_truncation():
    _fit(_dataset([2.0], _NONE, 2000), 0.21, alpha=None, naive_margin=None)


def test_1d_simple_truncation():
    # Ilyas et al. (2020), Section 4: kept only where z > C, here C = -1.
    _fit(_dataset([2.0], _SIMPLE, 2000), 0.22)


def test_1d_complex_truncation():
    phi = classification.Truncation(
        (-3.0, -2.0, 1.0), (-1.0, 0.5, 1.0), (1.5, INF, 1.0)
    )
    # The MLE itself lands at 0.63 of the naive fit's error here.
    _fit(_dataset([2.0], phi, 2000), 0.25, naive_margin=0.7)


def test_2d_complex_truncation():
    phi = classification.Truncation(
        (-INF, -2.0, 1.0), (-0.5, 0.5, 1.0), (2.0, INF, 1.0)
    )
    covariance = ch.tensor([[1.0, 0.5], [0.5, 1.0]])
    _fit(_dataset([1.0, -1.0], phi, 2000, covariance=covariance), 0.11)


def test_10d_simple_truncation():
    _fit(_dataset(_COEF_10D, _SIMPLE, 5000), 0.38)


def test_10d_complex_truncation():
    _fit(_dataset(_COEF_10D, _COMPLEX, 5000), 0.31)


def test_10d_no_truncation():
    _fit(_dataset(_COEF_10D, _NONE, 5000), 0.25, alpha=None, naive_margin=None)


@pytest.mark.slow
def test_100d_complex_truncation():
    dataset = _dataset((ch.ones(100) / 5.0).tolist(), _COMPLEX, 20000)
    _fit(dataset, 0.38)


def test_10d_randomized_truncation():
    phi = classification.Truncation((-2.0, 0.0, 0.3), (0.0, INF, 1.0))
    _fit(_dataset(_COEF_10D, phi, 5000), 0.5)


def test_10d_alpha_unknown():
    _fit(_dataset(_COEF_10D, _SIMPLE, 5000), 0.38, alpha=None)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_10d_seeds(seed):
    _fit(_dataset(_COEF_10D, _SIMPLE, 5000, seed=seed), 0.39)


def test_alpha_above_one_half_raises():
    with pytest.raises(ValueError, match="alpha"):
        stats.TruncatedLogisticRegression({}, _SIMPLE, 0.6)


def test_one_label_raises():
    model = stats.TruncatedLogisticRegression({}, _SIMPLE)
    with pytest.raises(ValueError, match="one label"):
        model.fit(ch.randn(10, 2), ch.ones(10, 1))
