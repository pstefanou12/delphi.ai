# Author: pstefanou12@
"""Tests for truncated probit regression.

One case per truncation set on the shared harness, each checked against the
true weights. Each tolerance is about twice the error of the exact truncated
MLE on the same sample, measured offline with the closed-form NLL.
"""

from collections.abc import Sequence
import math
from typing import Optional

import pytest
import torch as ch

from delphi import stats
from delphi.stats.truncated.tests import classification
from delphi.truncated import sampling

INF = math.inf
_NOISE = sampling.GAUSSIAN_NOISE
_COEF_10D = (2.0 * ch.ones(10) / math.sqrt(10)).tolist()
_SIMPLE = classification.PiecewiseSurvival((-1.0, INF, 1.0))
_COMPLEX = classification.PiecewiseSurvival(
    (-2.0, -1.0, 1.0), (-0.5, 1.0, 1.0), (2.0, INF, 1.0)
)
_NONE = classification.PiecewiseSurvival((-INF, INF, 1.0))


def _fit(
    simulation: classification.Simulation,
    tolerance: float,
    *,
    alpha_known: bool = True,
    naive_margin: Optional[float] = 0.5,
    **hyperparameters,
) -> stats.TruncatedProbitRegression:
    """Fit truncated probit regression to a simulation and check its estimates.

    Args:
        simulation: The sample and its truth.
        tolerance: As ``classification.assert_recovers``.
        alpha_known: Whether the model is given the smallest label mass.
        naive_margin: As ``classification.assert_recovers``.
        **hyperparameters: Passed to the model.

    Returns:
        The fitted model.
    """
    model = stats.TruncatedProbitRegression(
        hyperparameters, simulation.phi, simulation.alpha if alpha_known else None
    )
    classification.assert_recovers(
        model, simulation, tolerance, naive_margin=naive_margin
    )
    return model


def _simulate(
    coef: Sequence[float],
    phi: classification.PiecewiseSurvival,
    n: int,
    **kwargs,
) -> classification.Simulation:
    return classification.simulate(coef, 0.5, phi, n, noise=_NOISE, **kwargs)


def test_1d_no_truncation():
    _fit(_simulate([2.0], _NONE, 2000), 0.1, alpha_known=False, naive_margin=None)


def test_1d_simple_truncation():
    _fit(_simulate([2.0], _SIMPLE, 2000), 0.17)


def test_1d_complex_truncation():
    phi = classification.PiecewiseSurvival(
        (-3.0, -2.0, 1.0), (-1.0, 0.5, 1.0), (1.5, INF, 1.0)
    )
    _fit(_simulate([2.0], phi, 2000), 0.2)


def test_2d_complex_truncation():
    phi = classification.PiecewiseSurvival(
        (-INF, -2.0, 1.0), (-0.5, 0.5, 1.0), (2.0, INF, 1.0)
    )
    covariance = ch.tensor([[1.0, 0.5], [0.5, 1.0]])
    _fit(_simulate([1.0, -1.0], phi, 2000, covariance=covariance), 0.21)


def test_10d_simple_truncation():
    _fit(_simulate(_COEF_10D, _SIMPLE, 5000), 0.28)


def test_10d_complex_truncation():
    _fit(_simulate(_COEF_10D, _COMPLEX, 5000), 0.21)


def test_10d_no_truncation():
    _fit(_simulate(_COEF_10D, _NONE, 5000), 0.28, alpha_known=False, naive_margin=None)


@pytest.mark.slow
def test_100d_complex_truncation():
    # The MLE itself lands at half the naive fit's error at this n.
    simulation = _simulate((ch.ones(100) / 5.0).tolist(), _COMPLEX, 20000)
    _fit(simulation, 0.26, naive_margin=0.6)


def test_10d_randomized_truncation():
    phi = classification.PiecewiseSurvival((-2.0, 0.0, 0.3), (0.0, INF, 1.0))
    _fit(_simulate(_COEF_10D, phi, 5000), 0.19)


def test_10d_alpha_unknown():
    _fit(_simulate(_COEF_10D, _SIMPLE, 5000), 0.28, alpha_known=False)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_10d_seeds(seed):
    _fit(_simulate(_COEF_10D, _SIMPLE, 5000, seed=seed), 0.24)


def test_alpha_above_one_half_raises():
    with pytest.raises(ValueError, match="alpha"):
        stats.TruncatedProbitRegression({}, _SIMPLE, 0.6)


def test_one_label_raises():
    model = stats.TruncatedProbitRegression({}, _SIMPLE)
    with pytest.raises(ValueError, match="one label"):
        model.fit(ch.randn(10, 2), ch.ones(10, 1))
