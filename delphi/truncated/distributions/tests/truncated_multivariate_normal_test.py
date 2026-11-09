# Author: pstefanou12@
"""Tests for the TruncatedMultivariateNormal entry point.

Recovery is the estimators' own tests' job; these check that the arguments
select the right estimator, reach it, and that its estimates come back.
"""

from collections.abc import Callable
from typing import Optional

import pytest
import torch as ch
import torch.distributions

import delphi.oracle
from delphi.truncated.distributions import (
    known_truncated_multivariate_normal as known_mvn,
)
from delphi.truncated.distributions import (
    known_truncated_multivariate_normal_known_covariance as known_mvn_known_cov,
)
from delphi.truncated.distributions import truncated_multivariate_normal
from delphi.truncated.distributions import (
    unknown_truncated_multivariate_normal as unknown_mvn,
)
from delphi.truncated.distributions import (
    unknown_truncated_multivariate_normal_known_covariance as unknown_mvn_known_cov,
)

_COVARIANCE = ch.tensor([[1.0, 0.3], [0.3, 1.0]])
_ESTIMATES = ("avg", "ema", "final", "best")
# Short fits: these tests check the plumbing, not the accuracy.
_KNOWN_HYPERPARAMETERS = {"iterations": 20}
_UNKNOWN_HYPERPARAMETERS = {"iterations": 3, "degree": 4}


def _half_plane(x: ch.Tensor) -> ch.Tensor:
    return x[..., 0] > 0.0


def _sample(phi: Callable, size: int = 2000) -> ch.Tensor:
    ch.manual_seed(0)
    population = torch.distributions.MultivariateNormal(
        ch.zeros(2), _COVARIANCE
    ).sample([4 * size])
    return population[phi(population)][:size]


def _model(
    phi: Optional[Callable], covariance_matrix: Optional[ch.Tensor]
) -> truncated_multivariate_normal.TruncatedMultivariateNormal:
    hyperparameters = (
        _UNKNOWN_HYPERPARAMETERS if phi is None else _KNOWN_HYPERPARAMETERS
    )
    return truncated_multivariate_normal.TruncatedMultivariateNormal(
        phi=phi,
        covariance_matrix=covariance_matrix,
        hyperparameters=hyperparameters,
    )


@pytest.mark.parametrize(
    ("phi", "covariance_matrix", "estimator"),
    [
        (_half_plane, None, known_mvn.KnownTruncationMultivariateNormal),
        (
            _half_plane,
            _COVARIANCE,
            known_mvn_known_cov.KnownTruncationMultivariateNormalKnownCovariance,
        ),
        (None, None, unknown_mvn.UnknownTruncationMultivariateNormal),
        (
            None,
            _COVARIANCE,
            unknown_mvn_known_cov.UnknownTruncationMultivariateNormalKnownCovariance,
        ),
    ],
    ids=["known", "known_covariance", "unknown", "unknown_covariance"],
)
def test_arguments_select_the_estimator(phi, covariance_matrix, estimator):
    model = _model(phi, covariance_matrix)
    assert type(model.estimator_) is estimator
    expected = _UNKNOWN_HYPERPARAMETERS if phi is None else _KNOWN_HYPERPARAMETERS
    assert model.estimator_.hyperparameters.iterations == expected["iterations"]


@pytest.mark.parametrize(
    ("phi", "covariance_matrix"),
    [
        (_half_plane, None),
        (_half_plane, _COVARIANCE),
        (None, None),
        (None, _COVARIANCE),
    ],
    ids=["known", "known_covariance", "unknown", "unknown_covariance"],
)
def test_estimates_come_from_the_estimator(phi, covariance_matrix):
    model = _model(phi, covariance_matrix)
    assert model.fit(_sample(_half_plane)) is model
    for name in _ESTIMATES:
        loc = getattr(model, f"{name}_loc_")
        covariance = getattr(model, f"{name}_covariance_matrix_")
        assert ch.equal(loc, getattr(model.estimator_, f"{name}_loc_"))
        if covariance_matrix is None:
            expected = getattr(model.estimator_, f"{name}_covariance_matrix_")
        else:
            expected = covariance_matrix
        assert ch.equal(covariance, expected)
    assert ch.equal(model.loc_, model.avg_loc_)
    assert ch.equal(model.covariance_matrix_, model.avg_covariance_matrix_)


def test_known_truncation_set_is_phi():
    assert _model(_half_plane, None).truncation_set() is _half_plane


def test_unknown_truncation_set_is_recovered():
    model = _model(None, None).fit(_sample(_half_plane))
    assert isinstance(model.truncation_set(), delphi.oracle.RecoveredSet)


def test_alpha_reaches_the_estimator():
    model = truncated_multivariate_normal.TruncatedMultivariateNormal(
        phi=_half_plane, alpha=0.5
    )
    assert model.estimator_.alpha == 0.5


def test_covariance_that_is_not_square_raises():
    with pytest.raises(ValueError, match="covariance_matrix"):
        truncated_multivariate_normal.TruncatedMultivariateNormal(
            covariance_matrix=ch.ones(2, 3)
        )


def test_covariance_narrower_than_the_sample_raises():
    model = truncated_multivariate_normal.TruncatedMultivariateNormal(
        phi=_half_plane, covariance_matrix=_COVARIANCE
    )
    with pytest.raises(ValueError, match="covariance_matrix"):
        model.fit(ch.randn(100, 3))


@pytest.mark.parametrize(
    ("covariance_matrix", "estimator"),
    [
        (None, known_mvn.KnownTruncationMultivariateNormal),
        (
            _COVARIANCE,
            known_mvn_known_cov.KnownTruncationMultivariateNormalKnownCovariance,
        ),
    ],
    ids=["unknown", "unknown_covariance"],
)
def test_l1_method_selects_the_known_estimator_without_phi(
    covariance_matrix, estimator
):
    model = truncated_multivariate_normal.TruncatedMultivariateNormal(
        covariance_matrix=covariance_matrix, method="l1"
    )
    assert type(model.estimator_) is estimator
    assert model.estimator_.phi is None


def test_l1_truncation_set_is_learned():
    model = truncated_multivariate_normal.TruncatedMultivariateNormal(
        method="l1", hyperparameters=_KNOWN_HYPERPARAMETERS
    )
    with pytest.raises(RuntimeError, match="learned by fit"):
        model.truncation_set()
    model.fit(_sample(_half_plane))
    assert isinstance(model.truncation_set(), delphi.oracle.PolynomialThreshold)
    assert model.truncation_set() is model.estimator_.phi


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="method"):
        truncated_multivariate_normal.TruncatedMultivariateNormal(
            method="lasso",  # type: ignore[arg-type]
        )
