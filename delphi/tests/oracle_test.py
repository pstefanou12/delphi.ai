# Author: pstefanou12@
"""Tests for the stateful truncation sets."""

import pytest
import torch as ch
from torch import distributions

import delphi.oracle
from delphi.transformations import standardization


def _half_plane_sample() -> ch.Tensor:
    ch.manual_seed(0)
    population = ch.randn(40000, 2, dtype=ch.float64)
    return population[population[:, 0] > 0.0][:10000]


def _recovered() -> delphi.oracle.RecoveredSet:
    sample = _half_plane_sample()
    transform = standardization.Standardization(ch.eye(2, dtype=ch.float64))
    transform.fit(sample)
    return delphi.oracle.RecoveredSet(20, transform).fit(sample)


def _truth_in_its_coordinates(recovered: delphi.oracle.RecoveredSet):
    # The standard normal, seen through the fitted standardization.
    transform = recovered.transformation
    assert isinstance(transform, standardization.Standardization)
    assert transform.mean is not None
    return distributions.MultivariateNormal(
        -transform.mean, ch.eye(2, dtype=ch.float64)
    )


def test_identity_admits_every_sample():
    assert bool(delphi.oracle.Identity()(ch.randn(5, 3)).all())


def test_membership_needs_a_fitted_distribution():
    with pytest.raises(RuntimeError, match="given"):
        _recovered()(ch.zeros(3, 2))


def test_given_needs_a_fit():
    with pytest.raises(RuntimeError, match="fit"):
        delphi.oracle.RecoveredSet(4).given(
            distributions.MultivariateNormal(ch.zeros(2), ch.eye(2))
        )


def test_threshold_is_half_the_median_ratio_on_the_samples():
    recovered = _recovered()
    bound = recovered.given(_truth_in_its_coordinates(recovered))
    z = recovered.transform(_half_plane_sample())
    assert bound.threshold == pytest.approx(0.5 * float(bound.ratio(z).median()))
    assert recovered.distribution is None, "given must not change the original"


def test_recovers_a_half_plane_from_its_samples():
    recovered = _recovered()
    bound = recovered.given(_truth_in_its_coordinates(recovered))
    ch.manual_seed(1)
    population = ch.randn(20000, 2, dtype=ch.float64)
    error = float((bound(population) != (population[:, 0] > 0.0)).float().mean())
    assert error <= 0.05
