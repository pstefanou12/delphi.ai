# Author: pstefanou12@
"""Tests for the truncated Gamma distribution with unknown shape."""

import functools
from typing import Optional

import pytest
import torch as ch
from torch import distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_gamma_unknown_concentration
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)

_MODEL = truncated_gamma_unknown_concentration.TruncatedGammaUnknownConcentration


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dims: int,
    concentration: ch.Tensor,
    rate: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    helpers.assert_family_recovers(
        model_class=_MODEL,
        target=distributions.Gamma(concentration, rate),
        phi=phi,
        dims=dims,
        parameter=("concentration_", "rate_"),
        empirical=helpers.untruncated_mle(_MODEL, dims),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
    )


def test_one_dimension():
    _recovers(
        dims=1,
        concentration=ch.Tensor([3.0]),
        rate=ch.Tensor([2.0]),
        phi=lambda z: z[:, 0] > 2.0,
    )


def test_two_dimensions():
    concentration, rate = ch.Tensor([3.0, 2.0]), ch.Tensor([2.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Gamma(concentration, rate),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 100.0),
    )
    _recovers(dims=2, concentration=concentration, rate=rate, phi=calibrated.phi)


def test_two_dimensions_random_half_space():
    concentration, rate = ch.Tensor([3.0, 2.0]), ch.Tensor([2.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        distributions.Gamma(concentration, rate),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(dims=2, concentration=concentration, rate=rate, phi=calibrated.phi)


def test_one_dimension_identity():
    _recovers(
        dims=1,
        concentration=ch.Tensor([3.0]),
        rate=ch.Tensor([2.0]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    _recovers(
        dims=10,
        concentration=ch.linspace(1.5, 5.0, 10),
        rate=ch.linspace(0.5, 2.0, 10),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_one_dimension_small_mass():
    concentration, rate = ch.Tensor([3.0]), ch.Tensor([2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Gamma(concentration, rate),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 100.0),
    )
    _recovers(dims=1, concentration=concentration, rate=rate, phi=calibrated.phi)


def test_ten_dimensions_sum_threshold():
    concentration, rate = ch.linspace(1.5, 5.0, 10), ch.linspace(0.5, 2.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        distributions.Gamma(concentration, rate),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 500.0),
    )
    _recovers(dims=10, concentration=concentration, rate=rate, phi=calibrated.phi)


def test_indistinguishable_pair():
    helpers.assert_family_distinguishes(
        model_class=_MODEL,
        targets=(
            distributions.Gamma(ch.Tensor([2.0]), ch.Tensor([1.0])),
            distributions.Gamma(ch.Tensor([6.0]), ch.Tensor([3.0])),
        ),
        phi=lambda z: (z[:, 0] > 1.5) & (z[:, 0] < 2.5),
        dims=1,
        parameter=("concentration_", "rate_"),
        empirical=helpers.untruncated_mle(_MODEL, 1),
    )


@pytest.mark.slow
def test_twenty_dimensions_small_mass():
    concentration, rate = ch.linspace(1.5, 5.0, 20), ch.linspace(0.5, 2.0, 20)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        distributions.Gamma(concentration, rate),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 1000.0),
    )
    _recovers(dims=20, concentration=concentration, rate=rate, phi=calibrated.phi)
