# Author: pstefanou12@
"""Tests for the truncated Gamma distribution with unknown shape."""

import functools
from typing import Any, Optional

import pytest
import torch as ch
import torch.distributions

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
    known: bool = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=_MODEL,
        target=torch.distributions.Gamma(concentration, rate),
        phi=phi,
        dims=dims,
        parameter=("concentration_", "rate_"),
        empirical=helpers.untruncated_mle(_MODEL, dims),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        known=known,
        **unknown,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension(known):
    _recovers(
        dims=1,
        concentration=ch.tensor([3.0]),
        rate=ch.tensor([2.0]),
        phi=lambda z: z[:, 0] > 2.0,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions(known):
    concentration, rate = ch.tensor([3.0, 2.0]), ch.tensor([2.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.Gamma(concentration, rate),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 100.0),
    )
    _recovers(
        dims=2, concentration=concentration, rate=rate, phi=calibrated.phi, known=known
    )


# Unknown truncation: over seeds 0,1,2 the estimate's KL measured 0.320 to
# 0.454 (ema), 0.319 to 0.453 (avg).
@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_half_space(known):
    concentration, rate = ch.tensor([3.0, 2.0]), ch.tensor([2.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        torch.distributions.Gamma(concentration, rate),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(
        dims=2,
        concentration=concentration,
        rate=rate,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.91, "avg": 0.91},
    )


@helpers.KNOWN_AND_UNKNOWN_CI
def test_one_dimension_identity(known):
    _recovers(
        dims=1,
        concentration=ch.tensor([3.0]),
        rate=ch.tensor([2.0]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    _recovers(
        dims=10,
        concentration=ch.linspace(1.5, 5.0, 10),
        rate=ch.linspace(0.5, 2.0, 10),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_small_mass(known):
    concentration, rate = ch.tensor([3.0]), ch.tensor([2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.Gamma(concentration, rate),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 100.0),
    )
    _recovers(
        dims=1, concentration=concentration, rate=rate, phi=calibrated.phi, known=known
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_sum_threshold(known):
    concentration, rate = ch.linspace(1.5, 5.0, 10), ch.linspace(0.5, 2.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.Gamma(concentration, rate),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 500.0),
    )
    _recovers(
        dims=10, concentration=concentration, rate=rate, phi=calibrated.phi, known=known
    )


@helpers.KNOWN_AND_UNKNOWN
def test_indistinguishable_pair(known):
    helpers.assert_family_distinguishes(
        model_class=_MODEL,
        targets=(
            torch.distributions.Gamma(ch.tensor([2.0]), ch.tensor([1.0])),
            torch.distributions.Gamma(ch.tensor([6.0]), ch.tensor([3.0])),
        ),
        phi=lambda z: (z[:, 0] > 1.5) & (z[:, 0] < 2.5),
        dims=1,
        parameter=("concentration_", "rate_"),
        empirical=helpers.untruncated_mle(_MODEL, 1),
        known=known,
    )


# Unknown truncation: over seeds 0,1 the estimate's KL measured 0.143 to 0.334
# (ema), 0.143 to 0.333 (avg).
@pytest.mark.slow
@helpers.KNOWN_AND_UNKNOWN
def test_twenty_dimensions_small_mass(known):
    concentration, rate = ch.linspace(1.5, 5.0, 20), ch.linspace(0.5, 2.0, 20)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.Gamma(concentration, rate),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 1000.0),
    )
    _recovers(
        dims=20,
        concentration=concentration,
        rate=rate,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.67, "avg": 0.67},
    )
