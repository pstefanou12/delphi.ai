# Author: pstefanou12@
"""Tests for the truncated inverse Gaussian distribution."""

import functools
from typing import Any, Optional

import pytest
import torch as ch

from delphi.exponential_family import inverse_gaussian
import delphi.oracle
from delphi.truncated.distributions import truncated_inverse_gaussian
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _wald(loc: ch.Tensor, concentration: ch.Tensor) -> inverse_gaussian.InverseGaussian:
    return inverse_gaussian.InverseGaussian(
        inverse_gaussian.InverseGaussian.to_natural(ch.cat([loc, concentration])),
        loc.numel(),
    )


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dims: int,
    loc: ch.Tensor,
    concentration: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: bool = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_inverse_gaussian.TruncatedInverseGaussian,
        target=_wald(loc, concentration),
        phi=phi,
        dims=dims,
        parameter=("loc_", "concentration_"),
        empirical=helpers.untruncated_mle(
            truncated_inverse_gaussian.TruncatedInverseGaussian, dims
        ),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        rebuild=_wald,
        known=known,
        **unknown,
    )


@helpers.KNOWN_AND_UNKNOWN_CI
def test_one_dimension(known):
    _recovers(
        dims=1,
        loc=ch.tensor([1.0]),
        concentration=ch.tensor([2.0]),
        phi=lambda z: z[:, 0] > 1.5,
        known=known,
    )


# Unknown truncation, rho 0.3: over seeds 0,1,2 the estimate's KL measured
# 0.061 to 0.378 (ema), 0.064 to 0.369 (avg).
@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions(known):
    loc, concentration = ch.tensor([1.0, 2.0]), ch.tensor([2.0, 4.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        _wald(loc, concentration),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 50.0),
    )
    _recovers(
        dims=2,
        loc=loc,
        concentration=concentration,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.76, "avg": 0.74},
        hyperparameters={"truncation_set_rho": 0.3},
    )


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the estimate's KL reaches 0.61 against the naive fit's 1.49",
    )
)
def test_two_dimensions_random_half_space(known):
    loc, concentration = ch.tensor([1.0, 2.0]), ch.tensor([2.0, 4.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        _wald(loc, concentration),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(
        dims=2, loc=loc, concentration=concentration, phi=calibrated.phi, known=known
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_identity(known):
    _recovers(
        dims=1,
        loc=ch.tensor([1.0]),
        concentration=ch.tensor([2.0]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    _recovers(
        dims=10,
        loc=ch.linspace(0.5, 2.0, 10),
        concentration=ch.linspace(1.0, 5.0, 10),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_small_mass(known):
    loc, concentration = ch.tensor([1.0]), ch.tensor([2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        _wald(loc, concentration),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 50.0),
    )
    _recovers(
        dims=1, loc=loc, concentration=concentration, phi=calibrated.phi, known=known
    )


# Unknown truncation, degree 3: over seeds 0,1,2 the estimate's KL measured
# 0.167 to 0.192 (ema), 0.167 to 0.191 (avg).
@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_sum_threshold(known):
    loc, concentration = ch.linspace(0.5, 2.0, 10), ch.linspace(1.0, 5.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        _wald(loc, concentration),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 200.0),
    )
    _recovers(
        dims=10,
        loc=loc,
        concentration=concentration,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.39, "avg": 0.39},
        hyperparameters={"truncation_set_degree": 3},
    )


@helpers.KNOWN_AND_UNKNOWN
def test_indistinguishable_pair(known):
    helpers.assert_family_distinguishes(
        model_class=truncated_inverse_gaussian.TruncatedInverseGaussian,
        targets=(
            _wald(ch.tensor([1.0]), ch.tensor([3.0])),
            _wald(ch.tensor([3.0]), ch.tensor([1.0])),
        ),
        phi=lambda z: (z[:, 0] > 0.5) & (z[:, 0] < 1.5),
        dims=1,
        parameter=("loc_", "concentration_"),
        empirical=helpers.untruncated_mle(
            truncated_inverse_gaussian.TruncatedInverseGaussian, 1
        ),
        rebuild=_wald,
        known=known,
    )


@pytest.mark.slow
@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the estimate's KL 2.35 to 2.50 against the naive fit's 2.97 (seeds 0-1)",
    )
)
def test_twenty_dimensions_small_mass(known):
    loc, concentration = ch.linspace(0.5, 2.0, 20), ch.linspace(1.0, 5.0, 20)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        _wald(loc, concentration),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 400.0),
    )
    _recovers(
        dims=20, loc=loc, concentration=concentration, phi=calibrated.phi, known=known
    )
