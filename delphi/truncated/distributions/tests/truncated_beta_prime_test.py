# Author: pstefanou12@
"""Tests for the truncated beta prime distribution."""

import functools
from typing import Any, Optional

import pytest
import torch as ch

from delphi.exponential_family import beta_prime
import delphi.oracle
from delphi.truncated.distributions import truncated_beta_prime
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _beta_prime(
    concentration1: ch.Tensor, concentration0: ch.Tensor
) -> beta_prime.BetaPrime:
    return beta_prime.BetaPrime(
        beta_prime.BetaPrime.to_natural(ch.cat([concentration1, concentration0])),
        concentration1.numel(),
    )


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    concentration1: ch.Tensor,
    concentration0: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: bool = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_beta_prime.TruncatedBetaPrime,
        target=_beta_prime(concentration1, concentration0),
        phi=phi,
        parameter=("concentration1_", "concentration0_"),
        empirical=helpers.untruncated_mle(truncated_beta_prime.TruncatedBetaPrime),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        rebuild=_beta_prime,
        known=known,
        **unknown,
    )


# Unknown truncation needs degree 6 here; over seeds 0,1,2 it meets the shared
# bounds.
@helpers.KNOWN_AND_UNKNOWN_CI
def test_one_dimension(known):
    _recovers(
        concentration1=ch.tensor([3.0]),
        concentration0=ch.tensor([4.0]),
        phi=lambda z: z[:, 0] > 1.5,
        known=known,
        hyperparameters={"truncation_set_degree": 6},
    )


# Unknown truncation needs degree 6 here; over seeds 0,1,2 it meets the shared
# bounds.
@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions(known):
    concentration1, concentration0 = ch.tensor([3.0, 2.0]), ch.tensor([4.0, 6.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        _beta_prime(concentration1, concentration0),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 50.0),
    )
    _recovers(
        concentration1=concentration1,
        concentration0=concentration0,
        phi=calibrated.phi,
        known=known,
        hyperparameters={"truncation_set_degree": 6},
    )


# Unknown truncation, degree 6: over seeds 0,1,2 the estimate's KL measured
# 0.101 to 0.234 (ema), 0.100 to 0.238 (avg).
@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_half_space(known):
    concentration1, concentration0 = ch.tensor([3.0, 2.0]), ch.tensor([4.0, 6.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        _beta_prime(concentration1, concentration0),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(
        concentration1=concentration1,
        concentration0=concentration0,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.47, "avg": 0.48},
        hyperparameters={"truncation_set_degree": 6},
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_identity(known):
    _recovers(
        concentration1=ch.tensor([3.0]),
        concentration0=ch.tensor([4.0]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    _recovers(
        concentration1=ch.linspace(1.5, 5.0, 10),
        concentration0=ch.linspace(3.0, 8.0, 10),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


# Unknown truncation, degree 6: over seeds 0,1,2 the estimate's KL measured
# 0.011 to 0.237 (ema), 0.010 to 0.237 (avg).
@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_small_mass(known):
    concentration1, concentration0 = ch.tensor([3.0]), ch.tensor([4.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        _beta_prime(concentration1, concentration0),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 100.0),
    )
    _recovers(
        concentration1=concentration1,
        concentration0=concentration0,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.48, "avg": 0.48},
        hyperparameters={"truncation_set_degree": 6},
    )


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the estimate's KL reaches 0.43 against the naive fit's 0.88",
    )
)
def test_ten_dimensions_sum_threshold(known):
    concentration1 = ch.linspace(1.5, 5.0, 10)
    concentration0 = ch.linspace(3.0, 8.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        _beta_prime(concentration1, concentration0),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 500.0),
    )
    _recovers(
        concentration1=concentration1,
        concentration0=concentration0,
        phi=calibrated.phi,
        known=known,
    )


# Unknown truncation is flaky here: over seeds 0 to 14 at rho 0.3, six fits
# stay too close to tell apart and one learned set misses by 0.36, as CI hit.
# Rho 0.4, 0.5 and degree 2 or 6 fail as often or more.
@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=False,
        reason="unknown truncation: passes about half the seeds",
    ),
)
def test_indistinguishable_pair(known):
    helpers.assert_family_distinguishes(
        model_class=truncated_beta_prime.TruncatedBetaPrime,
        targets=(
            _beta_prime(ch.tensor([2.0]), ch.tensor([3.0])),
            _beta_prime(ch.tensor([6.0]), ch.tensor([11.0])),
        ),
        phi=lambda z: (z[:, 0] > 0.3) & (z[:, 0] < 0.7),
        parameter=("concentration1_", "concentration0_"),
        empirical=helpers.untruncated_mle(truncated_beta_prime.TruncatedBetaPrime),
        rebuild=_beta_prime,
        known=known,
        hyperparameters={"truncation_set_rho": 0.3},
    )


@pytest.mark.slow
@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the estimate's KL 3.3 to 3.5 exceeds the naive fit's 2.8 (seeds 0-1)",
    )
)
def test_twenty_dimensions_small_mass(known):
    concentration1 = ch.linspace(1.5, 5.0, 20)
    concentration0 = ch.linspace(3.0, 8.0, 20)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        _beta_prime(concentration1, concentration0),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 1000.0),
    )
    _recovers(
        concentration1=concentration1,
        concentration0=concentration0,
        phi=calibrated.phi,
        known=known,
    )
