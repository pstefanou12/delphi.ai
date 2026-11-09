# Author: pstefanou12@
"""Tests for the truncated logarithmic distribution."""

import functools
from typing import Any, Optional

import pytest
import torch as ch

from delphi.exponential_family import logarithmic
import delphi.oracle
from delphi.truncated.distributions import truncated_logarithmic
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _logarithmic(probs: ch.Tensor) -> logarithmic.Logarithmic:
    return logarithmic.Logarithmic(
        logarithmic.Logarithmic.to_natural(probs), probs.numel()
    )


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dims: int,
    probs: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: helpers.Known = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_logarithmic.TruncatedLogarithmic,
        target=_logarithmic(probs),
        phi=phi,
        dims=dims,
        parameter="probs_",
        empirical=helpers.untruncated_mle(
            truncated_logarithmic.TruncatedLogarithmic, dims
        ),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        rebuild=_logarithmic,
        known=known,
        **unknown,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension(known):
    _recovers(dims=1, probs=ch.tensor([0.9]), phi=lambda z: z[:, 0] > 5, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions(known):
    probs = ch.tensor([0.9, 0.7])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        _logarithmic(probs),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 200.0),
    )
    _recovers(dims=2, probs=probs, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_half_space(known):
    probs = ch.tensor([0.9, 0.7])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        _logarithmic(probs),
        target_alpha=(0.15, 0.25),
        bounds=(-200.0, 200.0),
    )
    _recovers(dims=2, probs=probs, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN_CI
def test_one_dimension_identity(known):
    _recovers(
        dims=1,
        probs=ch.tensor([0.9]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    _recovers(
        dims=10,
        probs=ch.linspace(0.5, 0.95, 10),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_small_mass(known):
    probs = ch.tensor([0.9])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        _logarithmic(probs),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 500.0),
    )
    _recovers(dims=1, probs=probs, phi=calibrated.phi, known=known)


# Unknown truncation, rho 0.5: over seeds 0,1,2 the estimate's KL measured
# 0.057 to 0.066 (ema), 0.058 to 0.063 (avg); the set error 0.108 to 0.117.
@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_sum_threshold(known):
    probs = ch.linspace(0.5, 0.95, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        _logarithmic(probs),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 2000.0),
    )
    _recovers(
        dims=10,
        probs=probs,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.14, "avg": 0.15},
        set_tolerance=0.24,
        hyperparameters={"truncation_set_rho": 0.5},
    )


@helpers.KNOWN_AND_UNKNOWN
def test_indistinguishable_pair(known):
    helpers.assert_family_distinguishes(
        model_class=truncated_logarithmic.TruncatedLogarithmic,
        targets=(_logarithmic(ch.tensor([0.5])), _logarithmic(ch.tensor([0.9]))),
        phi=lambda z: (z[:, 0] > 2.5) & (z[:, 0] < 4.5),
        dims=1,
        parameter="probs_",
        empirical=helpers.untruncated_mle(
            truncated_logarithmic.TruncatedLogarithmic, 1
        ),
        rebuild=_logarithmic,
        known=known,
    )


# Unknown truncation: over seeds 0,1 the estimate's KL measured 0.133 to 0.140
# (ema), 0.131 to 0.133 (avg).
@pytest.mark.slow
@helpers.KNOWN_AND_UNKNOWN
def test_twenty_dimensions_small_mass(known):
    probs = ch.linspace(0.5, 0.95, 20)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        _logarithmic(probs),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 4000.0),
    )
    _recovers(
        dims=20,
        probs=probs,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.28, "avg": 0.27},
    )
