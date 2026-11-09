# Author: pstefanou12@
"""Tests for the truncated Maxwell distribution."""

import functools
from typing import Any, Optional

import pytest
import torch as ch

from delphi.exponential_family import maxwell
import delphi.oracle
from delphi.truncated.distributions import truncated_maxwell
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _maxwell(scale: ch.Tensor) -> maxwell.Maxwell:
    return maxwell.Maxwell(maxwell.Maxwell.to_natural(scale), scale.numel())


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dims: int,
    scale: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: helpers.Known = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_maxwell.TruncatedMaxwell,
        target=_maxwell(scale),
        phi=phi,
        dims=dims,
        parameter="scale_",
        empirical=helpers.untruncated_mle(truncated_maxwell.TruncatedMaxwell, dims),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        rebuild=_maxwell,
        known=known,
        **unknown,
    )


@helpers.KNOWN_AND_UNKNOWN_CI
def test_one_dimension(known):
    _recovers(dims=1, scale=ch.tensor([1.5]), phi=lambda z: z[:, 0] > 3.0, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions(known):
    scale = ch.tensor([1.5, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        _maxwell(scale),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 50.0),
    )
    _recovers(dims=2, scale=scale, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_half_space(known):
    scale = ch.tensor([1.5, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        _maxwell(scale),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(dims=2, scale=scale, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_identity(known):
    _recovers(
        dims=1,
        scale=ch.tensor([1.5]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    _recovers(
        dims=10,
        scale=ch.linspace(0.5, 2.0, 10),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


# Unknown truncation needs degree 6 here; over seeds 0,1,2 it meets the shared
# bounds.
@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_small_mass(known):
    scale = ch.tensor([1.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        _maxwell(scale),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 50.0),
    )
    _recovers(
        dims=1,
        scale=scale,
        phi=calibrated.phi,
        known=known,
        hyperparameters={"truncation_set_degree": 6},
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_sum_threshold(known):
    scale = ch.linspace(0.5, 2.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        _maxwell(scale),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 200.0),
    )
    _recovers(dims=10, scale=scale, phi=calibrated.phi, known=known)


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the learned window is too coarse to separate the pair",
    )
)
def test_indistinguishable_pair(known):
    helpers.assert_family_distinguishes(
        model_class=truncated_maxwell.TruncatedMaxwell,
        targets=(_maxwell(ch.tensor([0.5])), _maxwell(ch.tensor([1.5]))),
        phi=lambda z: (z[:, 0] > 1.0) & (z[:, 0] < 1.2),
        dims=1,
        parameter="scale_",
        empirical=helpers.untruncated_mle(truncated_maxwell.TruncatedMaxwell, 1),
        rebuild=_maxwell,
        known=known,
    )


# Unknown truncation: over seeds 0,1 the estimate's KL measured 0.206 to 0.214
# (ema), 0.198 to 0.215 (avg).
@pytest.mark.slow
@helpers.KNOWN_AND_UNKNOWN
def test_twenty_dimensions_small_mass(known):
    scale = ch.linspace(0.5, 2.0, 20)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        _maxwell(scale),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 400.0),
    )
    _recovers(
        dims=20,
        scale=scale,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.43, "avg": 0.43},
    )
