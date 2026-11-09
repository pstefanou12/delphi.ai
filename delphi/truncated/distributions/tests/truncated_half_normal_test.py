# Author: pstefanou12@
"""Tests for the truncated half-normal distribution."""

import functools
from typing import Any, Optional

import pytest
import torch as ch
import torch.distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_half_normal
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    scale: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: bool = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_half_normal.TruncatedHalfNormal,
        target=torch.distributions.HalfNormal(scale),
        phi=phi,
        parameter="scale_",
        empirical=lambda s: ch.sqrt((s**2).mean(0)),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        known=known,
        **unknown,
    )


@helpers.KNOWN_AND_UNKNOWN_CI
def test_one_dimension(known):
    _recovers(scale=ch.tensor([1.5]), phi=lambda z: z[:, 0] > 2.0, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions(known):
    scale = ch.tensor([1.5, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.HalfNormal(scale),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 50.0),
    )
    _recovers(scale=scale, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_half_space(known):
    scale = ch.tensor([1.5, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        torch.distributions.HalfNormal(scale),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(scale=scale, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_identity(known):
    _recovers(
        scale=ch.tensor([1.5]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    scale = ch.linspace(0.5, 2.0, 10)
    _recovers(
        scale=scale,
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_small_mass(known):
    scale = ch.tensor([1.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.HalfNormal(scale),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 50.0),
    )
    _recovers(scale=scale, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_sum_threshold(known):
    scale = ch.linspace(0.5, 2.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.HalfNormal(scale),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 200.0),
    )
    _recovers(scale=scale, phi=calibrated.phi, known=known)


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the learned window is too coarse to separate the pair",
    )
)
def test_indistinguishable_pair(known):
    helpers.assert_family_distinguishes(
        model_class=truncated_half_normal.TruncatedHalfNormal,
        targets=(
            torch.distributions.HalfNormal(ch.tensor([0.5])),
            torch.distributions.HalfNormal(ch.tensor([2.0])),
        ),
        phi=lambda z: (z[:, 0] > 1.0) & (z[:, 0] < 1.3),
        parameter="scale_",
        empirical=lambda s: ch.sqrt((s**2).mean(0)),
        known=known,
    )
