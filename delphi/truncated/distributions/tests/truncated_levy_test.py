# Author: pstefanou12@
"""Tests for the truncated Lévy (known location) distribution."""

import functools
from typing import Any, Optional

import pytest
import torch as ch

from delphi.exponential_family import levy
import delphi.oracle
from delphi.truncated.distributions import truncated_levy
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _levy(loc: ch.Tensor, scale: ch.Tensor) -> levy.Levy:
    return levy.Levy(loc, levy.Levy.to_natural(scale), scale.numel())


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    loc: ch.Tensor,
    scale: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: bool = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_levy.TruncatedLevy,
        target=_levy(loc, scale),
        phi=phi,
        parameter="scale_",
        empirical=helpers.untruncated_mle(truncated_levy.TruncatedLevy, loc),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        model_args=(loc,),
        rebuild=functools.partial(_levy, loc),
        known=known,
        **unknown,
    )


@helpers.KNOWN_AND_UNKNOWN_CI
def test_one_dimension(known):
    _recovers(
        loc=ch.tensor([1.0]),
        scale=ch.tensor([2.0]),
        phi=lambda z: z[:, 0] < 4.0,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions(known):
    loc, scale = ch.tensor([1.0, -1.0]), ch.tensor([2.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0, below=True),
        _levy(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(1.0, 1000.0),
    )
    _recovers(loc=loc, scale=scale, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_half_space(known):
    loc, scale = ch.tensor([1.0, -1.0]), ch.tensor([2.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 5, 2),
        _levy(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(-1000.0, 1000.0),
    )
    _recovers(loc=loc, scale=scale, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_identity(known):
    _recovers(
        loc=ch.tensor([1.0]),
        scale=ch.tensor([2.0]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    _recovers(
        loc=ch.linspace(-2.0, 2.0, 10),
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
    loc, scale = ch.tensor([1.0]), ch.tensor([2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0, below=True),
        _levy(loc, scale),
        target_alpha=(0.008, 0.02),
        bounds=(1.0, 1e6),
    )
    _recovers(
        loc=loc,
        scale=scale,
        phi=calibrated.phi,
        known=known,
        hyperparameters={"truncation_set_degree": 6},
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_sum_threshold(known):
    loc, scale = ch.linspace(-2.0, 2.0, 10), ch.linspace(0.5, 2.0, 10)
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.sum_threshold, below=True),
        _levy(loc, scale),
        # At 15-25% mass the cut spreads over ten heavy tails, and the naive
        # estimate lands within the tolerance.
        target_alpha=(0.05, 0.1),
        bounds=(0.0, 1e5),
    )
    _recovers(loc=loc, scale=scale, phi=calibrated.phi, known=known)


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the learned window is too coarse to separate the pair",
    )
)
def test_indistinguishable_pair(known):
    loc = ch.tensor([0.0])
    helpers.assert_family_distinguishes(
        model_class=truncated_levy.TruncatedLevy,
        targets=(_levy(loc, ch.tensor([0.5])), _levy(loc, ch.tensor([2.0]))),
        phi=lambda z: (z[:, 0] > 0.3) & (z[:, 0] < 0.5),
        parameter="scale_",
        empirical=helpers.untruncated_mle(truncated_levy.TruncatedLevy, loc),
        model_args=(loc,),
        rebuild=functools.partial(_levy, loc),
        known=known,
    )


@pytest.mark.slow
@helpers.KNOWN_AND_UNKNOWN
def test_twenty_dimensions_small_mass(known):
    loc, scale = ch.linspace(-2.0, 2.0, 20), ch.linspace(0.5, 2.0, 20)
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.sum_threshold, below=True),
        _levy(loc, scale),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 1e7),
    )
    _recovers(loc=loc, scale=scale, phi=calibrated.phi, known=known)
