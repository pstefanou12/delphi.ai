# Author: pstefanou12@
"""Tests for the truncated Laplace (known location) distribution."""

import functools
from typing import Any, Optional

import pytest
import torch as ch
import torch.distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_laplace
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dims: int,
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
        model_class=truncated_laplace.TruncatedLaplace,
        target=torch.distributions.Laplace(loc, scale),
        phi=phi,
        dims=dims,
        parameter="scale_",
        empirical=lambda s: (s - loc).abs().mean(0),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        model_args=(loc,),
        rebuild=functools.partial(torch.distributions.Laplace, loc),
        known=known,
        **unknown,
    )


@helpers.known_and_unknown(pytest.mark.long, ci=True)
def test_one_dimension(known):
    _recovers(
        dims=1,
        loc=ch.tensor([0.0]),
        scale=ch.tensor([1.0]),
        phi=lambda z: z[:, 0] > 1.0,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions(known):
    loc, scale = ch.tensor([0.0, 1.0]), ch.tensor([1.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.Laplace(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(-20.0, 20.0),
    )
    _recovers(dims=2, loc=loc, scale=scale, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_half_space(known):
    loc, scale = ch.tensor([0.0, 1.0]), ch.tensor([1.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        torch.distributions.Laplace(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(dims=2, loc=loc, scale=scale, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_identity(known):
    _recovers(
        dims=1,
        loc=ch.tensor([0.0]),
        scale=ch.tensor([1.0]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    loc, scale = ch.zeros(10), ch.linspace(0.5, 2.0, 10)
    _recovers(
        dims=10,
        loc=loc,
        scale=scale,
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_small_mass(known):
    loc, scale = ch.tensor([0.0]), ch.tensor([1.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.Laplace(loc, scale),
        target_alpha=(0.008, 0.02),
        bounds=(-20.0, 20.0),
    )
    _recovers(dims=1, loc=loc, scale=scale, phi=calibrated.phi, known=known)


@pytest.mark.long
@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the estimate's KL reaches 0.53 against the naive fit's 0.71",
    )
)
def test_ten_dimensions_norm_threshold(known):
    loc, scale = ch.zeros(10), ch.linspace(0.5, 2.0, 10)
    # A sum bound barely moves |x - loc|, the only statistic here: the naive
    # estimate scores 0.0748 under it. Excluding a ball moves it everywhere.
    calibrated = truncation_sets.calibrate(
        truncation_sets.norm_threshold,
        torch.distributions.Laplace(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 100.0),
    )
    _recovers(dims=10, loc=loc, scale=scale, phi=calibrated.phi, known=known)


@pytest.mark.long
@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the learned window is too coarse to separate the pair",
    )
)
def test_indistinguishable_pair(known):
    loc = ch.tensor([0.0])
    helpers.assert_family_distinguishes(
        model_class=truncated_laplace.TruncatedLaplace,
        targets=(
            torch.distributions.Laplace(loc, ch.tensor([0.5])),
            torch.distributions.Laplace(loc, ch.tensor([2.0])),
        ),
        phi=lambda z: (z[:, 0] > 1.0) & (z[:, 0] < 1.3),
        dims=1,
        parameter="scale_",
        empirical=lambda s: (s - loc).abs().mean(0),
        model_args=(loc,),
        rebuild=functools.partial(torch.distributions.Laplace, loc),
        known=known,
    )
