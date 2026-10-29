# Author: pstefanou12@
"""Tests for the truncated Lévy (known location) distribution."""

import functools
from typing import Optional

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
    dims: int,
    loc: ch.Tensor,
    scale: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_levy.TruncatedLevy,
        target=_levy(loc, scale),
        phi=phi,
        dims=dims,
        parameter="scale_",
        empirical=helpers.untruncated_mle(truncated_levy.TruncatedLevy, dims, loc),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        model_args=(loc,),
        rebuild=functools.partial(_levy, loc),
    )


def test_one_dimension():
    _recovers(
        dims=1,
        loc=ch.Tensor([1.0]),
        scale=ch.Tensor([2.0]),
        phi=lambda z: z[:, 0] < 4.0,
    )


def test_two_dimensions():
    loc, scale = ch.Tensor([1.0, -1.0]), ch.Tensor([2.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0, below=True),
        _levy(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(1.0, 1000.0),
    )
    _recovers(dims=2, loc=loc, scale=scale, phi=calibrated.phi)


def test_two_dimensions_random_half_space():
    loc, scale = ch.Tensor([1.0, -1.0]), ch.Tensor([2.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 5, 2),
        _levy(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(-1000.0, 1000.0),
    )
    _recovers(dims=2, loc=loc, scale=scale, phi=calibrated.phi)


def test_one_dimension_identity():
    _recovers(
        dims=1,
        loc=ch.Tensor([1.0]),
        scale=ch.Tensor([2.0]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    _recovers(
        dims=10,
        loc=ch.linspace(-2.0, 2.0, 10),
        scale=ch.linspace(0.5, 2.0, 10),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_one_dimension_small_mass():
    loc, scale = ch.Tensor([1.0]), ch.Tensor([2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0, below=True),
        _levy(loc, scale),
        target_alpha=(0.008, 0.02),
        bounds=(1.0, 1e6),
    )
    _recovers(dims=1, loc=loc, scale=scale, phi=calibrated.phi)


def test_ten_dimensions_sum_threshold():
    loc, scale = ch.linspace(-2.0, 2.0, 10), ch.linspace(0.5, 2.0, 10)
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.sum_threshold, below=True),
        _levy(loc, scale),
        # At 15-25% mass the cut spreads over ten heavy tails, and the naive
        # estimate lands within the tolerance.
        target_alpha=(0.05, 0.1),
        bounds=(0.0, 1e5),
    )
    _recovers(dims=10, loc=loc, scale=scale, phi=calibrated.phi)


def test_indistinguishable_pair():
    loc = ch.Tensor([0.0])
    helpers.assert_family_distinguishes(
        model_class=truncated_levy.TruncatedLevy,
        targets=(_levy(loc, ch.Tensor([0.5])), _levy(loc, ch.Tensor([2.0]))),
        phi=lambda z: (z[:, 0] > 0.3) & (z[:, 0] < 0.5),
        dims=1,
        parameter="scale_",
        empirical=helpers.untruncated_mle(truncated_levy.TruncatedLevy, 1, loc),
        model_args=(loc,),
        rebuild=functools.partial(_levy, loc),
    )


@pytest.mark.slow
def test_twenty_dimensions_small_mass():
    loc, scale = ch.linspace(-2.0, 2.0, 20), ch.linspace(0.5, 2.0, 20)
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.sum_threshold, below=True),
        _levy(loc, scale),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 1e7),
    )
    _recovers(dims=20, loc=loc, scale=scale, phi=calibrated.phi)
