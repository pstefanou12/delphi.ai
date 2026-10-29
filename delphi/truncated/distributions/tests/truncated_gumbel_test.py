# Author: pstefanou12@
"""Tests for the truncated Gumbel (known scale) distribution."""

import functools
from typing import Optional

import pytest
import torch as ch
from torch import distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_gumbel
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
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_gumbel.TruncatedGumbel,
        target=distributions.Gumbel(loc, scale),
        phi=phi,
        dims=dims,
        parameter="loc_",
        empirical=helpers.untruncated_mle(
            truncated_gumbel.TruncatedGumbel, dims, scale
        ),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        model_args=(scale,),
        rebuild=lambda fitted: distributions.Gumbel(fitted, scale),
    )


def test_one_dimension():
    _recovers(
        dims=1,
        loc=ch.Tensor([1.0]),
        scale=ch.Tensor([2.0]),
        phi=lambda z: z[:, 0] > 4.0,
    )


def test_two_dimensions():
    loc, scale = ch.Tensor([1.0, -1.0]), ch.Tensor([2.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Gumbel(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(dims=2, loc=loc, scale=scale, phi=calibrated.phi)


def test_two_dimensions_random_half_space():
    loc, scale = ch.Tensor([1.0, -1.0]), ch.Tensor([2.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        distributions.Gumbel(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
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
        distributions.Gumbel(loc, scale),
        target_alpha=(0.008, 0.02),
        bounds=(-50.0, 50.0),
    )
    _recovers(dims=1, loc=loc, scale=scale, phi=calibrated.phi)


def test_ten_dimensions_sum_threshold():
    loc, scale = ch.linspace(-2.0, 2.0, 10), ch.linspace(0.5, 2.0, 10)
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.sum_threshold, below=True),
        distributions.Gumbel(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(-200.0, 200.0),
    )
    _recovers(dims=10, loc=loc, scale=scale, phi=calibrated.phi)


def test_indistinguishable_pair():
    scale = ch.Tensor([1.0])
    helpers.assert_family_distinguishes(
        model_class=truncated_gumbel.TruncatedGumbel,
        targets=(
            distributions.Gumbel(ch.Tensor([0.0]), scale),
            distributions.Gumbel(ch.Tensor([1.0]), scale),
        ),
        phi=lambda z: (z[:, 0] > 0.3) & (z[:, 0] < 0.7),
        dims=1,
        parameter="loc_",
        empirical=helpers.untruncated_mle(truncated_gumbel.TruncatedGumbel, 1, scale),
        model_args=(scale,),
        rebuild=lambda fitted: distributions.Gumbel(fitted, scale),
    )


@pytest.mark.slow
def test_twenty_dimensions_small_mass():
    loc, scale = ch.linspace(-2.0, 2.0, 20), ch.linspace(0.5, 2.0, 20)
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.sum_threshold, below=True),
        distributions.Gumbel(loc, scale),
        target_alpha=(0.008, 0.02),
        bounds=(-400.0, 400.0),
    )
    _recovers(dims=20, loc=loc, scale=scale, phi=calibrated.phi)
