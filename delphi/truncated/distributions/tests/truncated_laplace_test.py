# Author: pstefanou12@
"""Tests for the truncated Laplace (known location) distribution."""

import functools
from typing import Optional

import torch as ch
from torch import distributions

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
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_laplace.TruncatedLaplace,
        target=distributions.Laplace(loc, scale),
        phi=phi,
        dims=dims,
        parameter="scale_",
        empirical=lambda s: (s - loc).abs().mean(0),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        model_args=(loc,),
        rebuild=functools.partial(distributions.Laplace, loc),
    )


def test_one_dimension():
    _recovers(
        dims=1,
        loc=ch.Tensor([0.0]),
        scale=ch.Tensor([1.0]),
        phi=lambda z: z[:, 0] > 1.0,
    )


def test_two_dimensions():
    loc, scale = ch.Tensor([0.0, 1.0]), ch.Tensor([1.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Laplace(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(-20.0, 20.0),
    )
    _recovers(dims=2, loc=loc, scale=scale, phi=calibrated.phi)


def test_two_dimensions_random_half_space():
    loc, scale = ch.Tensor([0.0, 1.0]), ch.Tensor([1.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        distributions.Laplace(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(dims=2, loc=loc, scale=scale, phi=calibrated.phi)


def test_one_dimension_identity():
    _recovers(
        dims=1,
        loc=ch.Tensor([0.0]),
        scale=ch.Tensor([1.0]),
        phi=truncation_sets.identity,
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    loc, scale = ch.zeros(10), ch.linspace(0.5, 2.0, 10)
    _recovers(
        dims=10,
        loc=loc,
        scale=scale,
        phi=truncation_sets.identity,
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_one_dimension_small_mass():
    loc, scale = ch.Tensor([0.0]), ch.Tensor([1.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Laplace(loc, scale),
        target_alpha=(0.008, 0.02),
        bounds=(-20.0, 20.0),
    )
    _recovers(dims=1, loc=loc, scale=scale, phi=calibrated.phi)


def test_ten_dimensions_norm_threshold():
    loc, scale = ch.zeros(10), ch.linspace(0.5, 2.0, 10)
    # A sum bound barely moves |x - loc|, the only statistic here: the naive
    # estimate scores 0.0748 under it. Excluding a ball moves it everywhere.
    calibrated = truncation_sets.calibrate(
        truncation_sets.norm_threshold,
        distributions.Laplace(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 100.0),
    )
    _recovers(dims=10, loc=loc, scale=scale, phi=calibrated.phi)


def test_indistinguishable_pair():
    loc = ch.Tensor([0.0])
    helpers.assert_family_distinguishes(
        model_class=truncated_laplace.TruncatedLaplace,
        targets=(
            distributions.Laplace(loc, ch.Tensor([0.5])),
            distributions.Laplace(loc, ch.Tensor([2.0])),
        ),
        phi=lambda z: (z[:, 0] > 1.0) & (z[:, 0] < 1.3),
        dims=1,
        parameter="scale_",
        empirical=lambda s: (s - loc).abs().mean(0),
        model_args=(loc,),
        rebuild=functools.partial(distributions.Laplace, loc),
    )
