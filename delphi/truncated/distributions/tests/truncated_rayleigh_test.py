# Author: pstefanou12@
"""Tests for the truncated Rayleigh distribution."""

import functools
from typing import Optional

import pytest
import torch as ch

from delphi.exponential_family import rayleigh
import delphi.oracle
from delphi.truncated.distributions import truncated_rayleigh
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _rayleigh(scale: ch.Tensor) -> rayleigh.Rayleigh:
    return rayleigh.Rayleigh(rayleigh.Rayleigh.to_natural(scale), scale.numel())


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dims: int,
    scale: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_rayleigh.TruncatedRayleigh,
        target=_rayleigh(scale),
        phi=phi,
        dims=dims,
        parameter="scale_",
        empirical=helpers.untruncated_mle(truncated_rayleigh.TruncatedRayleigh, dims),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        rebuild=_rayleigh,
    )


def test_one_dimension():
    _recovers(dims=1, scale=ch.Tensor([1.5]), phi=lambda z: z[:, 0] > 3.0)


def test_two_dimensions():
    scale = ch.Tensor([1.5, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        _rayleigh(scale),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 50.0),
    )
    _recovers(dims=2, scale=scale, phi=calibrated.phi)


def test_two_dimensions_random_half_space():
    scale = ch.Tensor([1.5, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        _rayleigh(scale),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(dims=2, scale=scale, phi=calibrated.phi)


def test_one_dimension_identity():
    _recovers(
        dims=1,
        scale=ch.Tensor([1.5]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    _recovers(
        dims=10,
        scale=ch.linspace(0.5, 2.0, 10),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_one_dimension_small_mass():
    scale = ch.Tensor([1.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        _rayleigh(scale),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 50.0),
    )
    _recovers(dims=1, scale=scale, phi=calibrated.phi)


def test_ten_dimensions_sum_threshold():
    scale = ch.linspace(0.5, 2.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        _rayleigh(scale),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 200.0),
    )
    _recovers(dims=10, scale=scale, phi=calibrated.phi)


def test_indistinguishable_pair():
    helpers.assert_family_distinguishes(
        model_class=truncated_rayleigh.TruncatedRayleigh,
        targets=(_rayleigh(ch.Tensor([0.5])), _rayleigh(ch.Tensor([2.0]))),
        phi=lambda z: (z[:, 0] > 1.0) & (z[:, 0] < 1.2),
        dims=1,
        parameter="scale_",
        empirical=helpers.untruncated_mle(truncated_rayleigh.TruncatedRayleigh, 1),
        rebuild=_rayleigh,
    )


@pytest.mark.slow
def test_twenty_dimensions_small_mass():
    scale = ch.linspace(0.5, 2.0, 20)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        _rayleigh(scale),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 400.0),
    )
    _recovers(dims=20, scale=scale, phi=calibrated.phi)
