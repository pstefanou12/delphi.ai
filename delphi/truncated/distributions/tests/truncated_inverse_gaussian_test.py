# Author: pstefanou12@
"""Tests for the truncated inverse Gaussian distribution."""

import functools
from typing import Optional

import pytest
import torch as ch

from delphi.exponential_family import inverse_gaussian
import delphi.oracle
from delphi.truncated.distributions import truncated_inverse_gaussian
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _wald(loc: ch.Tensor, concentration: ch.Tensor) -> inverse_gaussian.InverseGaussian:
    return inverse_gaussian.InverseGaussian(
        inverse_gaussian.InverseGaussian.to_natural(ch.cat([loc, concentration])),
        loc.numel(),
    )


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dims: int,
    loc: ch.Tensor,
    concentration: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_inverse_gaussian.TruncatedInverseGaussian,
        target=_wald(loc, concentration),
        phi=phi,
        dims=dims,
        parameter=("loc_", "concentration_"),
        empirical=helpers.untruncated_mle(
            truncated_inverse_gaussian.TruncatedInverseGaussian, dims
        ),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        rebuild=_wald,
    )


def test_one_dimension():
    _recovers(
        dims=1,
        loc=ch.Tensor([1.0]),
        concentration=ch.Tensor([2.0]),
        phi=lambda z: z[:, 0] > 1.5,
    )


def test_two_dimensions():
    loc, concentration = ch.Tensor([1.0, 2.0]), ch.Tensor([2.0, 4.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        _wald(loc, concentration),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 50.0),
    )
    _recovers(dims=2, loc=loc, concentration=concentration, phi=calibrated.phi)


def test_two_dimensions_random_half_space():
    loc, concentration = ch.Tensor([1.0, 2.0]), ch.Tensor([2.0, 4.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        _wald(loc, concentration),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(dims=2, loc=loc, concentration=concentration, phi=calibrated.phi)


def test_one_dimension_identity():
    _recovers(
        dims=1,
        loc=ch.Tensor([1.0]),
        concentration=ch.Tensor([2.0]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    _recovers(
        dims=10,
        loc=ch.linspace(0.5, 2.0, 10),
        concentration=ch.linspace(1.0, 5.0, 10),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_one_dimension_small_mass():
    loc, concentration = ch.Tensor([1.0]), ch.Tensor([2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        _wald(loc, concentration),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 50.0),
    )
    _recovers(dims=1, loc=loc, concentration=concentration, phi=calibrated.phi)


def test_ten_dimensions_sum_threshold():
    loc, concentration = ch.linspace(0.5, 2.0, 10), ch.linspace(1.0, 5.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        _wald(loc, concentration),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 200.0),
    )
    _recovers(dims=10, loc=loc, concentration=concentration, phi=calibrated.phi)


def test_indistinguishable_pair():
    helpers.assert_family_distinguishes(
        model_class=truncated_inverse_gaussian.TruncatedInverseGaussian,
        targets=(
            _wald(ch.Tensor([1.0]), ch.Tensor([3.0])),
            _wald(ch.Tensor([3.0]), ch.Tensor([1.0])),
        ),
        phi=lambda z: (z[:, 0] > 0.5) & (z[:, 0] < 1.5),
        dims=1,
        parameter=("loc_", "concentration_"),
        empirical=helpers.untruncated_mle(
            truncated_inverse_gaussian.TruncatedInverseGaussian, 1
        ),
        rebuild=_wald,
    )


@pytest.mark.slow
def test_twenty_dimensions_small_mass():
    loc, concentration = ch.linspace(0.5, 2.0, 20), ch.linspace(1.0, 5.0, 20)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        _wald(loc, concentration),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 400.0),
    )
    _recovers(dims=20, loc=loc, concentration=concentration, phi=calibrated.phi)
