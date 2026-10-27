# Author: pstefanou12@
"""Tests for the truncated beta prime distribution."""

import functools
from typing import Optional

import pytest
import torch as ch

from delphi.exponential_family import beta_prime
import delphi.oracle
from delphi.truncated.distributions import truncated_beta_prime
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _beta_prime(
    concentration1: ch.Tensor, concentration0: ch.Tensor
) -> beta_prime.BetaPrime:
    return beta_prime.BetaPrime(
        beta_prime.BetaPrime.to_natural(ch.cat([concentration1, concentration0])),
        concentration1.numel(),
    )


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dims: int,
    concentration1: ch.Tensor,
    concentration0: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_beta_prime.TruncatedBetaPrime,
        target=_beta_prime(concentration1, concentration0),
        phi=phi,
        dims=dims,
        parameter=("concentration1_", "concentration0_"),
        empirical=helpers.untruncated_mle(
            truncated_beta_prime.TruncatedBetaPrime, dims
        ),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        rebuild=_beta_prime,
    )


def test_one_dimension():
    _recovers(
        dims=1,
        concentration1=ch.Tensor([3.0]),
        concentration0=ch.Tensor([4.0]),
        phi=lambda z: z[:, 0] > 1.5,
    )


def test_two_dimensions():
    concentration1, concentration0 = ch.Tensor([3.0, 2.0]), ch.Tensor([4.0, 6.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        _beta_prime(concentration1, concentration0),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 50.0),
    )
    _recovers(
        dims=2,
        concentration1=concentration1,
        concentration0=concentration0,
        phi=calibrated.phi,
    )


def test_two_dimensions_random_half_space():
    concentration1, concentration0 = ch.Tensor([3.0, 2.0]), ch.Tensor([4.0, 6.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        _beta_prime(concentration1, concentration0),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(
        dims=2,
        concentration1=concentration1,
        concentration0=concentration0,
        phi=calibrated.phi,
    )


def test_one_dimension_identity():
    _recovers(
        dims=1,
        concentration1=ch.Tensor([3.0]),
        concentration0=ch.Tensor([4.0]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    _recovers(
        dims=10,
        concentration1=ch.linspace(1.5, 5.0, 10),
        concentration0=ch.linspace(3.0, 8.0, 10),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_one_dimension_small_mass():
    concentration1, concentration0 = ch.Tensor([3.0]), ch.Tensor([4.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        _beta_prime(concentration1, concentration0),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 100.0),
    )
    _recovers(
        dims=1,
        concentration1=concentration1,
        concentration0=concentration0,
        phi=calibrated.phi,
    )


def test_ten_dimensions_sum_threshold():
    concentration1 = ch.linspace(1.5, 5.0, 10)
    concentration0 = ch.linspace(3.0, 8.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        _beta_prime(concentration1, concentration0),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 500.0),
    )
    _recovers(
        dims=10,
        concentration1=concentration1,
        concentration0=concentration0,
        phi=calibrated.phi,
    )


def test_indistinguishable_pair():
    helpers.assert_family_distinguishes(
        model_class=truncated_beta_prime.TruncatedBetaPrime,
        targets=(
            _beta_prime(ch.Tensor([2.0]), ch.Tensor([3.0])),
            _beta_prime(ch.Tensor([6.0]), ch.Tensor([11.0])),
        ),
        phi=lambda z: (z[:, 0] > 0.3) & (z[:, 0] < 0.7),
        dims=1,
        parameter=("concentration1_", "concentration0_"),
        empirical=helpers.untruncated_mle(truncated_beta_prime.TruncatedBetaPrime, 1),
        rebuild=_beta_prime,
    )


@pytest.mark.slow
def test_twenty_dimensions_small_mass():
    concentration1 = ch.linspace(1.5, 5.0, 20)
    concentration0 = ch.linspace(3.0, 8.0, 20)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        _beta_prime(concentration1, concentration0),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 1000.0),
    )
    _recovers(
        dims=20,
        concentration1=concentration1,
        concentration0=concentration0,
        phi=calibrated.phi,
    )
