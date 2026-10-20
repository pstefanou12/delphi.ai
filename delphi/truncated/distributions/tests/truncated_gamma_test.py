# Author: pstefanou12@
"""Tests for the truncated gamma (known concentration) distribution."""

import functools
from typing import Optional

import torch as ch
from torch import distributions

from delphi.truncated.distributions import truncated_gamma
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(
    dims: int,
    concentration: ch.Tensor,
    rate: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_gamma.TruncatedGamma,
        target=distributions.Gamma(concentration, rate),
        phi=phi,
        dims=dims,
        parameter="rate_",
        empirical=lambda s: concentration / s.mean(0),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        model_args=(concentration,),
        rebuild=functools.partial(distributions.Gamma, concentration),
    )


def test_one_dimension():
    _recovers(
        dims=1,
        concentration=ch.Tensor([2.0]),
        rate=ch.Tensor([1.0]),
        phi=lambda z: z[:, 0] > 2.0,
    )


def test_two_dimensions():
    concentration, rate = ch.Tensor([2.0, 3.0]), ch.Tensor([1.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Gamma(concentration, rate),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 50.0),
    )
    _recovers(
        dims=2,
        concentration=concentration,
        rate=rate,
        phi=calibrated.phi,
    )


def test_two_dimensions_random_half_space():
    concentration, rate = ch.Tensor([2.0, 3.0]), ch.Tensor([1.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        distributions.Gamma(concentration, rate),
        bounds=(-25.0, 25.0),
    )
    _recovers(
        dims=2,
        concentration=concentration,
        rate=rate,
        phi=calibrated.phi,
    )


def test_one_dimension_identity():
    _recovers(
        dims=1,
        concentration=ch.Tensor([2.0]),
        rate=ch.Tensor([1.0]),
        phi=truncation_sets.identity,
        hyperparameters={"iterations": 1500},
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    _recovers(
        dims=10,
        concentration=ch.linspace(1.5, 3.0, 10),
        rate=ch.linspace(0.5, 2.0, 10),
        phi=truncation_sets.identity,
        hyperparameters={"iterations": 1500},
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_one_dimension_small_mass():
    concentration, rate = ch.Tensor([2.0]), ch.Tensor([1.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Gamma(concentration, rate),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 50.0),
    )
    _recovers(
        dims=1,
        concentration=concentration,
        rate=rate,
        phi=calibrated.phi,
    )


def test_ten_dimensions_sum_threshold():
    concentration, rate = ch.linspace(1.5, 3.0, 10), ch.linspace(0.5, 2.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        distributions.Gamma(concentration, rate),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 200.0),
    )
    _recovers(
        dims=10,
        concentration=concentration,
        rate=rate,
        phi=calibrated.phi,
    )


def test_indistinguishable_pair():
    helpers.assert_family_distinguishes(
        model_class=truncated_gamma.TruncatedGamma,
        targets=(
            distributions.Gamma(ch.Tensor([2.0]), ch.Tensor([0.5])),
            distributions.Gamma(ch.Tensor([2.0]), ch.Tensor([1.5])),
        ),
        phi=lambda z: (z[:, 0] > 1.0) & (z[:, 0] < 1.5),
        dims=1,
        parameter="rate_",
        empirical=lambda s: 2.0 / s.mean(0),
        model_args=(ch.Tensor([2.0]),),
        rebuild=functools.partial(distributions.Gamma, ch.Tensor([2.0])),
    )
