# Author: pstefanou12@
"""Tests for the truncated gamma (known concentration) distribution."""

import functools
from typing import Any, Optional

import pytest
import torch as ch
import torch.distributions

import delphi.oracle
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
    known: bool = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_gamma.TruncatedGamma,
        target=torch.distributions.Gamma(concentration, rate),
        phi=phi,
        dims=dims,
        parameter="rate_",
        empirical=lambda s: concentration / s.mean(0),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        model_args=(concentration,),
        rebuild=functools.partial(torch.distributions.Gamma, concentration),
        known=known,
        **unknown,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension(known):
    _recovers(
        dims=1,
        concentration=ch.tensor([2.0]),
        rate=ch.tensor([1.0]),
        phi=lambda z: z[:, 0] > 2.0,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions(known):
    concentration, rate = ch.tensor([2.0, 3.0]), ch.tensor([1.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.Gamma(concentration, rate),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 50.0),
    )
    _recovers(
        dims=2,
        concentration=concentration,
        rate=rate,
        phi=calibrated.phi,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_half_space(known):
    concentration, rate = ch.tensor([2.0, 3.0]), ch.tensor([1.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        torch.distributions.Gamma(concentration, rate),
        bounds=(-25.0, 25.0),
    )
    _recovers(
        dims=2,
        concentration=concentration,
        rate=rate,
        phi=calibrated.phi,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN_CI
def test_one_dimension_identity(known):
    _recovers(
        dims=1,
        concentration=ch.tensor([2.0]),
        rate=ch.tensor([1.0]),
        phi=delphi.oracle.Identity(),
        hyperparameters={"iterations": 1500},
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    _recovers(
        dims=10,
        concentration=ch.linspace(1.5, 3.0, 10),
        rate=ch.linspace(0.5, 2.0, 10),
        phi=delphi.oracle.Identity(),
        hyperparameters={"iterations": 1500},
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


# Unknown truncation needs degree 6 here; over seeds 0,1,2 it meets the shared
# bounds.
@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_small_mass(known):
    concentration, rate = ch.tensor([2.0]), ch.tensor([1.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.Gamma(concentration, rate),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 50.0),
    )
    _recovers(
        dims=1,
        concentration=concentration,
        rate=rate,
        phi=calibrated.phi,
        known=known,
        hyperparameters={"truncation_set_degree": 6},
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_sum_threshold(known):
    concentration, rate = ch.linspace(1.5, 3.0, 10), ch.linspace(0.5, 2.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.Gamma(concentration, rate),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 200.0),
    )
    _recovers(
        dims=10,
        concentration=concentration,
        rate=rate,
        phi=calibrated.phi,
        known=known,
    )


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the learned window is too coarse to separate the pair",
    )
)
def test_indistinguishable_pair(known):
    helpers.assert_family_distinguishes(
        model_class=truncated_gamma.TruncatedGamma,
        targets=(
            torch.distributions.Gamma(ch.tensor([2.0]), ch.tensor([0.5])),
            torch.distributions.Gamma(ch.tensor([2.0]), ch.tensor([1.5])),
        ),
        phi=lambda z: (z[:, 0] > 1.0) & (z[:, 0] < 1.5),
        dims=1,
        parameter="rate_",
        empirical=lambda s: 2.0 / s.mean(0),
        model_args=(ch.tensor([2.0]),),
        rebuild=functools.partial(torch.distributions.Gamma, ch.tensor([2.0])),
        known=known,
    )


# Seeds 0-2: KL at most 0.0017 (ema), 0.0019 (avg); bounds are twice that.
@pytest.mark.slow
def test_twenty_dimensions_sum_threshold():
    concentration, rate = ch.linspace(1.5, 3.0, 20), ch.linspace(0.5, 2.0, 20)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.Gamma(concentration, rate),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 400.0),
    )
    _recovers(
        dims=20,
        concentration=concentration,
        rate=rate,
        phi=calibrated.phi,
        tolerances={"ema": 0.0035, "avg": 0.0038},
        max_coordinate=0.00072,
    )
