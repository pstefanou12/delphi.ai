# Author: pstefanou12@
"""Tests for the truncated inverse gamma (known shape) distribution."""

import functools
from typing import Any, Optional

import pytest
import torch as ch
import torch.distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_inverse_gamma
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
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
        model_class=truncated_inverse_gamma.TruncatedInverseGamma,
        target=torch.distributions.InverseGamma(concentration, rate),
        phi=phi,
        dims=dims,
        parameter="rate_",
        empirical=lambda s: concentration / (1.0 / s).mean(0),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        model_args=(concentration,),
        rebuild=functools.partial(torch.distributions.InverseGamma, concentration),
        known=known,
        **unknown,
    )


# Unknown truncation needs rho 0.3 here; over seeds 0,1,2 it meets the shared
# bounds.
@helpers.KNOWN_AND_UNKNOWN_CI
def test_one_dimension(known):
    _recovers(
        dims=1,
        concentration=ch.tensor([3.0]),
        rate=ch.tensor([2.0]),
        phi=lambda z: z[:, 0] > 1.2,
        known=known,
        hyperparameters={"truncation_set_rho": 0.3},
    )


# Unknown truncation needs rho 0.3 here; over seeds 0,1,2 it meets the shared
# bounds.
@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions(known):
    concentration, rate = ch.tensor([3.0, 4.0]), ch.tensor([2.0, 1.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.InverseGamma(concentration, rate),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 100.0),
    )
    _recovers(
        dims=2,
        concentration=concentration,
        rate=rate,
        phi=calibrated.phi,
        known=known,
        hyperparameters={"truncation_set_rho": 0.3},
    )


# Unknown truncation, rho 0.3: over seeds 0,1,2 the estimate's KL measured
# 0.043 to 0.129 (ema), 0.055 to 0.142 (avg).
@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_half_space(known):
    concentration, rate = ch.tensor([3.0, 4.0]), ch.tensor([2.0, 1.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        torch.distributions.InverseGamma(concentration, rate),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(
        dims=2,
        concentration=concentration,
        rate=rate,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.26, "avg": 0.29},
        hyperparameters={"truncation_set_rho": 0.3},
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_identity(known):
    _recovers(
        dims=1,
        concentration=ch.tensor([3.0]),
        rate=ch.tensor([2.0]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    concentration, rate = 3.0 * ch.ones(10), ch.linspace(0.5, 3.0, 10)
    _recovers(
        dims=10,
        concentration=concentration,
        rate=rate,
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


# Unknown truncation needs degree 2 and rho 0.5 here; over seeds 0,1,2 it
# meets the shared bounds.
@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_small_mass(known):
    concentration, rate = ch.tensor([3.0]), ch.tensor([2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.InverseGamma(concentration, rate),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 100.0),
    )
    _recovers(
        dims=1,
        concentration=concentration,
        rate=rate,
        phi=calibrated.phi,
        known=known,
        hyperparameters={"truncation_set_degree": 2, "truncation_set_rho": 0.5},
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_sum_threshold(known):
    concentration, rate = 3.0 * ch.ones(10), ch.linspace(0.5, 3.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.InverseGamma(concentration, rate),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 500.0),
    )
    _recovers(
        dims=10, concentration=concentration, rate=rate, phi=calibrated.phi, known=known
    )


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the learned window is too coarse to separate the pair",
    )
)
def test_indistinguishable_pair(known):
    concentration = ch.tensor([3.0])
    helpers.assert_family_distinguishes(
        model_class=truncated_inverse_gamma.TruncatedInverseGamma,
        targets=(
            torch.distributions.InverseGamma(concentration, ch.tensor([1.0])),
            torch.distributions.InverseGamma(concentration, ch.tensor([4.0])),
        ),
        phi=lambda z: (z[:, 0] > 0.6) & (z[:, 0] < 0.8),
        dims=1,
        parameter="rate_",
        empirical=lambda s: concentration / (1.0 / s).mean(0),
        model_args=(concentration,),
        rebuild=functools.partial(torch.distributions.InverseGamma, concentration),
        known=known,
    )
