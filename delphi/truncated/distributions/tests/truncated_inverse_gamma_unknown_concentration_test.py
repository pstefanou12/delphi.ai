# Author: pstefanou12@
"""Tests for the truncated inverse gamma distribution with unknown shape."""

import functools
from typing import Any, Optional

import pytest
import torch as ch
import torch.distributions

import delphi.oracle
from delphi.truncated.distributions import (
    truncated_inverse_gamma_unknown_concentration as truncated_unknown_shape,
)
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
        model_class=truncated_unknown_shape.TruncatedInverseGammaUnknownConcentration,
        target=torch.distributions.InverseGamma(concentration, rate),
        phi=phi,
        dims=dims,
        parameter=("concentration_", "rate_"),
        empirical=helpers.untruncated_mle(
            truncated_unknown_shape.TruncatedInverseGammaUnknownConcentration, dims
        ),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        known=known,
        **unknown,
    )


# Unknown truncation needs degree 6 here; over seeds 0,1,2 it meets the shared
# bounds.
@helpers.known_and_unknown(pytest.mark.long, ci=True)
def test_one_dimension(known):
    _recovers(
        dims=1,
        concentration=ch.tensor([3.0]),
        rate=ch.tensor([2.0]),
        phi=lambda z: z[:, 0] > 1.2,
        known=known,
        hyperparameters={"truncation_set_degree": 6},
    )


# Unknown truncation, degree 6: over seeds 0,1,2 the estimate's KL measured
# 0.008 to 0.118 (ema), 0.008 to 0.117 (avg).
@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions(known):
    concentration, rate = ch.tensor([3.0, 2.0]), ch.tensor([2.0, 0.5])
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
        unknown_tolerances={"ema": 0.24, "avg": 0.24},
        hyperparameters={"truncation_set_degree": 6},
    )


# Unknown truncation, degree 6: over seeds 0,1,2 the estimate's KL measured
# 0.121 to 0.248 (ema), 0.121 to 0.252 (avg).
@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_half_space(known):
    concentration, rate = ch.tensor([3.0, 2.0]), ch.tensor([2.0, 0.5])
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
        unknown_tolerances={"ema": 0.5, "avg": 0.51},
        hyperparameters={"truncation_set_degree": 6},
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
    _recovers(
        dims=10,
        concentration=ch.linspace(1.5, 5.0, 10),
        rate=ch.linspace(0.5, 2.0, 10),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


# Unknown truncation, degree 6: over seeds 0,1,2 the estimate's KL measured
# 0.040 to 1.291 (ema), 0.037 to 1.304 (avg).
@pytest.mark.long
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
        unknown_tolerances={"ema": 2.6, "avg": 2.7},
        hyperparameters={"truncation_set_degree": 6},
    )


# Unknown truncation, rho 0.5: over seeds 0,1,2 the estimate's KL measured
# 0.181 to 0.253 (ema), 0.181 to 0.254 (avg); the set error 0.118 to 0.133.
@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_sum_threshold(known):
    concentration, rate = ch.linspace(1.5, 5.0, 10), ch.linspace(0.5, 2.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.InverseGamma(concentration, rate),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 500.0),
    )
    _recovers(
        dims=10,
        concentration=concentration,
        rate=rate,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.51, "avg": 0.51},
        set_tolerance=0.27,
        hyperparameters={"truncation_set_rho": 0.5},
    )


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the learned window is too coarse to separate the pair",
    )
)
def test_indistinguishable_pair(known):
    helpers.assert_family_distinguishes(
        model_class=truncated_unknown_shape.TruncatedInverseGammaUnknownConcentration,
        targets=(
            torch.distributions.InverseGamma(ch.tensor([2.0]), ch.tensor([1.0])),
            torch.distributions.InverseGamma(ch.tensor([6.0]), ch.tensor([2.6])),
        ),
        phi=lambda z: (z[:, 0] > 0.2) & (z[:, 0] < 0.6),
        dims=1,
        parameter=("concentration_", "rate_"),
        empirical=helpers.untruncated_mle(
            truncated_unknown_shape.TruncatedInverseGammaUnknownConcentration, 1
        ),
        known=known,
    )


# Unknown truncation: over seeds 0,1 the estimate's KL measured 0.212 to 0.234
# (ema), 0.212 to 0.234 (avg).
@pytest.mark.slow
@helpers.KNOWN_AND_UNKNOWN
def test_twenty_dimensions_small_mass(known):
    concentration, rate = ch.linspace(1.5, 5.0, 20), ch.linspace(0.5, 2.0, 20)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.InverseGamma(concentration, rate),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 1000.0),
    )
    _recovers(
        dims=20,
        concentration=concentration,
        rate=rate,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.47, "avg": 0.47},
    )
