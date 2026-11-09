# Author: pstefanou12@
"""Tests for the truncated Gumbel (known scale) distribution."""

import functools
from typing import Any, Optional

import pytest
import torch as ch
import torch.distributions

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
    known: bool = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_gumbel.TruncatedGumbel,
        target=torch.distributions.Gumbel(loc, scale),
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
        rebuild=lambda fitted: torch.distributions.Gumbel(fitted, scale),
        known=known,
        **unknown,
    )


@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension(known):
    _recovers(
        dims=1,
        loc=ch.tensor([1.0]),
        scale=ch.tensor([2.0]),
        phi=lambda z: z[:, 0] > 4.0,
        known=known,
    )


# Unknown truncation: over seeds 0,1,2 the estimate's KL measured 0.151 to
# 0.222 (ema), 0.150 to 0.227 (avg).
@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions(known):
    loc, scale = ch.tensor([1.0, -1.0]), ch.tensor([2.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.Gumbel(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(
        dims=2,
        loc=loc,
        scale=scale,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.45, "avg": 0.46},
    )


# Unknown truncation, rho 0.5: over seeds 0,1,2 the estimate's KL measured
# 0.022 to 0.380 (ema), 0.019 to 0.381 (avg).
@helpers.known_and_unknown(pytest.mark.long, ci=True)
def test_two_dimensions_random_half_space(known):
    loc, scale = ch.tensor([1.0, -1.0]), ch.tensor([2.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        torch.distributions.Gumbel(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(
        dims=2,
        loc=loc,
        scale=scale,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.77, "avg": 0.77},
        hyperparameters={"truncation_set_rho": 0.5},
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_identity(known):
    _recovers(
        dims=1,
        loc=ch.tensor([1.0]),
        scale=ch.tensor([2.0]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    _recovers(
        dims=10,
        loc=ch.linspace(-2.0, 2.0, 10),
        scale=ch.linspace(0.5, 2.0, 10),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


# Unknown truncation, degree 6: over seeds 0,1,2 the estimate's KL measured
# 0.035 to 0.743 (ema), 0.035 to 0.744 (avg).
@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_small_mass(known):
    loc, scale = ch.tensor([1.0]), ch.tensor([2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0, below=True),
        torch.distributions.Gumbel(loc, scale),
        target_alpha=(0.008, 0.02),
        bounds=(-50.0, 50.0),
    )
    _recovers(
        dims=1,
        loc=loc,
        scale=scale,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 1.5, "avg": 1.5},
        hyperparameters={"truncation_set_degree": 6},
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_sum_threshold(known):
    loc, scale = ch.linspace(-2.0, 2.0, 10), ch.linspace(0.5, 2.0, 10)
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.sum_threshold, below=True),
        torch.distributions.Gumbel(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(-200.0, 200.0),
    )
    _recovers(dims=10, loc=loc, scale=scale, phi=calibrated.phi, known=known)


@pytest.mark.long
@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the learned window is too coarse to separate the pair",
    )
)
def test_indistinguishable_pair(known):
    scale = ch.tensor([1.0])
    helpers.assert_family_distinguishes(
        model_class=truncated_gumbel.TruncatedGumbel,
        targets=(
            torch.distributions.Gumbel(ch.tensor([0.0]), scale),
            torch.distributions.Gumbel(ch.tensor([1.0]), scale),
        ),
        phi=lambda z: (z[:, 0] > 0.3) & (z[:, 0] < 0.7),
        dims=1,
        parameter="loc_",
        empirical=helpers.untruncated_mle(truncated_gumbel.TruncatedGumbel, 1, scale),
        model_args=(scale,),
        rebuild=lambda fitted: torch.distributions.Gumbel(fitted, scale),
        known=known,
    )


# Unknown truncation: over seeds 0,1 the estimate's KL measured 0.596 to 0.630
# (ema), 0.598 to 0.631 (avg).
@pytest.mark.slow
@helpers.KNOWN_AND_UNKNOWN
def test_twenty_dimensions_small_mass(known):
    loc, scale = ch.linspace(-2.0, 2.0, 20), ch.linspace(0.5, 2.0, 20)
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.sum_threshold, below=True),
        torch.distributions.Gumbel(loc, scale),
        target_alpha=(0.008, 0.02),
        bounds=(-400.0, 400.0),
    )
    _recovers(
        dims=20,
        loc=loc,
        scale=scale,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 1.3, "avg": 1.3},
    )
