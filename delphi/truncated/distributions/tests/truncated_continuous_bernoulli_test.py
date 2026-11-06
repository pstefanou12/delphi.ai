# Author: pstefanou12@
"""Tests for the truncated continuous Bernoulli distribution.

There is no small-mass test: a 1% tail is a sliver near 0 or 1 where the
density is nearly flat, and the Cramer-Rao bound on the logit is 4.44 there,
against a true 1.39.
"""

import functools
from typing import Any, Optional

import pytest
import torch as ch
import torch.distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_continuous_bernoulli
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dims: int,
    probs: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: bool = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_continuous_bernoulli.TruncatedContinuousBernoulli,
        target=torch.distributions.ContinuousBernoulli(probs),
        phi=phi,
        dims=dims,
        parameter="probs_",
        empirical=helpers.untruncated_mle(
            truncated_continuous_bernoulli.TruncatedContinuousBernoulli, dims
        ),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        known=known,
        **unknown,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension(known):
    _recovers(dims=1, probs=ch.tensor([0.8]), phi=lambda z: z[:, 0] < 0.5, known=known)


# Unknown truncation, rho 0.3: over seeds 0,1,2 the estimate's KL measured
# 0.036 to 0.145 (ema), 0.043 to 0.152 (avg).
@helpers.KNOWN_AND_UNKNOWN_CI
def test_two_dimensions(known):
    probs = ch.tensor([0.8, 0.3])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.ContinuousBernoulli(probs),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 1.0),
    )
    _recovers(
        dims=2,
        probs=probs,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.29, "avg": 0.31},
        hyperparameters={"truncation_set_rho": 0.3},
    )


# Unknown truncation: 0.027 to 0.083 over seeds 0 to 4 here, and 0.103 in CI.
@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_half_space(known):
    probs = ch.tensor([0.8, 0.3])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        torch.distributions.ContinuousBernoulli(probs),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(
        dims=2,
        probs=probs,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.17, "avg": 0.18},
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_identity(known):
    _recovers(
        dims=1,
        probs=ch.tensor([0.8]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


# Unknown truncation: over seeds 0,1,2 the set error 0.149 to 0.154.
@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    probs = ch.linspace(0.2, 0.8, 10)
    _recovers(
        dims=10,
        probs=probs,
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
        set_tolerance=0.31,
    )


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the estimate's KL reaches 0.20 against the shared 0.15 on seed 0, and on another seed of three the Fisher schedule's Hessian estimate turns non-positive",
    )
)
def test_ten_dimensions_sum_threshold(known):
    probs = ch.linspace(0.2, 0.8, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.ContinuousBernoulli(probs),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 10.0),
    )
    _recovers(dims=10, probs=probs, phi=calibrated.phi, known=known)


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the learned window is too coarse to separate the pair",
    )
)
def test_indistinguishable_pair(known):
    helpers.assert_family_distinguishes(
        model_class=truncated_continuous_bernoulli.TruncatedContinuousBernoulli,
        targets=(
            torch.distributions.ContinuousBernoulli(ch.tensor([0.2])),
            torch.distributions.ContinuousBernoulli(ch.tensor([0.8])),
        ),
        phi=lambda z: (z[:, 0] > 0.3) & (z[:, 0] < 0.7),
        dims=1,
        parameter="probs_",
        empirical=helpers.untruncated_mle(
            truncated_continuous_bernoulli.TruncatedContinuousBernoulli, 1
        ),
        known=known,
    )
