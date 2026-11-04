# Author: pstefanou12@
"""Tests for the truncated continuous Bernoulli distribution.

There is no small-mass test: a 1% tail is a sliver near 0 or 1 where the
density is nearly flat, and the Cramer-Rao bound on the logit is 4.44 there,
against a true 1.39.
"""

import functools
from typing import Optional

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
    )


def test_one_dimension():
    _recovers(dims=1, probs=ch.tensor([0.8]), phi=lambda z: z[:, 0] < 0.5)


def test_two_dimensions():
    probs = ch.tensor([0.8, 0.3])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.ContinuousBernoulli(probs),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 1.0),
    )
    _recovers(dims=2, probs=probs, phi=calibrated.phi)


def test_two_dimensions_random_half_space():
    probs = ch.tensor([0.8, 0.3])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        torch.distributions.ContinuousBernoulli(probs),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(dims=2, probs=probs, phi=calibrated.phi)


def test_one_dimension_identity():
    _recovers(
        dims=1,
        probs=ch.tensor([0.8]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    probs = ch.linspace(0.2, 0.8, 10)
    _recovers(
        dims=10,
        probs=probs,
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_sum_threshold():
    probs = ch.linspace(0.2, 0.8, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.ContinuousBernoulli(probs),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 10.0),
    )
    _recovers(dims=10, probs=probs, phi=calibrated.phi)


def test_indistinguishable_pair():
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
    )


def test_unknown_truncation():
    # Over seeds 0-4 the estimate's divergence measured 0.0000 to 0.0022 and the
    # learned set's error 0.0000 to 0.0003.
    helpers.assert_family_learns_set(
        model_class=truncated_continuous_bernoulli.TruncatedContinuousBernoulli,
        target=torch.distributions.ContinuousBernoulli(ch.tensor([0.8])),
        phi=lambda z: z[:, 0] < 0.5,
        dims=1,
        parameter="probs_",
        empirical=helpers.untruncated_mle(
            truncated_continuous_bernoulli.TruncatedContinuousBernoulli, 1
        ),
        tolerance=0.005,
        set_tolerance=0.005,
    )
