# Author: pstefanou12@
"""Tests for the truncated binomial (known trials) distribution."""

import functools
from typing import Any, Optional

import pytest
import torch as ch
import torch.distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_binomial
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dims: int,
    total_count: ch.Tensor,
    probs: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: helpers.Known = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_binomial.TruncatedBinomial,
        target=torch.distributions.Binomial(total_count, probs),
        phi=phi,
        dims=dims,
        parameter="probs_",
        empirical=lambda s: s.mean(0) / total_count,
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        model_args=(total_count,),
        rebuild=functools.partial(torch.distributions.Binomial, total_count),
        known=known,
        **unknown,
    )


# Unknown truncation needs degree 2 and rho 0.5 here; over seeds 0,1,2 it
# meets the shared bounds.
@helpers.KNOWN_AND_UNKNOWN_CI
def test_one_dimension(known):
    _recovers(
        dims=1,
        total_count=ch.tensor([10.0]),
        probs=ch.tensor([0.3]),
        phi=lambda z: z[:, 0] > 4,
        known=known,
        hyperparameters={"truncation_set_degree": 2, "truncation_set_rho": 0.5},
    )


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the estimate's KL reaches 0.95 against the naive fit's 1.24",
    )
)
def test_two_dimensions(known):
    total_count, probs = ch.tensor([10.0, 20.0]), ch.tensor([0.3, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.Binomial(total_count, probs),
        target_alpha=(0.10, 0.40),
        bounds=(-1.0, 60.0),
    )
    _recovers(
        dims=2, total_count=total_count, probs=probs, phi=calibrated.phi, known=known
    )


# Unknown truncation needs degree 6 here; over seeds 0,1,2 it meets the shared
# bounds.
@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_half_space(known):
    total_count, probs = ch.tensor([10.0, 20.0]), ch.tensor([0.3, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        torch.distributions.Binomial(total_count, probs),
        target_alpha=(0.10, 0.40),
        bounds=(-50.0, 50.0),
    )
    _recovers(
        dims=2,
        total_count=total_count,
        probs=probs,
        phi=calibrated.phi,
        known=known,
        hyperparameters={"truncation_set_degree": 6},
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_identity(known):
    _recovers(
        dims=1,
        total_count=ch.tensor([10.0]),
        probs=ch.tensor([0.3]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    total_count, probs = 10.0 * ch.ones(10), ch.linspace(0.2, 0.6, 10)
    _recovers(
        dims=10,
        total_count=total_count,
        probs=probs,
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


# Unknown truncation is flaky here: nearly all the mass is on z = 7, so a
# learned set that keeps one point leaves the objective flat. Over seeds 0
# to 4 at degree 6, three recover, one reaches KL 0.64, and one CI run hit a
# flat Hessian; degree 4 or 8 and rho 0.3 or 0.5 do no better.
@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=False,
        raises=(AssertionError, ValueError),
        reason="unknown truncation: the learned set can keep a single point",
    ),
)
def test_one_dimension_small_mass(known):
    total_count, probs = ch.tensor([10.0]), ch.tensor([0.3])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.Binomial(total_count, probs),
        target_alpha=(0.005, 0.03),
        bounds=(-1.0, 60.0),
    )
    _recovers(
        dims=1,
        total_count=total_count,
        probs=probs,
        phi=calibrated.phi,
        known=known,
        hyperparameters={"truncation_set_degree": 6},
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_sum_threshold(known):
    total_count, probs = 10.0 * ch.ones(10), ch.linspace(0.2, 0.6, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.Binomial(total_count, probs),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 200.0),
    )
    _recovers(
        dims=10, total_count=total_count, probs=probs, phi=calibrated.phi, known=known
    )


# Unknown truncation needs degree 6 here; over seeds 0,1,2 it meets the shared
# bounds.
@helpers.KNOWN_AND_UNKNOWN
def test_indistinguishable_pair(known):
    total_count = ch.tensor([20.0])
    helpers.assert_family_distinguishes(
        model_class=truncated_binomial.TruncatedBinomial,
        targets=(
            torch.distributions.Binomial(total_count, ch.tensor([0.35])),
            torch.distributions.Binomial(total_count, ch.tensor([0.5])),
        ),
        phi=lambda z: (z[:, 0] > 7.5) & (z[:, 0] < 9.5),
        dims=1,
        parameter="probs_",
        empirical=lambda s: s.mean(0) / total_count,
        model_args=(total_count,),
        rebuild=functools.partial(torch.distributions.Binomial, total_count),
        known=known,
        hyperparameters={"truncation_set_degree": 6},
    )
