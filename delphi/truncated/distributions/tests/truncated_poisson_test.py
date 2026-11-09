# Author: pstefanou12@
"""Tests for the truncated Poisson distribution."""

import functools
from typing import Any, Optional

import pytest
import torch as ch
import torch.distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_poisson
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(
    rate: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    dataset_size: int = 10000,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: bool = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_poisson.TruncatedPoisson,
        target=torch.distributions.Poisson(rate),
        phi=phi,
        parameter="rate_",
        empirical=lambda s: s.mean(0),
        hyperparameters=hyperparameters,
        dataset_size=dataset_size,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        known=known,
        **unknown,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension(known):
    _recovers(
        rate=ch.tensor([1.0]),
        phi=lambda z: z > 1,
        hyperparameters={"batch_size": 1},
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions(known):
    rate = ch.tensor([1.0, 2.0])
    # A discrete family reaches only the masses its integer thresholds give,
    # so the band is wider than the continuous families use.
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.Poisson(rate),
        target_alpha=(0.10, 0.30),
        bounds=(0.0, 30.0),
    )
    _recovers(
        rate=rate,
        phi=calibrated.phi,
        hyperparameters={"batch_size": 1},
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_twenty_dimensions(known):
    rate = 2 * ch.ones(20)
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.Poisson(rate),
        target_alpha=(0.10, 0.30),
        bounds=(0.0, 30.0),
    )
    _recovers(
        rate=rate,
        phi=calibrated.phi,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_triangle(known):
    _recovers(
        rate=ch.tensor([1.0, 2.0]),
        phi=truncation_sets.triangle([[-0.5, -0.5], [3.0, 0.5], [0.5, 3.0]]),
        hyperparameters={"batch_size": 1},
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN_CI
def test_two_dimensions_random_half_space(known):
    rate = ch.tensor([1.0, 2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        torch.distributions.Poisson(rate),
        bounds=(-25.0, 25.0),
    )
    _recovers(
        rate=rate,
        phi=calibrated.phi,
        hyperparameters={"batch_size": 1},
        known=known,
    )


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the estimate's KL 0.60 is over half the naive fit's 0.91",
    )
)
def test_five_dimensions_random_polytope(known):
    rate = 2 * ch.ones(5)
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_polytope, 0, 5, 4),
        torch.distributions.Poisson(rate),
        bounds=(-25.0, 25.0),
    )
    _recovers(
        rate=rate,
        phi=calibrated.phi,
        dataset_size=5000,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_identity(known):
    _recovers(
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
        rate=ch.linspace(1.0, 3.0, 10),
        phi=delphi.oracle.Identity(),
        hyperparameters={"iterations": 1500},
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_sum_threshold(known):
    rate = ch.linspace(1.0, 3.0, 10)
    # One scalar condition, so sampling stays cheap, but it biases every
    # coordinate instead of one.
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.Poisson(rate),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 200.0),
    )
    _recovers(
        rate=rate,
        phi=calibrated.phi,
        hyperparameters={"lr": 1e-1},
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_small_mass(known):
    rate = ch.tensor([1.0])
    # The regime both papers state their guarantees in; nothing else in the
    # suite goes near it.
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.Poisson(rate),
        target_alpha=(0.005, 0.03),
        bounds=(0.0, 50.0),
    )
    _recovers(
        rate=rate,
        phi=calibrated.phi,
        known=known,
    )


# Unknown truncation needs degree 6 here; over seeds 0,1,2 it meets the shared
# bounds.
@helpers.KNOWN_AND_UNKNOWN
def test_indistinguishable_pair(known):
    # Truncated to a narrow window the two samples agree to within sampling
    # error, so only an estimator that knows the window can tell them apart.
    helpers.assert_family_distinguishes(
        model_class=truncated_poisson.TruncatedPoisson,
        targets=(
            torch.distributions.Poisson(ch.tensor([2.0])),
            torch.distributions.Poisson(ch.tensor([5.0])),
        ),
        phi=lambda z: (z[:, 0] >= 3) & (z[:, 0] <= 4),
        parameter="rate_",
        empirical=lambda s: s.mean(0),
        known=known,
        hyperparameters={"truncation_set_degree": 6},
    )


def test_phases_widen_a_budget_too_small_for_the_fit():
    target = torch.distributions.Poisson(ch.tensor([3.0]))

    def phi(z):
        return z[:, 0] >= 4

    ch.manual_seed(0)
    sample = helpers.generate_truncated_distribution_dataset(target, phi, 10000)
    # Without alpha the budget is min_radius, here far below the log(1 / alpha)
    # of 1.04 the fit needs, so one phase is pinned near the naive start.
    hyperparameters = {"min_radius": 0.05, "rate": 2.0, "iterations": 500}
    fits = {}
    for label, cap in (("one phase", {"max_phases": 1}), ("phases", {})):
        ch.manual_seed(1)
        model = truncated_poisson.TruncatedPoisson(
            phi, hyperparameters=hyperparameters | cap
        )
        fits[label] = model.fit(sample.S)

    def divergence(model):
        fitted = torch.distributions.Poisson(model.avg_rate_)
        return float(torch.distributions.kl.kl_divergence(fitted, target).sum())

    assert divergence(fits["one phase"]) > 0.1
    assert divergence(fits["phases"]) < 0.01
    assert len(fits["phases"].procedure.phases) > 1
    # It ends on its own, once a budget no longer binds.
    assert fits["phases"].procedure.stop_reason == "budget_not_binding"
