# Author: pstefanou12@
"""Tests for the truncated Poisson distribution."""

import functools
from typing import Optional

import torch as ch
from torch import distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_poisson
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(
    dims: int,
    rate: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    dataset_size: int = 10000,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_poisson.TruncatedPoisson,
        target=distributions.Poisson(rate),
        phi=phi,
        dims=dims,
        parameter="rate_",
        empirical=lambda s: s.mean(0),
        hyperparameters=hyperparameters,
        dataset_size=dataset_size,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
    )


def test_one_dimension():
    _recovers(
        dims=1,
        rate=ch.Tensor([1.0]),
        phi=lambda z: z > 1,
        hyperparameters={"batch_size": 1},
    )


def test_two_dimensions():
    rate = ch.Tensor([1.0, 2.0])
    # A discrete family reaches only the masses its integer thresholds give,
    # so the band is wider than the continuous families use.
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Poisson(rate),
        target_alpha=(0.10, 0.30),
        bounds=(0.0, 30.0),
    )
    _recovers(
        dims=2,
        rate=rate,
        phi=calibrated.phi,
        hyperparameters={"batch_size": 1},
    )


def test_twenty_dimensions():
    rate = 2 * ch.ones(20)
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Poisson(rate),
        target_alpha=(0.10, 0.30),
        bounds=(0.0, 30.0),
    )
    _recovers(
        dims=20,
        rate=rate,
        phi=calibrated.phi,
    )


def test_two_dimensions_triangle():
    _recovers(
        dims=2,
        rate=ch.Tensor([1.0, 2.0]),
        phi=truncation_sets.triangle([[-0.5, -0.5], [3.0, 0.5], [0.5, 3.0]]),
        hyperparameters={"batch_size": 1},
    )


def test_two_dimensions_random_half_space():
    rate = ch.Tensor([1.0, 2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        distributions.Poisson(rate),
        bounds=(-25.0, 25.0),
    )
    _recovers(
        dims=2,
        rate=rate,
        phi=calibrated.phi,
        hyperparameters={"batch_size": 1},
    )


def test_five_dimensions_random_polytope():
    rate = 2 * ch.ones(5)
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_polytope, 0, 5, 4),
        distributions.Poisson(rate),
        bounds=(-25.0, 25.0),
    )
    _recovers(
        dims=5,
        rate=rate,
        phi=calibrated.phi,
        dataset_size=5000,
    )


def test_one_dimension_identity():
    _recovers(
        dims=1,
        rate=ch.Tensor([1.0]),
        phi=delphi.oracle.Identity(),
        hyperparameters={"iterations": 1500},
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    _recovers(
        dims=10,
        rate=ch.linspace(1.0, 3.0, 10),
        phi=delphi.oracle.Identity(),
        hyperparameters={"iterations": 1500},
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_sum_threshold():
    rate = ch.linspace(1.0, 3.0, 10)
    # One scalar condition, so sampling stays cheap, but it biases every
    # coordinate instead of one.
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        distributions.Poisson(rate),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 200.0),
    )
    _recovers(
        dims=10,
        rate=rate,
        phi=calibrated.phi,
        hyperparameters={"lr": 1e-1},
    )


def test_one_dimension_small_mass():
    rate = ch.Tensor([1.0])
    # The regime both papers state their guarantees in; nothing else in the
    # suite goes near it.
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Poisson(rate),
        target_alpha=(0.005, 0.03),
        bounds=(0.0, 50.0),
    )
    _recovers(
        dims=1,
        rate=rate,
        phi=calibrated.phi,
    )


def test_indistinguishable_pair():
    # Truncated to a narrow window the two samples agree to within sampling
    # error, so only an estimator that knows the window can tell them apart.
    helpers.assert_family_distinguishes(
        model_class=truncated_poisson.TruncatedPoisson,
        targets=(
            distributions.Poisson(ch.Tensor([2.0])),
            distributions.Poisson(ch.Tensor([5.0])),
        ),
        phi=lambda z: (z[:, 0] >= 3) & (z[:, 0] <= 4),
        dims=1,
        parameter="rate_",
        empirical=lambda s: s.mean(0),
    )


def test_phases_widen_a_budget_too_small_for_the_fit():
    target = distributions.Poisson(ch.Tensor([3.0]))

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
        model = truncated_poisson.TruncatedPoisson(hyperparameters | cap, phi, 1)
        fits[label] = model.fit(sample.S)

    def divergence(model):
        fitted = distributions.Poisson(model.avg_rate_)
        return float(distributions.kl.kl_divergence(fitted, target).sum())

    assert divergence(fits["one phase"]) > 0.1
    assert divergence(fits["phases"]) < 0.01
    assert len(fits["phases"].procedure.phases) > 1
    # It ends on its own, once a budget no longer binds.
    assert fits["phases"].procedure.stop_reason == "budget_not_binding"
