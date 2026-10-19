# Author: pstefanou12@
"""Tests for the truncated Poisson distribution."""

import functools

import torch as ch
from torch import distributions

from delphi.truncated.distributions import truncated_poisson
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(
    dims: int,
    rate: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: dict,
    dataset_size: int = 10000,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_poisson.TruncatedPoisson,
        target=distributions.Poisson(rate),
        phi=phi,
        dims=dims,
        parameter="lambda_",
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
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 1,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
        },
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
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 1,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
        },
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
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
        },
    )


def test_two_dimensions_triangle():
    _recovers(
        dims=2,
        rate=ch.Tensor([1.0, 2.0]),
        phi=truncation_sets.triangle([[-0.5, -0.5], [3.0, 0.5], [0.5, 3.0]]),
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 1,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
        },
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
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 1,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
        },
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
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
        },
        dataset_size=5000,
    )


def test_one_dimension_identity():
    # phi admits everything, so the procedure should reduce to the
    # untruncated MLE and the naive estimate is the right answer.
    _recovers(
        dims=1,
        rate=ch.Tensor([1.0]),
        phi=truncation_sets.identity,
        hyperparameters={
            "iterations": 1500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
        },
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    _recovers(
        dims=10,
        rate=ch.linspace(1.0, 3.0, 10),
        phi=truncation_sets.identity,
        hyperparameters={
            "iterations": 1500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
        },
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
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-1,
        },
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
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
        },
    )
