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
            "verbose": True,
        },
    )


def test_two_dimensions():
    _recovers(
        dims=2,
        rate=ch.Tensor([1.0, 2.0]),
        phi=lambda z: z[:, 0] > 0.25,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 1,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
            "verbose": True,
        },
    )


def test_twenty_dimensions():
    _recovers(
        dims=20,
        rate=2 * ch.ones(20),
        phi=lambda z: z[:, 0] > 1,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
            "verbose": True,
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
