# Author: pstefanou12@
"""Tests for the truncated exponential distribution."""

import functools

import torch as ch
from torch import distributions

from delphi.truncated.distributions import truncated_exponential
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(
    dims: int,
    rate: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: dict,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_exponential.TruncatedExponential,
        target=distributions.Exponential(rate),
        phi=phi,
        dims=dims,
        parameter="lambda_",
        empirical=lambda s: 1.0 / s.mean(0),
        hyperparameters=hyperparameters,
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
    _recovers(
        dims=2,
        rate=ch.Tensor([1.0, 2.0]),
        phi=lambda z: z[:, 0] > 0.5,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
        },
    )


def test_twenty_dimensions():
    _recovers(
        dims=20,
        rate=2 * ch.ones(20),
        phi=lambda z: z[:, 0] > 0.25,
        hyperparameters={
            "iterations": 1500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-1,
        },
    )


def test_two_dimensions_random_half_space():
    rate = ch.Tensor([1.0, 2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 1, 2),
        distributions.Exponential(rate),
        bounds=(-25.0, 25.0),
    )
    _recovers(
        dims=2,
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


def test_two_dimensions_random_ellipsoid():
    rate = ch.Tensor([1.0, 2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_ellipsoid, 0, 2),
        distributions.Exponential(rate),
        bounds=(1e-4, 1e4),
    )
    # A bounded set leaves the empirical rate far from the truth, so the lr
    # the axis-aligned tests use still measures 0.3173 after 2500 steps.
    _recovers(
        dims=2,
        rate=rate,
        phi=calibrated.phi,
        hyperparameters={
            "iterations": 5000,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-1,
        },
    )
