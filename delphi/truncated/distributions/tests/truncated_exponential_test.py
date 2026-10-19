# Author: pstefanou12@
"""Tests for the truncated exponential distribution."""

import functools

import pytest
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
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_exponential.TruncatedExponential,
        target=distributions.Exponential(rate),
        phi=phi,
        dims=dims,
        parameter="lambda_",
        empirical=lambda s: 1.0 / s.mean(0),
        hyperparameters=hyperparameters,
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
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Exponential(rate),
        target_alpha=(0.15, 0.25),
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


def test_twenty_dimensions():
    rate = 2 * ch.ones(20)
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Exponential(rate),
        target_alpha=(0.15, 0.25),
    )
    _recovers(
        dims=20,
        rate=rate,
        phi=calibrated.phi,
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
        rate=ch.linspace(1.0, 2.0, 10),
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
    rate = ch.linspace(1.0, 2.0, 10)
    # One scalar condition, so sampling stays cheap, but it biases every
    # coordinate instead of one.
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        distributions.Exponential(rate),
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
        distributions.Exponential(rate),
        target_alpha=(0.008, 0.02),
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


def test_ten_dimensions_norm_threshold():
    rate = ch.linspace(1.0, 2.0, 10)
    # Couples every coordinate like the sum does, and also excludes the mode.
    calibrated = truncation_sets.calibrate(
        truncation_sets.norm_threshold,
        distributions.Exponential(rate),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 100.0),
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


def test_unit_alpha_is_rejected():
    with pytest.raises(ValueError, match="nothing was truncated"):
        truncated_exponential.TruncatedExponential(
            {"iterations": 10, "trials": 1, "batch_size": 10},
            truncation_sets.identity,
            1,
            alpha=1.0,
        )
