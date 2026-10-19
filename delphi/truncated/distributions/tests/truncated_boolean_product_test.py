# Author: pstefanou12@
"""Tests for the truncated boolean product distribution."""

import functools

import torch as ch
from torch import distributions

from delphi.truncated.distributions import truncated_boolean_product
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(
    dims: int,
    probability: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: dict,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_boolean_product.TruncatedBooleanProduct,
        target=distributions.Bernoulli(probability),
        phi=phi,
        dims=dims,
        parameter="p_",
        empirical=lambda s: s.mean(0),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
    )


def _not_both_set(z):
    """Return membership for the set excluding the all-ones corner.

    Args:
        z: Samples to test.

    Returns:
        Boolean mask over the batch.
    """
    return ~((z[:, 0] == 1) * (z[:, 1] == 1))


def test_two_dimensions():
    _recovers(
        dims=2,
        probability=ch.Tensor([0.5, 0.75]),
        phi=_not_both_set,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 1,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-1,
            "project": False,
        },
    )


def test_twenty_dimensions():
    _recovers(
        dims=20,
        probability=ch.Tensor([0.5, 0.75, 0.3, 0.4, 0.8]).repeat(4),
        phi=_not_both_set,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 1,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-1,
        },
    )


def test_five_dimensions_random_polytope():
    # Two dimensions would leave four atoms, too few for a set carrying
    # survival mass near a third to keep every coordinate identifiable.
    probability = ch.Tensor([0.5, 0.75, 0.3, 0.4, 0.8])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_polytope, 0, 5, 4),
        distributions.Bernoulli(probability),
        bounds=(-25.0, 25.0),
    )
    _recovers(
        dims=5,
        probability=probability,
        phi=calibrated.phi,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 1,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-1,
            "project": False,
        },
    )


def test_one_dimension_identity():
    # phi admits everything, so the procedure should reduce to the
    # untruncated MLE and the naive estimate is the right answer.
    _recovers(
        dims=1,
        probability=ch.Tensor([0.4]),
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
        probability=ch.linspace(0.3, 0.7, 10),
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
    probability = ch.linspace(0.3, 0.7, 10)
    # One scalar condition, so sampling stays cheap, but it biases every
    # coordinate instead of one.
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        distributions.Bernoulli(probability),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 200.0),
    )
    _recovers(
        dims=10,
        probability=probability,
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
