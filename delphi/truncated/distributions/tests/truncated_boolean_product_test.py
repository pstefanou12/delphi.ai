# Author: pstefanou12@
"""Tests for the truncated boolean product distribution."""

import torch as ch
from torch import distributions

from delphi.truncated.distributions import truncated_boolean_product
from delphi.truncated.distributions.tests import helpers

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(dims, probability, phi, hyperparameters):
    helpers.assert_family_recovers(
        model_class=truncated_boolean_product.TruncatedBooleanProduct,
        target=distributions.Bernoulli(probability),
        phi=phi,
        dims=dims,
        parameter="p_",
        empirical=lambda s: s.mean(0),
        hyperparameters=hyperparameters,
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
            "verbose": True,
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
            "verbose": True,
        },
    )
