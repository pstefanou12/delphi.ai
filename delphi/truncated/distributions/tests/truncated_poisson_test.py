# Author: pstefanou12@
"""Tests for the truncated Poisson distribution."""

import torch as ch
from torch import distributions
from torch.distributions import kl

from delphi.truncated.distributions import truncated_poisson
from delphi.truncated.distributions.tests import helpers

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(dims, rate, phi, hyperparameters):
    """Fit a truncated Poisson and assert it recovers the true rates.

    Args:
        dims: Number of dimensions.
        rate: True rate per dimension.
        phi: Truncation set oracle.
        hyperparameters: Hyperparameter dict for the procedure.
    """
    ch.manual_seed(0)
    target = distributions.Poisson(rate)
    S, alpha, _ = helpers.generate_truncated_distribution_dataset(  # pylint: disable=invalid-name
        target, phi, 10000
    )
    truncated = truncated_poisson.TruncatedPoisson(hyperparameters, phi, dims, alpha)
    truncated.fit(S)

    helpers.assert_recovers(
        estimate=lambda name: distributions.Poisson(
            getattr(truncated, f"{name}_lambda_")
        ),
        divergence=lambda fitted: kl.kl_divergence(fitted, target).sum(),
        baseline=distributions.Poisson(S.mean(0)),
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
