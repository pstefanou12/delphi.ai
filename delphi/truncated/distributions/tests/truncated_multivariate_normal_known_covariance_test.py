# Author: pstefanou12@
"""Tests for the truncated multivariate normal with known covariance."""

import torch as ch
from torch import distributions
from torch.distributions import kl

from delphi.truncated.distributions import (
    truncated_multivariate_normal_known_covariance,
)
from delphi.truncated.distributions.tests import helpers

ch.set_printoptions(precision=4, sci_mode=False)


def test_two_dimensions():
    ch.manual_seed(0)
    dims = 2
    target = distributions.MultivariateNormal(ch.zeros(dims), 2 * ch.eye(dims))

    def phi(x):
        return x[:, 0] > 0

    S, alpha, _ = helpers.generate_truncated_distribution_dataset(  # pylint: disable=invalid-name
        target, phi, 5000
    )

    hyperparameters = {
        "iterations": 2500,
        "trials": 1,
        "batch_size": 1,
        "num_samples": 10000,
        "lr": 1e-2,
        "verbose": True,
    }
    truncated = truncated_multivariate_normal_known_covariance.TruncatedMultivariateNormalKnownCovariance(
        hyperparameters, phi, dims, target.covariance_matrix, alpha
    )
    truncated.fit(S)

    helpers.assert_recovers(
        estimate=lambda name: distributions.MultivariateNormal(
            getattr(truncated, f"{name}_loc_"), target.covariance_matrix
        ),
        divergence=lambda fitted: kl.kl_divergence(fitted, target),
        baseline=distributions.MultivariateNormal(
            S.mean(0, keepdim=True), target.covariance_matrix
        ),
    )


def test_ten_dimensions():
    ch.manual_seed(0)
    dims = 10
    target = distributions.MultivariateNormal(ch.zeros(dims), 10 * ch.eye(dims))

    def phi(x):
        return x[:, 0] > 0

    S, alpha, _ = helpers.generate_truncated_distribution_dataset(  # pylint: disable=invalid-name
        target, phi, 1000
    )

    # phi is standardized too, or the fit sees the wrong truncation set.
    emp_loc = S.mean(0, keepdim=True)
    true_scale = ch.sqrt(target.covariance_matrix.diag())
    scaled_covariance = (
        ch.diag(true_scale).inverse()
        @ target.covariance_matrix
        @ ch.diag(true_scale).inverse()
    )

    def phi_standardised(x):
        return x[:, 0] > (0 - emp_loc[0, 0]) / true_scale[0]

    hyperparameters = {
        "epochs": 2,
        "trials": 1,
        "batch_size": 1,
        "num_samples": 1000,
        "optimizer": "sgd",
        "lr": 1e-2,
        "verbose": True,
    }
    truncated = truncated_multivariate_normal_known_covariance.TruncatedMultivariateNormalKnownCovariance(
        hyperparameters, phi_standardised, dims, scaled_covariance, alpha
    )
    truncated.fit((S - emp_loc) / true_scale)

    helpers.assert_recovers(
        estimate=lambda name: distributions.MultivariateNormal(
            getattr(truncated, f"{name}_loc_") * true_scale + emp_loc,
            target.covariance_matrix,
        ),
        divergence=lambda fitted: kl.kl_divergence(fitted, target),
        baseline=distributions.MultivariateNormal(emp_loc, target.covariance_matrix),
    )
