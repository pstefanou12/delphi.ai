# Author: pstefanou12@
"""Tests for the truncated multivariate normal with known covariance."""

import functools
import math

import torch as ch
from torch import distributions
from torch.distributions import kl

from delphi.truncated.distributions import (
    truncated_multivariate_normal_known_covariance,
)
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dims: int,
    covariance_matrix: ch.Tensor,
    phi: helpers.Oracle,
    dataset_size: int,
    hyperparameters: dict,
) -> None:
    """Fit a known-covariance truncated normal and assert it recovers the mean.

    The data goes in unstandardized, so phi is the set as stated and any
    geometry survives the round trip.

    Args:
        dims: Number of dimensions.
        covariance_matrix: Known covariance, supplied to the fit.
        phi: Truncation set oracle.
        dataset_size: Truncated samples to draw.
        hyperparameters: Hyperparameter dict for the procedure.
    """
    ch.manual_seed(0)
    target = distributions.MultivariateNormal(ch.zeros(dims), covariance_matrix)
    S, alpha, _ = helpers.generate_truncated_distribution_dataset(  # pylint: disable=invalid-name
        target, phi, dataset_size
    )
    truncated = truncated_multivariate_normal_known_covariance.TruncatedMultivariateNormalKnownCovariance(
        hyperparameters, phi, dims, covariance_matrix, alpha
    )
    truncated.fit(S)

    helpers.assert_recovers(
        estimate=lambda name: distributions.MultivariateNormal(
            getattr(truncated, f"{name}_loc_"), covariance_matrix
        ),
        divergence=lambda fitted: kl.kl_divergence(fitted, target),
        baseline=distributions.MultivariateNormal(
            S.mean(0, keepdim=True), covariance_matrix
        ),
    )


def test_two_dimensions():
    _recovers(
        dims=2,
        covariance_matrix=2 * ch.eye(2),
        phi=lambda x: x[:, 0] > 0,
        dataset_size=5000,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 1,
            "num_samples": 10000,
            "lr": 1e-2,
        },
    )


def test_two_dimensions_crescent():
    _recovers(
        dims=2,
        covariance_matrix=2 * ch.eye(2),
        phi=truncation_sets.crescent(2.0, 6.0, 0.0, math.pi),
        dataset_size=5000,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 1,
            "num_samples": 1000,
            "lr": 1e-2,
        },
    )


def test_two_dimensions_triangle():
    _recovers(
        dims=2,
        covariance_matrix=2 * ch.eye(2),
        phi=truncation_sets.triangle([[0.0, 0.0], [4.0, 1.0], [1.0, 4.0]]),
        dataset_size=5000,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 1,
            "num_samples": 1000,
            "lr": 1e-2,
        },
    )


def test_five_dimensions_random_half_space():
    covariance_matrix = 2 * ch.eye(5)
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 5),
        distributions.MultivariateNormal(ch.zeros(5), covariance_matrix),
        bounds=(-25.0, 25.0),
    )
    _recovers(
        dims=5,
        covariance_matrix=covariance_matrix,
        phi=calibrated.phi,
        dataset_size=5000,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 1,
            "num_samples": 1000,
            "lr": 1e-2,
        },
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
