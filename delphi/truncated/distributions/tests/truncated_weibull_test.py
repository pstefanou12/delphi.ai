# Author: pstefanou12@
"""Tests for the truncated Weibull distribution."""

import functools

import torch as ch
from torch import distributions

from delphi.truncated.distributions import truncated_weibull
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(
    dims: int,
    concentration: ch.Tensor,
    scale: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: dict,
) -> None:
    """Fit a truncated Weibull and assert it recovers the untruncated MLE.

    The comparison is against the MLE of the draw before truncation, not the
    true scale: the latter would charge the estimator for sampling error that
    no estimator can remove.

    Args:
        dims: Number of dimensions.
        concentration: Known shape per dimension.
        scale: True scale per dimension.
        phi: Truncation set oracle.
        hyperparameters: Hyperparameter dict for the procedure.
    """
    ch.manual_seed(0)
    target = distributions.Weibull(scale, concentration)
    S, alpha, parent = helpers.generate_truncated_distribution_dataset(  # pylint: disable=invalid-name
        target, phi, 10000
    )
    truncated = truncated_weibull.TruncatedWeibull(
        hyperparameters, phi, dims, concentration, alpha
    )
    truncated.fit(S)

    mle = parent.pow(concentration).mean(0).pow(1.0 / concentration)
    helpers.assert_recovers(
        estimate=lambda name: getattr(truncated, f"{name}_lambda_").flatten(),
        divergence=lambda fitted: helpers.weibull_kl_divergence(
            fitted, mle, concentration
        ).sum(),
        baseline=S.pow(concentration).mean(0).pow(1.0 / concentration).flatten(),
    )


def test_one_dimension():
    _recovers(
        dims=1,
        concentration=ch.Tensor([2.0]),
        scale=ch.Tensor([1.0]),
        phi=lambda z: z > 1.5,
        hyperparameters={
            "iterations": 1500,
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
        concentration=ch.Tensor([1.0, 1.0]),
        scale=ch.Tensor([1.0, 2.0]),
        phi=lambda z: z[:, 0] > 1.0,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
        },
    )


def test_two_dimensions_differing_shapes():
    _recovers(
        dims=2,
        concentration=ch.Tensor([2.0, 3.0]),
        scale=ch.Tensor([1.0, 2.0]),
        phi=lambda z: z[:, 0] > 1.0,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 100,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
        },
    )


def test_twenty_dimensions_differing_shapes():
    _recovers(
        dims=20,
        concentration=ch.Tensor([2.0, 3.0, 1.0, 2.0, 1.0]).repeat(4),
        scale=ch.Tensor([1.0, 2.0, 3.0, 1.0, 5.0]).repeat(4),
        phi=lambda z: z[:, 0] > 1.0,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
        },
    )


def test_two_dimensions_random_half_space():
    concentration, scale = ch.Tensor([2.0, 2.0]), ch.Tensor([1.0, 2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 2, 2),
        distributions.Weibull(scale, concentration),
        bounds=(-25.0, 25.0),
    )
    _recovers(
        dims=2,
        concentration=concentration,
        scale=scale,
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


def test_two_dimensions_random_disc_union():
    concentration, scale = ch.Tensor([2.0, 2.0]), ch.Tensor([1.0, 2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_disc_union, 1, 2, 6),
        distributions.Weibull(scale, concentration),
        bounds=(1e-4, 1e3),
    )
    _recovers(
        dims=2,
        concentration=concentration,
        scale=scale,
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
