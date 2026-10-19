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
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
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
        pass_alpha: Supply the measured survival mass to the model.
        baseline_must_fail: Forwarded to ``assert_recovers``.
    """
    ch.manual_seed(0)
    target = distributions.Weibull(scale, concentration)
    S, alpha, parent = helpers.generate_truncated_distribution_dataset(  # pylint: disable=invalid-name
        target, phi, 10000
    )
    args = (alpha,) if pass_alpha else ()
    truncated = truncated_weibull.TruncatedWeibull(
        hyperparameters, phi, dims, concentration, *args
    )
    truncated.fit(S)

    mle = parent.pow(concentration).mean(0).pow(1.0 / concentration)
    helpers.assert_recovers(
        estimate=lambda name: getattr(truncated, f"{name}_lambda_").flatten(),
        divergence=lambda fitted: helpers.weibull_kl_divergence(
            fitted, mle, concentration
        ),
        baseline=S.pow(concentration).mean(0).pow(1.0 / concentration).flatten(),
        baseline_must_fail=baseline_must_fail,
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


def test_one_dimension_identity():
    # phi admits everything, so the procedure should reduce to the
    # untruncated MLE and the naive estimate is the right answer.
    _recovers(
        dims=1,
        concentration=ch.Tensor([2.0]),
        scale=ch.Tensor([1.0]),
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
        concentration=ch.Tensor([2.0, 3.0, 1.0, 2.0, 1.0]).repeat(2),
        scale=ch.Tensor([1.0, 2.0, 3.0, 1.0, 5.0]).repeat(2),
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


def test_one_dimension_small_mass():
    scale, concentration = ch.Tensor([1.0]), ch.Tensor([2.0])
    # The regime both papers state their guarantees in; nothing else in the
    # suite goes near it.
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Weibull(scale, concentration),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 50.0),
    )
    _recovers(
        dims=1,
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


def test_indistinguishable_pair():
    # Truncated to a narrow window the two samples agree to within sampling
    # error, so only an estimator that knows the window can tell them apart.
    concentration = ch.Tensor([2.0])
    scales = (ch.Tensor([1.0]), ch.Tensor([2.0]))

    def phi(z):
        return (z[:, 0] > 1.0) & (z[:, 0] < 1.2)

    fits, naive = [], []
    for scale in scales:
        ch.manual_seed(0)
        sample = helpers.generate_truncated_distribution_dataset(
            distributions.Weibull(scale, concentration), phi, 10000
        )
        truncated = truncated_weibull.TruncatedWeibull(
            {
                "iterations": 2500,
                "trials": 1,
                "batch_size": 10,
                "num_samples": 1000,
                "optimizer": "sgd",
                "lr": 1e-2,
                "scheduler": "fisher",
            },
            phi,
            1,
            concentration,
            sample.alpha,
        )
        truncated.fit(sample.S)
        fits.append(truncated)
        naive.append(sample.S.pow(concentration).mean(0).pow(1.0 / concentration))

    for name in helpers.TOLERANCES:
        helpers.assert_distinguishes(
            targets=scales,
            recovered=(
                getattr(fits[0], f"{name}_lambda_").flatten(),
                getattr(fits[1], f"{name}_lambda_").flatten(),
            ),
            naive=(naive[0], naive[1]),
            divergence=lambda a, b: helpers.weibull_kl_divergence(a, b, concentration),
        )
