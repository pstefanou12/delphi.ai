# Author: pstefanou12@
"""Tests for the truncated multivariate normal."""

import functools
import math
from typing import Optional

import torch as ch
from torch import distributions
from torch.distributions import kl

from delphi.truncated.distributions import truncated_multivariate_normal
from delphi.truncated.distributions.tests import helpers, truncation_sets
from delphi.utils import helpers as delphi_helpers

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dims: int,
    covariance_matrix: ch.Tensor,
    phi: helpers.Oracle,
    dataset_size: int,
    hyperparameters: dict,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    tolerances: Optional[dict[str, float]] = None,
) -> None:
    """Fit a truncated multivariate normal and assert it recovers the truth.

    Args:
        dims: Number of dimensions.
        covariance_matrix: True covariance.
        phi: Truncation set oracle, stated on the unstandardized data.
        dataset_size: Truncated samples to draw.
        hyperparameters: Hyperparameter dict for the procedure.
    """
    ch.manual_seed(0)
    target = distributions.MultivariateNormal(ch.zeros(dims), covariance_matrix)

    S, alpha, _ = helpers.generate_truncated_distribution_dataset(  # pylint: disable=invalid-name
        target, phi, dataset_size
    )

    emp_loc = S.mean(0, keepdim=True)
    emp_var = S.var(0)
    emp_scale = ch.sqrt(emp_var)
    emp_sigma_diag = ch.diag(emp_scale)

    def phi_standardised(x):
        # Undo the standardization before asking phi, or the fit sees a
        # different set; composing this way holds for any geometry.
        return phi(x * emp_scale + emp_loc)

    args = (alpha,) if pass_alpha else ()
    truncated = truncated_multivariate_normal.TruncatedMultivariateNormal(
        hyperparameters, phi_standardised, dims, *args
    )
    truncated.fit((S - emp_loc) / emp_scale)

    helpers.assert_recovers(
        estimate=lambda name: distributions.MultivariateNormal(
            getattr(truncated, f"{name}_loc_") * emp_scale + emp_loc,
            emp_sigma_diag
            @ getattr(truncated, f"{name}_covariance_matrix_")
            @ emp_sigma_diag,
        ),
        divergence=lambda fitted: kl.kl_divergence(fitted, target),
        baseline=distributions.MultivariateNormal(emp_loc, delphi_helpers.cov(S)),
        baseline_must_fail=baseline_must_fail,
        tolerances=tolerances,
    )


def test_two_dimensions():
    _recovers(
        dims=2,
        covariance_matrix=2 * ch.eye(2),
        phi=lambda z: z[:, 0] > 0,
        dataset_size=1000,
        hyperparameters={
            "iterations": 5000,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-1,
            "covariance_matrix_lr": 1e-2,
        },
    )


def test_ten_dimensions():
    _recovers(
        dims=10,
        covariance_matrix=10 * ch.eye(10),
        phi=lambda z: z[:, 0] > 0,
        dataset_size=5000,
        hyperparameters={
            "iterations": 5000,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-1,
            "covariance_matrix_lr": 1e-2,
        },
    )


def test_two_dimensions_annulus():
    _recovers(
        dims=2,
        covariance_matrix=2 * ch.eye(2),
        phi=truncation_sets.annulus(2.5, 5.0),
        dataset_size=1000,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-1,
            "covariance_matrix_lr": 1e-2,
        },
    )


def test_two_dimensions_crescent():
    _recovers(
        dims=2,
        covariance_matrix=2 * ch.eye(2),
        phi=truncation_sets.crescent(2.0, 6.0, 0.0, math.pi),
        dataset_size=1000,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-1,
            "covariance_matrix_lr": 1e-2,
        },
    )


def test_two_dimensions_random_disc_union():
    covariance_matrix = 2 * ch.eye(2)
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_disc_union, 0, 2, 6),
        distributions.MultivariateNormal(ch.zeros(2), covariance_matrix),
        bounds=(1e-4, 1e3),
    )
    _recovers(
        dims=2,
        covariance_matrix=covariance_matrix,
        phi=calibrated.phi,
        dataset_size=1000,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-1,
            "covariance_matrix_lr": 1e-2,
        },
    )


def test_one_dimension_no_truncation():
    # phi admits everything, so the procedure should reduce to the
    # untruncated MLE and the naive estimate is the right answer.
    _recovers(
        dims=1,
        covariance_matrix=2 * ch.eye(1),
        phi=truncation_sets.identity,
        dataset_size=5000,
        hyperparameters={
            "iterations": 1500,
            "trials": 1,
            "batch_size": 1,
            "num_samples": 10000,
            "optimizer": "sgd",
            "lr": 1e-2,
        },
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_no_truncation():
    _recovers(
        dims=10,
        covariance_matrix=2 * ch.eye(10),
        phi=truncation_sets.identity,
        dataset_size=5000,
        hyperparameters={
            "iterations": 1500,
            "trials": 1,
            "batch_size": 1,
            "num_samples": 10000,
            "optimizer": "sgd",
            "lr": 1e-2,
        },
        pass_alpha=False,
        baseline_must_fail=False,
        # Only the running average converges here. Over 1500 and 4000 steps
        # the EMA measures 0.0910 then 0.1069 against a 0.1 bound, and the
        # validation-selected iterate 0.6199 then 0.3348 against 0.2, so
        # neither is short of steps.
        tolerances={"avg": 0.15},
    )
