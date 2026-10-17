# Author: pstefanou12@
"""Tests for the truncated multivariate normal."""

import torch as ch
from torch import distributions
from torch.distributions import kl

from delphi.truncated.distributions import truncated_multivariate_normal
from delphi.truncated.distributions.tests import helpers
from delphi.utils import helpers as delphi_helpers

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(
    dims: int,
    covariance_matrix: ch.Tensor,
    dataset_size: int,
    hyperparameters: dict,
) -> None:
    """Fit a truncated multivariate normal and assert it recovers the truth.

    Args:
        dims: Number of dimensions.
        covariance_matrix: True covariance.
        dataset_size: Truncated samples to draw.
        hyperparameters: Hyperparameter dict for the procedure.
    """
    ch.manual_seed(0)
    target = distributions.MultivariateNormal(ch.zeros(dims), covariance_matrix)

    def phi(x):
        return x[:, 0] > 0

    S, alpha, _ = helpers.generate_truncated_distribution_dataset(  # pylint: disable=invalid-name
        target, phi, dataset_size
    )

    # phi is standardized too, or the fit sees the wrong truncation set.
    emp_loc = S.mean(0, keepdim=True)
    emp_var = S.var(0)
    emp_scale = ch.sqrt(emp_var)
    emp_sigma_diag = ch.diag(emp_scale)

    def phi_standardised(x):
        return x[:, 0] > (0 - emp_loc[0, 0]) / emp_scale[0]

    truncated = truncated_multivariate_normal.TruncatedMultivariateNormal(
        hyperparameters, phi_standardised, dims, alpha
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
    )


def test_two_dimensions():
    _recovers(
        dims=2,
        covariance_matrix=2 * ch.eye(2),
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
