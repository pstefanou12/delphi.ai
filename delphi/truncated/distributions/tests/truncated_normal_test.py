# Author: pstefanou12@
"""Tests for the truncated normal."""

import torch as ch
from torch import distributions
from torch.distributions import kl

from delphi.truncated.distributions import truncated_normal
from delphi.truncated.distributions.tests import helpers

ch.set_printoptions(precision=4, sci_mode=False)


def test_one_dimension():
    ch.manual_seed(0)
    target = distributions.MultivariateNormal(ch.zeros(1), ch.eye(1))
    left = ch.Tensor([0.0])

    def phi(x):
        return (x > left).prod(dim=-1, keepdim=True)

    S, alpha, _ = helpers.generate_truncated_distribution_dataset(  # pylint: disable=invalid-name
        target, phi, 1000
    )

    # The procedure is run on standardised data, so the transform is undone on
    # the way out; phi moves with it.
    emp_loc, emp_var = S.mean(0), S.var(0)
    emp_scale = ch.sqrt(emp_var)
    left_standardised = ((left - emp_loc) / emp_scale).flatten()

    def phi_standardised(x):
        return (x > left_standardised).prod(dim=-1, keepdim=True)

    hyperparameters = {
        "iterations": 1500,
        "trials": 1,
        "batch_size": 10,
        "num_samples": 1000,
        "lr": 1e-2,
        "verbose": True,
        "early_stopping": True,
        "tol": 1e-3,
        "val_interval": 100,
    }
    truncated = truncated_normal.TruncatedNormal(
        hyperparameters, phi_standardised, alpha
    )
    truncated.fit((S - emp_loc) / emp_scale)

    helpers.assert_recovers(
        estimate=lambda name: distributions.MultivariateNormal(
            getattr(truncated, f"{name}_loc_") * emp_scale + emp_loc,
            getattr(truncated, f"{name}_covariance_matrix_") * emp_var,
        ),
        divergence=lambda fitted: kl.kl_divergence(fitted, target),
        baseline=distributions.MultivariateNormal(emp_loc, target.covariance_matrix),
    )
