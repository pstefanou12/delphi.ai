# Author: pstefanou12@
"""Tests for the truncated normal."""

import torch as ch
import torch.distributions
import torch.distributions.kl

from delphi.truncated.distributions import truncated_normal
from delphi.truncated.distributions.tests import helpers

ch.set_printoptions(precision=4, sci_mode=False)


def test_one_dimension():
    ch.manual_seed(0)
    target = torch.distributions.MultivariateNormal(ch.zeros(1), ch.eye(1))
    left = ch.tensor([0.0])

    def phi(x):
        return (x > left).prod(dim=-1, keepdim=True)

    S, alpha, _ = helpers.generate_truncated_distribution_dataset(  # pylint: disable=invalid-name
        target, phi, 1000
    )

    # phi is standardized too, or the fit sees the wrong truncation set.
    emp_loc, emp_var = S.mean(0), S.var(0)
    emp_scale = ch.sqrt(emp_var)
    left_standardised = ((left - emp_loc) / emp_scale).flatten()

    def phi_standardised(x):
        return (x > left_standardised).prod(dim=-1, keepdim=True)

    hyperparameters = {
        "iterations": 1500,
        "early_stopping": True,
        "tol": 1e-3,
        "val_interval": 100,
    }
    truncated = truncated_normal.TruncatedNormal(
        phi_standardised, alpha, hyperparameters=hyperparameters
    )
    truncated.fit((S - emp_loc) / emp_scale)

    helpers.assert_recovers(
        estimate=lambda name: torch.distributions.MultivariateNormal(
            getattr(truncated, f"{name}_loc_") * emp_scale + emp_loc,
            getattr(truncated, f"{name}_covariance_matrix_") * emp_var,
        ),
        divergence=lambda fitted: torch.distributions.kl.kl_divergence(fitted, target),
        baseline=torch.distributions.MultivariateNormal(
            emp_loc, target.covariance_matrix
        ),
    )


def test_unknown_truncation():
    # Over seeds 0-4 the estimate's divergence measured 0.0006 to 0.012 and the
    # learned set's error 0.0009 to 0.0022.
    helpers.assert_family_learns_set(
        model_class=lambda phi, dims, *args, **kwargs: truncated_normal.TruncatedNormal(
            phi, *args, **kwargs
        ),
        target=torch.distributions.MultivariateNormal(ch.zeros(1), ch.eye(1)),
        phi=lambda z: z[:, 0] > 0.0,
        dims=1,
        parameter=("loc_", "covariance_matrix_"),
        empirical=lambda s: (s.mean(0), ch.atleast_2d(s.var(0))),
        rebuild=torch.distributions.MultivariateNormal,
        tolerance=0.023,
        set_tolerance=0.005,
    )
