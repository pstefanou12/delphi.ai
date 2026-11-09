# Author: pstefanou12@
"""Tests for the truncated normal."""

import pytest
import torch as ch
import torch.distributions
import torch.distributions.kl

from delphi.truncated.distributions import truncated_normal
from delphi.truncated.distributions.tests import helpers

ch.set_printoptions(precision=4, sci_mode=False)


@helpers.known_and_unknown(
    ci=True,
    constrained_ci=False,
    constrained_marks=(
        pytest.mark.xfail(
            reason=(
                "constrained learner: the reference, fit to the halved "
                "sample, puts 9% of its mass off the set where the truth "
                "puts 50%, so the program holds p near 1 everywhere; the "
                "set errs on 0.32"
            )
        ),
    ),
)
def test_one_dimension(known):
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
    truncated = helpers.build(
        truncated_normal.TruncatedNormal,
        phi_standardised,
        known,
        alpha,
        hyperparameters=hyperparameters,
    )
    truncated.fit((S - emp_loc) / emp_scale)
    if helpers.learns_set(known):
        helpers.assert_learned_set(
            truncated, phi, target, to_model=lambda x: (x - emp_loc) / emp_scale
        )

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
