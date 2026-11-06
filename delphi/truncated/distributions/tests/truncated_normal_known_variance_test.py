# Author: pstefanou12@
"""Tests for the truncated normal with known variance."""

import torch as ch
import torch.distributions
import torch.distributions.kl

from delphi.truncated.distributions import truncated_normal_known_variance
from delphi.truncated.distributions.tests import helpers

ch.set_printoptions(precision=4, sci_mode=False)


@helpers.KNOWN_AND_UNKNOWN_CI
def test_one_dimension(known):
    ch.manual_seed(0)
    covariance_matrix = ch.eye(1)
    target = torch.distributions.MultivariateNormal(ch.zeros(1), covariance_matrix)
    left = ch.tensor([0.0])

    def phi(x):
        return (x > left).prod(dim=-1, keepdim=True)

    S, alpha, _ = helpers.generate_truncated_distribution_dataset(  # pylint: disable=invalid-name
        target, phi, 1000
    )

    # phi is shifted too, or the fit sees the wrong truncation set.
    emp_loc = S.mean(0)
    left_centred = (left - emp_loc).flatten()

    def phi_centred(x):
        return (x > left_centred).prod(dim=-1, keepdim=True)

    hyperparameters = {
        "iterations": 1500,
        "lr": 1e-1,
        "early_stopping": True,
        "tol": 5e-2,
    }
    truncated = helpers.build(
        truncated_normal_known_variance.TruncatedNormalKnownVariance,
        phi_centred,
        known,
        covariance_matrix,
        alpha,
        hyperparameters=hyperparameters,
    )
    truncated.fit(S - emp_loc)
    if not known:
        helpers.assert_learned_set(
            truncated, phi, target, to_model=lambda x: x - emp_loc
        )

    helpers.assert_recovers(
        estimate=lambda name: torch.distributions.MultivariateNormal(
            getattr(truncated, f"{name}_loc_") + emp_loc, covariance_matrix
        ),
        divergence=lambda fitted: torch.distributions.kl.kl_divergence(fitted, target),
        baseline=torch.distributions.MultivariateNormal(emp_loc, covariance_matrix),
    )
