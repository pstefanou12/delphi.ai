# Author: pstefanou12@
"""Tests for the truncated normal with known variance."""

import torch as ch
import torch.distributions
import torch.distributions.kl

from delphi.truncated.distributions import truncated_normal_known_variance
from delphi.truncated.distributions.tests import helpers

ch.set_printoptions(precision=4, sci_mode=False)


def test_one_dimension():
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
    truncated = truncated_normal_known_variance.TruncatedNormalKnownVariance(
        phi_centred, covariance_matrix, alpha, hyperparameters=hyperparameters
    )
    truncated.fit(S - emp_loc)

    helpers.assert_recovers(
        estimate=lambda name: torch.distributions.MultivariateNormal(
            getattr(truncated, f"{name}_loc_") + emp_loc, covariance_matrix
        ),
        divergence=lambda fitted: torch.distributions.kl.kl_divergence(fitted, target),
        baseline=torch.distributions.MultivariateNormal(emp_loc, covariance_matrix),
    )


def test_unknown_truncation():
    # Over seeds 0-4 the estimate's divergence measured 0.004 to 0.013 and the
    # learned set's error 0.070 to 0.085.
    helpers.assert_family_learns_set(
        model_class=lambda phi, dims, *args, **kwargs: (
            truncated_normal_known_variance.TruncatedNormalKnownVariance(
                phi, ch.eye(1), *args, **kwargs
            )
        ),
        target=torch.distributions.MultivariateNormal(ch.zeros(1), ch.eye(1)),
        phi=lambda z: z[:, 0] > 0.0,
        dims=1,
        parameter="loc_",
        empirical=lambda s: s.mean(0),
        rebuild=lambda loc: torch.distributions.MultivariateNormal(loc, ch.eye(1)),
        tolerance=0.026,
        set_tolerance=0.17,
    )
