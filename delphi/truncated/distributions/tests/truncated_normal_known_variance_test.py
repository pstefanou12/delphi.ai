# Author: pstefanou12@
"""Tests for the truncated normal with known variance."""

import torch as ch
from torch import distributions
from torch.distributions import kl

from delphi.truncated.distributions import truncated_normal_known_variance
from delphi.truncated.distributions.tests import helpers

ch.set_printoptions(precision=4, sci_mode=False)


def test_one_dimension():
    ch.manual_seed(0)
    covariance_matrix = ch.eye(1)
    target = distributions.MultivariateNormal(ch.zeros(1), covariance_matrix)
    left = ch.Tensor([0.0])

    def phi(x):
        return (x > left).prod(dim=-1, keepdim=True)

    S, alpha, _ = helpers.generate_truncated_distribution_dataset(  # pylint: disable=invalid-name
        target, phi, 1000
    )

    # The procedure is run on centered data, so the shift is undone on the way
    # out; phi moves with it.
    emp_loc = S.mean(0)
    left_centred = (left - emp_loc).flatten()

    def phi_centred(x):
        return (x > left_centred).prod(dim=-1, keepdim=True)

    hyperparameters = {
        "iterations": 1500,
        "trials": 1,
        "batch_size": 10,
        "num_samples": 1000,
        "lr": 1e-1,
        "verbose": True,
        "early_stopping": True,
        "tol": 5e-2,
    }
    truncated = truncated_normal_known_variance.TruncatedNormalKnownVariance(
        hyperparameters, phi_centred, covariance_matrix, alpha
    )
    truncated.fit(S - emp_loc)

    helpers.assert_recovers(
        estimate=lambda name: distributions.MultivariateNormal(
            getattr(truncated, f"{name}_loc_") + emp_loc, covariance_matrix
        ),
        divergence=lambda fitted: kl.kl_divergence(fitted, target),
        baseline=distributions.MultivariateNormal(emp_loc, covariance_matrix),
    )
