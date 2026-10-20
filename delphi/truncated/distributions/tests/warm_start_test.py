# Author: pstefanou12@
"""Tests for refining the initialization to the untruncated MLE."""

import torch as ch
from torch import distributions

from delphi.exponential_family import beta
from delphi.truncated.distributions import truncated_beta
from delphi.truncated.distributions.tests import helpers


class _WarmStartedBeta(truncated_beta.TruncatedBeta):
    needs_warm_start = True


def _phi(z):
    return z[:, 0] > 0.3


def _sample():
    ch.manual_seed(0)
    target = distributions.Beta(ch.Tensor([2.0]), ch.Tensor([5.0]))
    return helpers.generate_truncated_distribution_dataset(target, _phi, 10000)


def _untruncated_log_likelihood(theta, sample):
    concentration1, concentration0 = beta.Beta.to_canonical(theta)
    return distributions.Beta(concentration1, concentration0).log_prob(sample.S)


def test_raises_the_untruncated_likelihood():
    sample = _sample()
    hyperparameters = {"iterations": 1}
    cold = truncated_beta.TruncatedBeta(hyperparameters, _phi, 1, sample.alpha)
    warm = _WarmStartedBeta(hyperparameters, _phi, 1, sample.alpha)
    cold.fit(sample.S)
    warm.fit(sample.S)
    assert (
        _untruncated_log_likelihood(warm.emp_theta, sample).mean()
        > _untruncated_log_likelihood(cold.emp_theta, sample).mean()
    )


def test_restores_the_truncation_and_settings():
    sample = _sample()
    model = _WarmStartedBeta({"iterations": 1}, _phi, 1, sample.alpha)
    model.fit(sample.S)
    assert model.phi is _phi
    assert model.criterion_params[0] is _phi
    assert model.hyperparameters.project
    assert model.hyperparameters.iterations == 1
