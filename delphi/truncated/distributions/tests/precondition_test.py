# Author: pstefanou12@
"""Tests for preconditioning the steps by the inverse curvature."""

import torch as ch

from delphi.exponential_family import inverse_gaussian
from delphi.truncated.distributions import truncated_inverse_gaussian
from delphi.truncated.distributions.tests import helpers


def _phi(z):
    return z[:, 0] > 1.5


def _sample():
    ch.manual_seed(0)
    target = inverse_gaussian.InverseGaussian(
        inverse_gaussian.InverseGaussian.to_natural(ch.Tensor([1.0, 2.0])), 1
    )
    return helpers.generate_truncated_distribution_dataset(target, _phi, 1000)


def _one_step(sample, precondition: bool):
    model = truncated_inverse_gaussian.TruncatedInverseGaussian(
        {
            "iterations": 1,
            "max_phases": 1,
            "scheduler": None,
            "project": False,
            "precondition": precondition,
        },
        _phi,
        1,
        sample.alpha,
    )
    # The same seed gives both fits the same batch and rejection draws.
    ch.manual_seed(1)
    model.fit(sample.S)
    assert model.final_params is not None and model.emp_theta is not None
    return model, (model.final_params - model.emp_theta).detach()


def test_steps_by_the_inverse_sample_covariance():
    sample = _sample()
    plain_model, plain = _one_step(sample, precondition=False)
    _, preconditioned = _one_step(sample, precondition=True)
    covariance = plain_model.train_loader_.dataset.S_grad.T.cov()
    expected = ch.linalg.solve(covariance, plain)  # pylint: disable=not-callable
    # Each step is read back as a difference of float32 thetas near -14.
    assert ch.allclose(preconditioned, expected, rtol=1e-2)


def test_off_by_default_outside_the_families_that_need_it():
    model = truncated_inverse_gaussian.TruncatedInverseGaussian(
        {"precondition": False}, _phi, 1
    )
    assert not model.hyperparameters.precondition
    assert truncated_inverse_gaussian.TruncatedInverseGaussian(
        {}, _phi, 1
    ).hyperparameters.precondition
