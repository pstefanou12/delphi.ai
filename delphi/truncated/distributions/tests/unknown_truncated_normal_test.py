# Author: pstefanou12@
"""Tests for the normal with unknown truncation."""

import pytest
import torch as ch
from torch import distributions
from torch.distributions import kl

from delphi.truncated.distributions import unknown_truncated_normal
from delphi.truncated.distributions.tests import helpers as test_helpers

ch.set_printoptions(precision=4, sci_mode=False)


# TODO: pstefanou12@ - re-enable with the unknown-truncation rework.
# _reparameterize_canon_form is defined only on the unknown-variance class.
@pytest.mark.skip(
    reason="unknown-truncation rework: AttributeError on _reparameterize_canon_form"
)
def test_unknown_truncation_normal_known_variance():
    """1D normal with unknown truncation and known variance recovers true mean."""
    ch.manual_seed(0)
    dims = 1
    M = distributions.MultivariateNormal(ch.zeros(dims), ch.eye(dims))

    left = ch.Tensor([0.0])

    def phi(x):
        return (x > left).prod(dim=-1, keepdim=True)

    num_samples = 10000
    S, alpha, parent = test_helpers.generate_truncated_distribution_dataset(
        M, phi, num_samples
    )
    print(f"alpha: {alpha}")
    print(f"num_samples: {num_samples}")

    emp_loc = S.mean(0)
    print(f"emp loc:\n {emp_loc}")
    emp_var = S.var(0, keepdim=True)
    print(f"emp var:\n {emp_var}")
    emp_scale = ch.sqrt(emp_var)

    S_std_norm = (S - emp_loc) / emp_scale
    true_cov_std = M.covariance_matrix / emp_var

    print(f"emp loc: {emp_loc}")
    print(f"known variance: {M.covariance_matrix}")
    k = 12
    print(f"k: {k}")

    # train algorithm
    args = {
        "epochs": 2,
        "trials": 1,
        "batch_size": 10,
        "lr": 1e-1,
        "early_stopping": True,
        "verbose": True,
    }
    truncated = unknown_truncated_normal.UnknownTruncationNormal(
        args, k, alpha, dims, variance=true_cov_std
    )
    truncated.fit(S_std_norm)
    best_loc = truncated.best_loc_ * ch.sqrt(emp_var) + emp_loc
    print(f"best loc:\n {best_loc.T}")
    best_m = distributions.MultivariateNormal(best_loc, M.covariance_matrix)
    kl_truncated = kl.kl_divergence(best_m, M)
    print(f"truncated kl divergence: {kl_truncated.item():.3f}")

    ema_loc = truncated.ema_loc_ * ch.sqrt(emp_var) + emp_loc
    print(f"ema loc:\n {ema_loc.T}")
    ema_m = distributions.MultivariateNormal(ema_loc, M.covariance_matrix)
    ema_kl_div = kl.kl_divergence(ema_m, M)
    print(f"ema kl divergence: {ema_kl_div}")

    avg_loc = truncated.avg_loc_ * ch.sqrt(emp_var) + emp_loc
    print(f"avg loc:\n {avg_loc.T}")
    avg_m = distributions.MultivariateNormal(avg_loc, M.covariance_matrix)
    avg_kl_div = kl.kl_divergence(avg_m, M)
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_emp = kl.kl_divergence(
        distributions.MultivariateNormal(emp_loc, M.covariance_matrix), M
    )
    print(f"empirical kl divergence: {kl_emp.item():.3f}")
    print(f"truncated kl divergence: {kl_truncated.item():.3f}")
    assert ema_kl_div <= 1e-1, (
        f"EMA KL divergence to true distribution exceeds 0.1: {ema_kl_div}"
    )
    assert avg_kl_div <= 0.15, (
        f"Average KL divergence to true distribution exceeds 0.15: {avg_kl_div}"
    )
    assert kl_truncated <= 0.2, (
        f"Best KL divergence to true distribution exceeds 0.2: {kl_truncated}"
    )


# TODO: pstefanou12@ - re-enable with the unknown-truncation rework. A
# Tensor is formatted with a float format spec in an f-string.
@pytest.mark.skip(
    reason="unknown-truncation rework: TypeError formatting a Tensor as a float"
)
def test_unknown_truncation_normal():
    """1D normal with unknown truncation recovers true mean and variance."""
    ch.manual_seed(0)
    dims = 1
    # generate ground-truth data
    M = distributions.MultivariateNormal(ch.zeros(dims), ch.eye(dims))
    right = ch.Tensor([0.0])

    def phi(x):
        return (x < right).prod(dim=-1)

    num_samples = 10000
    S, alpha, parent = test_helpers.generate_truncated_distribution_dataset(
        M, phi, num_samples
    )
    print(f"alpha: {alpha}")
    print(f"num_samples: {num_samples}")
    emp_loc = S.mean(0)
    emp_var = S.var(0, keepdim=True)
    emp_scale = ch.sqrt(emp_var)
    S_std_norm = (S - emp_loc) / emp_scale

    print(f"emp loc: {emp_loc}")
    print(f"emp variance: {emp_var}")

    k = 20

    # train algorithm
    args = {
        "epochs": 3,
        "trials": 1,
        "batch_size": 10,
        "lr": 1e-1,
        "covariance_matrix_lr": 1e-1,
        "early_stopping": True,
        "verbose": True,
    }
    truncated = unknown_truncated_normal.UnknownTruncationNormal(args, k, alpha, dims)
    truncated.fit(S_std_norm)
    # rescale distribution
    best_loc = truncated.best_loc_ * emp_scale + emp_loc
    best_variance = truncated.best_variance_ * emp_var
    print(f"best loc:\n {best_loc.T}")
    print(f"best variance:\n {best_variance}")
    best_m = distributions.MultivariateNormal(best_loc, best_variance)

    ema_loc = truncated.ema_loc_ * emp_scale + emp_loc
    ema_variance = truncated.ema_variance_ * emp_var
    print(f"ema loc:\n {ema_loc.T}")
    print(f"ema variance:\n {ema_variance}")
    ema_m = distributions.MultivariateNormal(ema_loc, ema_variance)
    ema_kl_div = kl.kl_divergence(ema_m, M)
    print(f"ema kl divergence: {ema_kl_div}")

    avg_loc = truncated.avg_loc_ * emp_scale + emp_loc
    avg_variance = truncated.avg_variance_ * emp_var
    print(f"avg loc:\n {avg_loc.T}")
    print(f"avg variance:\n {avg_variance}")
    avg_m = distributions.MultivariateNormal(avg_loc, avg_variance)
    avg_kl_div = kl.kl_divergence(avg_m, M)
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_truncated = kl.kl_divergence(best_m, M)
    kl_emp = kl.kl_divergence(distributions.MultivariateNormal(emp_loc, emp_var), M)
    print(f"empirical kl divergence: {kl_emp.item():.3f}")
    print(f"truncated kl divergence: {kl_truncated.item():.3f}")
    assert ema_kl_div <= 1e-1, (
        f"EMA KL divergence to true distribution exceeds 0.1: {ema_kl_div}"
    )
    assert avg_kl_div <= 0.15, (
        f"Average KL divergence to true distribution exceeds 0.15: {avg_kl_div}"
    )
    assert kl_truncated <= 0.2, (
        f"Best KL divergence to true distribution exceeds 0.2: {kl_truncated}"
    )
