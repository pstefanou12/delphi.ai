# Author: pstefanou12@
"""Tests for the truncated normal."""

import torch as ch
from torch import distributions
from torch.distributions import kl

from delphi.truncated.distributions import truncated_normal
from delphi.truncated.distributions.tests import helpers

ch.set_printoptions(precision=4, sci_mode=False)


def test_truncated_normal():
    """Right-truncated 1D normal with unknown variance converges to true params."""
    ch.manual_seed(0)
    M = distributions.MultivariateNormal(ch.zeros(1), ch.eye(1))

    left = ch.Tensor([0.0])

    def phi(x):
        return (x > left).prod(dim=-1, keepdim=True)

    num_samples = 1000
    S, alpha, parent = helpers.generate_truncated_distribution_dataset(
        M, phi, num_samples
    )
    print(f"alpha: {alpha}")
    print(f"num_samples: {num_samples}")

    emp_loc = S.mean(0)
    emp_var = S.var(0)
    emp_scale = ch.sqrt(emp_var)
    print(f"emp loc:\n {emp_loc}")
    print(f"emp var:\n {emp_var}")

    S_std_norm = (S - emp_loc) / emp_scale
    left_std_norm = ((left - emp_loc) / emp_scale).flatten()

    def phi_std_norm(x):
        return (x > left_std_norm).prod(dim=-1, keepdim=True)

    hyperparameters = {
        "iterations": 1500,
        "batch_size": 10,
        "trials": 1,
        "verbose": True,
        "lr": 1e-2,
        "num_samples": 1000,
        "early_stopping": True,
        "tol": 1e-3,
        "val_interval": 100,
    }
    truncated = truncated_normal.TruncatedNormal(hyperparameters, phi_std_norm, alpha)
    truncated.fit(S_std_norm)

    best_loc = truncated.best_loc_ * emp_scale + emp_loc
    best_covariance_matrix = truncated.best_covariance_matrix_ * emp_var
    print(f"best loc:\n {best_loc}")
    print(f"best covariance matrix:\n {best_covariance_matrix}")
    best_m = distributions.MultivariateNormal(best_loc, best_covariance_matrix)
    kl_truncated = kl.kl_divergence(best_m, M)
    print(f"truncated kl divergence: {kl_truncated.item():.3f}")

    ema_loc = truncated.ema_loc_ * emp_scale + emp_loc
    ema_covariance_matrix = truncated.ema_covariance_matrix_ * emp_var
    print(f"ema loc:\n {ema_loc}")
    print(f"ema covariance matrix:\n {ema_covariance_matrix}")
    ema_m = distributions.MultivariateNormal(ema_loc, ema_covariance_matrix)
    ema_kl_div = kl.kl_divergence(ema_m, M)
    print(f"ema kl divergence: {ema_kl_div}")

    avg_loc = truncated.avg_loc_ * emp_scale + emp_loc
    avg_covariance_matrix = truncated.avg_covariance_matrix_ * emp_var
    print(f"avg loc:\n {avg_loc}")
    print(f"avg covariance matrix:\n {avg_covariance_matrix}")
    avg_m = distributions.MultivariateNormal(avg_loc, avg_covariance_matrix)
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
