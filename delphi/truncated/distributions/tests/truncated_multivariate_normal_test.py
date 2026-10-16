# Author: pstefanou12@
"""Tests for the truncated multivariate normal."""

import torch as ch
from torch import distributions
from torch.distributions import kl

from delphi.truncated.distributions import truncated_multivariate_normal
from delphi.truncated.distributions.tests import helpers
from delphi.utils import helpers as delphi_helpers

ch.set_printoptions(precision=4, sci_mode=False)


def test_truncated_2_dim_multivariate_normal():
    """Right-truncated 2D MVN with unknown covariance converges to true params."""
    ch.manual_seed(0)
    dims = 2
    M = distributions.MultivariateNormal(ch.zeros(dims), 2 * ch.eye(dims))
    num_samples = 1000

    def phi(x):
        return x[:, 0] > 0

    S, alpha, parent = helpers.generate_truncated_distribution_dataset(
        M, phi, num_samples
    )

    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")

    emp_loc = S.mean(0, keepdim=True)
    emp_covariance_matrix = delphi_helpers.cov(S)
    emp_var = S.var(0)
    emp_sigma_diag = ch.diag(ch.sqrt(emp_var))

    print(f"empirical mean:\n {emp_loc.T}")
    print(f"empirical covariance matrix:\n {emp_covariance_matrix}")

    def phi_std_norm(x):
        return x[:, 0] > (0 - emp_loc[0, 0]) / ch.sqrt(emp_var[0])

    S_std_norm = (S - emp_loc) / ch.sqrt(emp_var)

    # train algorithm
    hyperparameters = {
        "iterations": 5000,
        "batch_size": 10,
        "num_samples": 1000,
        "verbose": True,
        "lr": 1e-1,
        "optimizer": "sgd",
        "covariance_matrix_lr": 1e-2,
    }
    truncated = truncated_multivariate_normal.TruncatedMultivariateNormal(
        hyperparameters, phi_std_norm, dims, alpha
    )
    truncated.fit(S_std_norm)
    best_loc = truncated.best_loc_ * ch.sqrt(emp_var) + emp_loc
    best_covariance_matrix = (
        emp_sigma_diag @ truncated.best_covariance_matrix_ @ emp_sigma_diag
    )
    print(f"best loc:\n {best_loc.T}")
    print(f"best covariance matrix:\n {best_covariance_matrix}")
    best_m = distributions.MultivariateNormal(best_loc, best_covariance_matrix)
    kl_truncated = kl.kl_divergence(best_m, M)
    print(f"truncated kl divergence: {kl_truncated.item():.3f}")

    ema_loc = truncated.ema_loc_ * ch.sqrt(emp_var) + emp_loc
    ema_covariance_matrix = (
        emp_sigma_diag @ truncated.ema_covariance_matrix_ @ emp_sigma_diag
    )
    print(f"ema loc:\n {ema_loc.T}")
    print(f"ema covariance matrix:\n {ema_covariance_matrix}")
    ema_m = distributions.MultivariateNormal(ema_loc, ema_covariance_matrix)
    ema_kl_div = kl.kl_divergence(ema_m, M)
    print(f"ema kl divergence: {ema_kl_div}")

    avg_loc = truncated.avg_loc_ * ch.sqrt(emp_var) + emp_loc
    avg_covariance_matrix = (
        emp_sigma_diag @ truncated.avg_covariance_matrix_ @ emp_sigma_diag
    )
    print(f"avg loc:\n {avg_loc.T}")
    print(f"avg covariance matrix:\n {avg_covariance_matrix}")
    avg_m = distributions.MultivariateNormal(avg_loc, avg_covariance_matrix)
    avg_kl_div = kl.kl_divergence(avg_m, M)
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_emp = kl.kl_divergence(
        distributions.MultivariateNormal(emp_loc, emp_covariance_matrix), M
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


def test_truncated_10_dim_multivariate_normal():
    """Right-truncated 10D MVN with unknown covariance converges to true params."""
    ch.manual_seed(0)
    dims = 10
    M = distributions.MultivariateNormal(ch.zeros(dims), 10 * ch.eye(dims))

    def phi(x):
        return x[:, 0] > 0

    num_samples = 5000
    S, alpha, parent = helpers.generate_truncated_distribution_dataset(
        M, phi, num_samples
    )
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")

    emp_loc = S.mean(0, keepdim=True)
    emp_covariance_matrix = delphi_helpers.cov(S)
    emp_var = S.var(0)
    emp_sigma_diag = ch.diag(ch.sqrt(emp_var))

    print(f"empirical mean:\n {emp_loc.T}")
    print(f"empirical covariance matrix:\n {emp_covariance_matrix}")

    def phi_std_norm(x):
        return x[:, 0] > ((0 - emp_loc[0, 0]) / ch.sqrt(emp_var[0]))

    S_std_norm = (S - emp_loc) / ch.sqrt(emp_var)

    # train algorithm
    hyperparameters = {
        "iterations": 5000,
        "trials": 1,
        "batch_size": 10,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-1,
        "covariance_matrix_lr": 1e-2,
    }
    truncated = truncated_multivariate_normal.TruncatedMultivariateNormal(
        hyperparameters, phi_std_norm, dims, alpha
    )
    truncated.fit(S_std_norm)

    best_loc = truncated.best_loc_ * ch.sqrt(emp_var) + emp_loc
    best_covariance_matrix = (
        emp_sigma_diag @ truncated.best_covariance_matrix_ @ emp_sigma_diag
    )
    print(f"best loc:\n {best_loc.T}")
    print(f"best covariance matrix:\n {best_covariance_matrix}")
    best_m = distributions.MultivariateNormal(best_loc, best_covariance_matrix)
    kl_truncated = kl.kl_divergence(best_m, M)
    print(f"truncated kl divergence: {kl_truncated.item():.3f}")

    ema_loc = truncated.ema_loc_ * ch.sqrt(emp_var) + emp_loc
    ema_covariance_matrix = (
        emp_sigma_diag @ truncated.ema_covariance_matrix_ @ emp_sigma_diag
    )
    print(f"ema loc:\n {ema_loc.T}")
    print(f"ema covariance matrix:\n {ema_covariance_matrix}")
    ema_m = distributions.MultivariateNormal(ema_loc, ema_covariance_matrix)
    ema_kl_div = kl.kl_divergence(ema_m, M)
    print(f"ema kl divergence: {ema_kl_div}")

    avg_loc = truncated.avg_loc_ * ch.sqrt(emp_var) + emp_loc
    avg_covariance_matrix = (
        emp_sigma_diag @ truncated.avg_covariance_matrix_ @ emp_sigma_diag
    )
    print(f"avg loc:\n {avg_loc.T}")
    print(f"avg covariance matrix:\n {avg_covariance_matrix}")
    avg_m = distributions.MultivariateNormal(avg_loc, avg_covariance_matrix)
    avg_kl_div = kl.kl_divergence(avg_m, M)
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_emp = kl.kl_divergence(
        distributions.MultivariateNormal(emp_loc, emp_covariance_matrix), M
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
