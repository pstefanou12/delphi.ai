# Author: pstefanou12@
"""Tests for unknown truncation exponential family distributions."""

import torch as ch
from torch import distributions
from torch.distributions import kl

import delphi.oracle
from delphi.truncated.distributions import unknown_truncated_multivariate_normal
from delphi.truncated.distributions import unknown_truncated_normal
from delphi.utils import helpers
from tests import helpers as test_helpers

ch.set_printoptions(precision=4, sci_mode=False)


def test_unknown_truncation_normal_known_variance():
    """1D normal with unknown truncation and known variance recovers true mean."""
    ch.manual_seed(0)
    dims = 1
    M = distributions.MultivariateNormal(ch.zeros(dims), ch.eye(dims))

    phi = delphi.oracle.Left_Distribution(ch.Tensor([0.0]))
    num_samples = 10000
    S, alpha = test_helpers.generate_truncated_distribution_dataset(M, phi, num_samples)
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


def test_unknown_truncation_normal():
    """1D normal with unknown truncation recovers true mean and variance."""
    ch.manual_seed(0)
    dims = 1
    # generate ground-truth data
    M = distributions.MultivariateNormal(ch.zeros(dims), ch.eye(dims))
    phi = delphi.oracle.Right_Distribution(ch.Tensor([0.0]))
    num_samples = 10000
    S, alpha = test_helpers.generate_truncated_distribution_dataset(M, phi, num_samples)
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
    kl_emp = kl.kl_divergence(
        distributions.MultivariateNormal(emp_loc, emp_var), M
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


def test_unknown_truncation_multivariate_normal():
    """10D sphere-truncated MVN with unknown truncation recovers true params."""
    ch.manual_seed(0)
    dims = 10
    M = distributions.MultivariateNormal(ch.zeros(dims), ch.eye(dims))

    W = distributions.Uniform(-0.5, 0.5)
    centroid = W.sample([dims])
    phi = delphi.oracle.Sphere(M.covariance_matrix, centroid, 3.5)

    num_samples = 5000
    S, alpha = test_helpers.generate_truncated_distribution_dataset(M, phi, num_samples)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")

    emp_loc = S.mean(0)
    emp_cov = helpers.cov(S)
    L = ch.linalg.cholesky(emp_cov)  # pylint: disable=not-callable
    L_inv = ch.linalg.inv(L)  # pylint: disable=not-callable
    S_white = (S - emp_loc) @ L_inv.T

    k = 12
    args = {
        "epochs": 25,
        "batch_size": 100,
        "verbose": True,
    }
    truncated = unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormal(
        args, k, alpha, dims
    )
    truncated.fit(S_white)

    best_loc = truncated.best_loc_ @ L.T + emp_loc
    best_covariance_matrix = L @ truncated.best_covariance_matrix_ @ L.T
    print(f"best loc:\n {best_loc.T}")
    print(f"best covariance matrix:\n {best_covariance_matrix}")
    best_m = distributions.MultivariateNormal(best_loc, best_covariance_matrix)
    kl_truncated = kl.kl_divergence(best_m, M)
    print(f"truncated kl divergence: {kl_truncated.item():.3f}")

    ema_loc = truncated.ema_loc_ @ L.T + emp_loc
    ema_covariance_matrix = L @ truncated.ema_covariance_matrix_ @ L.T
    print(f"ema loc:\n {ema_loc.T}")
    print(f"ema covariance matrix:\n {ema_covariance_matrix}")
    ema_m = distributions.MultivariateNormal(ema_loc, ema_covariance_matrix)
    ema_kl_div = kl.kl_divergence(ema_m, M)
    print(f"ema kl divergence: {ema_kl_div}")

    avg_loc = truncated.avg_loc_ @ L.T + emp_loc
    avg_covariance_matrix = L @ truncated.avg_covariance_matrix_ @ L.T
    print(f"avg loc:\n {avg_loc.T}")
    print(f"avg covariance matrix:\n {avg_covariance_matrix}")
    avg_m = distributions.MultivariateNormal(avg_loc, avg_covariance_matrix)
    avg_kl_div = kl.kl_divergence(avg_m, M)
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_emp = kl.kl_divergence(
        distributions.MultivariateNormal(emp_loc, emp_cov), M
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
