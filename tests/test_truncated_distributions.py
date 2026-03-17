# Author: pstefanou12@
"""Tests for truncated exponential family distributions."""

import torch as ch
from torch import distributions
from torch.distributions import kl

from delphi import oracle
from delphi.truncated.distributions import truncated_boolean_product
from delphi.truncated.distributions import truncated_exponential
from delphi.truncated.distributions import truncated_multivariate_normal
from delphi.truncated.distributions import truncated_multivariate_normal_known_covariance
from delphi.truncated.distributions import truncated_normal
from delphi.truncated.distributions import truncated_normal_known_variance
from delphi.truncated.distributions import truncated_poisson
from delphi.truncated.distributions import truncated_weibull
from delphi.utils import helpers as delphi_helpers
from tests import helpers

ch.set_printoptions(precision=4, sci_mode=False)


def test_truncated_normal_known_variance():
    """Right-truncated 1D normal with known variance converges to true mean."""
    ch.manual_seed(0)
    M = distributions.MultivariateNormal(ch.zeros(1), ch.eye(1))
    phi = oracle.Left_Distribution(ch.Tensor([0.0]))
    num_samples = 1000
    S, alpha = helpers.generate_truncated_distribution_dataset(M, phi, num_samples)
    print(f"alpha: {alpha}")
    print(f"num_samples: {num_samples}")

    emp_loc = S.mean(0)
    print(f"emp loc:\n {emp_loc}")
    print(f"known variance:\n {ch.eye(1)}")
    S_std_norm = S - emp_loc
    phi_std_norm = oracle.Left_Distribution((phi.left - emp_loc).flatten())

    args = {
        "batch_size": 10,
        "trials": 1,
        "verbose": True,
        "lr": 1e-1,
        "num_samples": 1000,
        "early_stopping": True,
        "tol": 5e-2,
        "iterations": 1500,
    }
    truncated = truncated_normal_known_variance.TruncatedNormalKnownVariance(args, phi_std_norm, alpha, ch.eye(1))
    truncated.fit(S_std_norm)

    best_loc = truncated.best_loc_ + emp_loc
    print(f"best loc:\n {best_loc}")
    best_m = distributions.MultivariateNormal(best_loc, ch.eye(1))
    kl_truncated = kl.kl_divergence(best_m, M)
    print(f"truncated kl divergence: {kl_truncated.item():.3f}")

    ema_loc = truncated.ema_loc_ + emp_loc
    print(f"ema loc:\n {ema_loc}")
    ema_m = distributions.MultivariateNormal(ema_loc, ch.eye(1))
    ema_kl_div = kl.kl_divergence(ema_m, M)
    print(f"ema kl divergence: {ema_kl_div}")

    avg_loc = truncated.avg_loc_ + emp_loc
    print(f"avg loc:\n {avg_loc}")
    avg_m = distributions.MultivariateNormal(avg_loc, ch.eye(1))
    avg_kl_div = kl.kl_divergence(avg_m, M)
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_emp = kl.kl_divergence(distributions.MultivariateNormal(emp_loc, M.covariance_matrix), M)
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


def test_truncated_normal():
    """Right-truncated 1D normal with unknown variance converges to true params."""
    ch.manual_seed(0)
    M = distributions.MultivariateNormal(ch.zeros(1), ch.eye(1))

    phi = oracle.Left_Distribution(ch.Tensor([0.0]))
    num_samples = 1000
    S, alpha = helpers.generate_truncated_distribution_dataset(M, phi, num_samples)
    print(f"alpha: {alpha}")
    print(f"num_samples: {num_samples}")

    emp_loc = S.mean(0)
    emp_var = S.var(0)
    emp_scale = ch.sqrt(emp_var)
    print(f"emp loc:\n {emp_loc}")
    print(f"emp var:\n {emp_var}")

    S_std_norm = (S - emp_loc) / emp_scale
    phi_std_norm = oracle.Left_Distribution(
        ((phi.left - emp_loc) / emp_scale).flatten()
    )

    args = {
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
    truncated = truncated_normal.TruncatedNormal(args, phi_std_norm, alpha, 1)
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
    kl_emp = kl.kl_divergence(distributions.MultivariateNormal(emp_loc, M.covariance_matrix), M)
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


def test_truncated_2_dim_multivariate_normal_known_covariance_matrix():
    """Right-truncated 2D MVN with known covariance converges to true mean."""
    ch.manual_seed(0)
    dims = 2
    M = distributions.MultivariateNormal(ch.zeros(dims), 2 * ch.eye(dims))

    num_samples = 5000

    def phi(x):
        return x[:, 0] > 0

    S, alpha = helpers.generate_truncated_distribution_dataset(M, phi, num_samples)

    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")

    emp_loc = S.mean(0, keepdim=True)
    print(f"empirical mean:\n {emp_loc.T}")

    args = {
        "iterations": 2500,
        "trials": 1,
        "batch_size": 1,
        "num_samples": 10000,
        "verbose": True,
        "lr": 1e-2,
    }
    truncated = truncated_multivariate_normal_known_covariance.TruncatedMultivariateNormalKnownCovariance(
        args, phi, alpha, dims, M.covariance_matrix
    )
    truncated.fit(S)

    best_loc = truncated.best_loc_
    print(f"best loc:\n {best_loc.T}")
    best_m = distributions.MultivariateNormal(best_loc, M.covariance_matrix)
    kl_truncated = kl.kl_divergence(best_m, M)
    print(f"truncated kl divergence: {kl_truncated.item():.3f}")

    ema_loc = truncated.ema_loc_
    print(f"ema loc:\n {ema_loc.T}")
    ema_m = distributions.MultivariateNormal(ema_loc, M.covariance_matrix)
    ema_kl_div = kl.kl_divergence(ema_m, M)
    print(f"ema kl divergence: {ema_kl_div}")

    avg_loc = truncated.avg_loc_
    print(f"avg loc:\n {avg_loc.T}")
    avg_m = distributions.MultivariateNormal(avg_loc, M.covariance_matrix)
    avg_kl_div = kl.kl_divergence(avg_m, M)
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_emp = kl.kl_divergence(distributions.MultivariateNormal(emp_loc, M.covariance_matrix), M)
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


def test_truncated_2_dim_multivariate_normal():
    """Right-truncated 2D MVN with unknown covariance converges to true params."""
    ch.manual_seed(0)
    dims = 2
    M = distributions.MultivariateNormal(ch.zeros(dims), 2 * ch.eye(dims))
    num_samples = 1000

    def phi(x):
        return x[:, 0] > 0

    S, alpha = helpers.generate_truncated_distribution_dataset(M, phi, num_samples)

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
    args = {
        "iterations": 5000,
        "batch_size": 10,
        "num_samples": 1000,
        "verbose": True,
        "lr": 1e-1,
        "optimizer": "sgd",
        "covariance_matrix_lr": 1e-2,
    }
    truncated = truncated_multivariate_normal.TruncatedMultivariateNormal(args, phi_std_norm, alpha, dims)
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
    kl_emp = kl.kl_divergence(distributions.MultivariateNormal(emp_loc, emp_covariance_matrix), M)
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


def test_truncated_10_dim_multivariate_normal_known_covariance_matrix():
    """Right-truncated 10D MVN with known covariance converges to true mean."""
    ch.manual_seed(0)
    dims = 10
    M = distributions.MultivariateNormal(ch.zeros(dims), 10 * ch.eye(dims))

    def phi(x):
        return x[:, 0] > 0

    num_samples = 1000
    S, alpha = helpers.generate_truncated_distribution_dataset(M, phi, num_samples)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")

    emp_loc = S.mean(0, keepdim=True)
    true_var = M.covariance_matrix.diag()

    print(f"empirical mean:\n {emp_loc.T}")

    def phi_std_norm(x):
        return x[:, 0] > ((0 - emp_loc[0, 0]) / ch.sqrt(true_var[0]))

    S_std_norm = (S - emp_loc) / ch.sqrt(true_var)

    # train algorithm
    args = {
        "epochs": 2,
        "batch_size": 1,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-2,
    }

    scaled_cov = (
        ch.diag(ch.sqrt(true_var)).inverse()
        @ M.covariance_matrix
        @ ch.diag(ch.sqrt(true_var)).inverse()
    )
    truncated = truncated_multivariate_normal_known_covariance.TruncatedMultivariateNormalKnownCovariance(
        args, phi_std_norm, alpha, dims, scaled_cov
    )
    truncated.fit(S_std_norm)
    best_loc = truncated.best_loc_ * ch.sqrt(true_var) + emp_loc
    print(f"best loc:\n {best_loc.T}")
    best_m = distributions.MultivariateNormal(best_loc, M.covariance_matrix)
    kl_truncated = kl.kl_divergence(best_m, M)
    print(f"truncated kl divergence: {kl_truncated.item():.3f}")

    ema_loc = truncated.ema_loc_ * ch.sqrt(true_var) + emp_loc
    print(f"ema loc:\n {ema_loc.T}")
    ema_m = distributions.MultivariateNormal(ema_loc, M.covariance_matrix)
    ema_kl_div = kl.kl_divergence(ema_m, M)
    print(f"ema kl divergence: {ema_kl_div}")

    avg_loc = truncated.avg_loc_ * ch.sqrt(true_var) + emp_loc
    print(f"avg loc:\n {avg_loc.T}")
    avg_m = distributions.MultivariateNormal(avg_loc, M.covariance_matrix)
    avg_kl_div = kl.kl_divergence(avg_m, M)
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_emp = kl.kl_divergence(distributions.MultivariateNormal(emp_loc, M.covariance_matrix), M)
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
    S, alpha = helpers.generate_truncated_distribution_dataset(M, phi, num_samples)
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
    args = {
        "iterations": 5000,
        "trials": 1,
        "batch_size": 10,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-1,
        "covariance_matrix_lr": 1e-2,
    }
    truncated = truncated_multivariate_normal.TruncatedMultivariateNormal(args, phi_std_norm, alpha, dims)
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
    kl_emp = kl.kl_divergence(distributions.MultivariateNormal(emp_loc, emp_covariance_matrix), M)
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


def test_truncated_boolean_product_2_dims():
    """2D truncated Bernoulli product recovers true probabilities."""
    ch.manual_seed(0)
    dims = 2
    p = ch.Tensor([0.5, 0.75])
    print(f"true p: {p}")
    dist = distributions.Bernoulli(p)

    def phi(z):
        return ~((z[:, 0] == 1) * (z[:, 1] == 1))

    num_samples = 10000
    S, alpha = helpers.generate_truncated_distribution_dataset(dist, phi, num_samples)

    emp_p = S.mean(0)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_p: {emp_p}")

    args = {
        "iterations": 2500,
        "trials": 1,
        "batch_size": 1,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-1,
        "max_phases": 1000000,
        "rate": 1.5,
        "project": False,
    }

    truncated = truncated_boolean_product.TruncatedBooleanProduct(args, phi, alpha, dims)
    truncated.fit(S)

    best_p = truncated.best_p_
    print(f"best p:\n {best_p.T}")
    best_m = distributions.Bernoulli(best_p)

    ema_p = truncated.ema_p_
    print(f"ema p:\n {ema_p.T}")
    ema_m = distributions.Bernoulli(ema_p)
    ema_kl_div = kl.kl_divergence(ema_m, dist).sum()
    print(f"ema kl divergence: {ema_kl_div}")

    avg_p = truncated.avg_p_
    print(f"avg p:\n {avg_p.T}")
    avg_m = distributions.Bernoulli(avg_p)
    avg_kl_div = kl.kl_divergence(avg_m, dist).sum()
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_truncated = kl.kl_divergence(best_m, dist).sum()
    kl_emp = kl.kl_divergence(distributions.Bernoulli(emp_p), dist).sum()
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


def test_truncated_boolean_product_20_dims():
    """20D truncated Bernoulli product recovers true probabilities."""
    ch.manual_seed(0)
    dims = 20
    p = ch.Tensor(
        [
            0.5,
            0.75,
            0.3,
            0.4,
            0.8,
            0.5,
            0.75,
            0.3,
            0.4,
            0.8,
            0.5,
            0.75,
            0.3,
            0.4,
            0.8,
            0.5,
            0.75,
            0.3,
            0.4,
            0.8,
        ]
    )
    print(f"true p: {p}")
    dist = distributions.Bernoulli(p)

    def phi(z):
        return ~((z[:, 0] == 1) * (z[:, 1] == 1))

    num_samples = 10000

    S, alpha = helpers.generate_truncated_distribution_dataset(dist, phi, num_samples)
    emp_p = S.mean(0)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_p: {emp_p}")

    args = {
        "iterations": 2500,
        "trials": 1,
        "batch_size": 1,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-1,
    }

    truncated = truncated_boolean_product.TruncatedBooleanProduct(args, phi, alpha, dims)
    truncated.fit(S)

    best_p = truncated.best_p_
    print(f"best p:\n {best_p.T}")
    best_m = distributions.Bernoulli(best_p)

    ema_p = truncated.ema_p_
    print(f"ema p:\n {ema_p.T}")
    ema_m = distributions.Bernoulli(ema_p)
    ema_kl_div = kl.kl_divergence(ema_m, dist).sum()
    print(f"ema kl divergence: {ema_kl_div}")

    avg_p = truncated.avg_p_
    print(f"avg p:\n {avg_p.T}")
    avg_m = distributions.Bernoulli(avg_p)
    avg_kl_div = kl.kl_divergence(avg_m, dist).sum()
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_truncated = kl.kl_divergence(best_m, dist).sum()
    kl_emp = kl.kl_divergence(distributions.Bernoulli(emp_p), dist).sum()
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


def test_truncated_exponential():
    """1D truncated exponential recovers true rate."""
    ch.manual_seed(0)
    dims = 1
    lambda_ = ch.Tensor([1])
    print(f"true lambda: {lambda_}")
    dist = distributions.Exponential(lambda_)

    def phi(z):
        return z > 1

    num_samples = 10000
    S, alpha = helpers.generate_truncated_distribution_dataset(dist, phi, num_samples)

    emp_lambda = 1.0 / S.mean(0)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_lambda: {emp_lambda}")

    args = {
        "iterations": 2500,
        "trials": 1,
        "batch_size": 1,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-2,
        "max_phases": 1000000,
        "rate": 1.5,
    }

    truncated = truncated_exponential.TruncatedExponential(args, phi, alpha, dims)
    truncated.fit(S)

    best_lambda = truncated.best_lambda_
    print(f"best lambda:\n {best_lambda.T}")
    best_m = distributions.Exponential(best_lambda)

    ema_lambda = truncated.ema_lambda_
    print(f"ema lambda:\n {ema_lambda.T}")
    ema_m = distributions.Exponential(ema_lambda)
    ema_kl_div = kl.kl_divergence(ema_m, dist).sum()
    print(f"ema kl divergence: {ema_kl_div}")

    avg_lambda = truncated.avg_lambda_
    print(f"avg lambda:\n {avg_lambda.T}")
    avg_m = distributions.Exponential(avg_lambda)
    avg_kl_div = kl.kl_divergence(avg_m, dist).sum()
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_truncated = kl.kl_divergence(best_m, dist).sum()
    kl_emp = kl.kl_divergence(distributions.Exponential(emp_lambda), dist).sum()
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


def test_truncated_exponential_2_dims():
    """2D truncated exponential recovers true rates."""
    ch.manual_seed(0)
    dims = 2
    lambda_ = ch.Tensor([1, 2.0])
    print(f"true lambda: {lambda_}")
    dist = distributions.Exponential(lambda_)

    def phi(z):
        return z[:, 0] > 0.5

    num_samples = 10000
    S, alpha = helpers.generate_truncated_distribution_dataset(dist, phi, num_samples)

    emp_lambda = 1.0 / S.mean(0)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_lambda: {emp_lambda}")

    args = {
        "iterations": 2500,
        "trials": 1,
        "batch_size": 10,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-2,
    }

    truncated = truncated_exponential.TruncatedExponential(args, phi, alpha, dims)
    truncated.fit(S)

    best_lambda = truncated.best_lambda_
    print(f"best lambda:\n {best_lambda.T}")
    best_m = distributions.Exponential(best_lambda)

    ema_lambda = truncated.ema_lambda_
    print(f"ema lambda:\n {ema_lambda.T}")
    ema_m = distributions.Exponential(ema_lambda)
    ema_kl_div = kl.kl_divergence(ema_m, dist).sum()
    print(f"ema kl divergence: {ema_kl_div}")

    avg_lambda = truncated.avg_lambda_
    print(f"avg lambda:\n {avg_lambda.T}")
    avg_m = distributions.Exponential(avg_lambda)
    avg_kl_div = kl.kl_divergence(avg_m, dist).sum()
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_truncated = kl.kl_divergence(best_m, dist).sum()
    kl_emp = kl.kl_divergence(distributions.Exponential(emp_lambda), dist).sum()
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


def test_truncated_exponential_20_dims():
    """20D truncated exponential recovers true rates."""
    ch.manual_seed(0)
    dims = 20
    lambda_ = 2 * ch.ones(
        20,
    )
    print(f"true lambda: {lambda_}")
    dist = distributions.Exponential(lambda_)

    def phi(z):
        return z[:, 0] > 0.25

    num_samples = 10000
    S, alpha = helpers.generate_truncated_distribution_dataset(dist, phi, num_samples)

    emp_lambda = 1.0 / S.mean(0)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_lambda: {emp_lambda}")

    args = {
        "iterations": 1500,
        "trials": 1,
        "batch_size": 10,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-1,
    }

    truncated = truncated_exponential.TruncatedExponential(args, phi, alpha, dims)
    truncated.fit(S)

    best_lambda = truncated.best_lambda_
    print(f"best lambda:\n {best_lambda.T}")
    best_m = distributions.Exponential(best_lambda)

    ema_lambda = truncated.ema_lambda_
    print(f"ema lambda:\n {ema_lambda.T}")
    ema_m = distributions.Exponential(ema_lambda)
    ema_kl_div = kl.kl_divergence(ema_m, dist).sum()
    print(f"ema kl divergence: {ema_kl_div}")

    avg_lambda = truncated.avg_lambda_
    print(f"avg lambda:\n {avg_lambda.T}")
    avg_m = distributions.Exponential(avg_lambda)
    avg_kl_div = kl.kl_divergence(avg_m, dist).sum()
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_truncated = kl.kl_divergence(best_m, dist).sum()
    kl_emp = kl.kl_divergence(distributions.Exponential(emp_lambda), dist).sum()
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


def test_truncated_poisson():
    """1D truncated Poisson recovers true rate."""
    ch.manual_seed(0)
    dims = 1
    lambda_ = ch.Tensor([1])
    print(f"true lambda: {lambda_}")
    dist = distributions.Poisson(lambda_)

    def phi(z):
        return z > 1

    num_samples = 10000
    S, alpha = helpers.generate_truncated_distribution_dataset(dist, phi, num_samples)

    emp_lambda = S.mean(0)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_lambda: {emp_lambda}")

    args = {
        "iterations": 2500,
        "trials": 1,
        "batch_size": 1,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-2,
        "max_phases": 1000000,
        "rate": 1.5,
    }

    truncated = truncated_poisson.TruncatedPoisson(args, phi, alpha, dims)
    truncated.fit(S)

    best_lambda = truncated.best_lambda_
    print(f"best lambda:\n {best_lambda.T}")
    best_m = distributions.Poisson(best_lambda)

    ema_lambda = truncated.ema_lambda_
    print(f"ema lambda:\n {ema_lambda.T}")
    ema_m = distributions.Poisson(ema_lambda)
    ema_kl_div = kl.kl_divergence(ema_m, dist).sum()
    print(f"ema kl divergence: {ema_kl_div}")

    avg_lambda = truncated.avg_lambda_
    print(f"avg lambda:\n {avg_lambda.T}")
    avg_m = distributions.Poisson(avg_lambda)
    avg_kl_div = kl.kl_divergence(avg_m, dist).sum()
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_truncated = kl.kl_divergence(best_m, dist).sum()
    kl_emp = kl.kl_divergence(distributions.Poisson(emp_lambda), dist).sum()
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


def test_truncated_poisson_2_dims():
    """2D truncated Poisson recovers true rates."""
    ch.manual_seed(0)
    dims = 2
    lambda_ = ch.Tensor([1.0, 2.0])
    print(f"true lambda: {lambda_}")
    dist = distributions.Poisson(lambda_)

    def phi(z):
        return z[:, 0] > 0.25

    num_samples = 10000
    S, alpha = helpers.generate_truncated_distribution_dataset(dist, phi, num_samples)

    emp_lambda = S.mean(0)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_lambda: {emp_lambda}")

    args = {
        "iterations": 2500,
        "trials": 1,
        "batch_size": 1,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-2,
    }

    truncated = truncated_poisson.TruncatedPoisson(args, phi, alpha, dims)
    truncated.fit(S)

    best_lambda = truncated.best_lambda_
    print(f"best lambda:\n {best_lambda.T}")
    best_m = distributions.Poisson(best_lambda)

    ema_lambda = truncated.ema_lambda_
    print(f"ema lambda:\n {ema_lambda.T}")
    ema_m = distributions.Poisson(ema_lambda)
    ema_kl_div = kl.kl_divergence(ema_m, dist).sum()
    print(f"ema kl divergence: {ema_kl_div}")

    avg_lambda = truncated.avg_lambda_
    print(f"avg lambda:\n {avg_lambda.T}")
    avg_m = distributions.Poisson(avg_lambda)
    avg_kl_div = kl.kl_divergence(avg_m, dist).sum()
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_truncated = kl.kl_divergence(best_m, dist).sum()
    kl_emp = kl.kl_divergence(distributions.Poisson(emp_lambda), dist).sum()
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


def test_truncated_poisson_20_dims():
    """20D truncated Poisson recovers true rates."""
    ch.manual_seed(0)
    dims = 20
    lambda_ = 2 * ch.ones(
        20,
    )
    print(f"true lambda: {lambda_}")
    dist = distributions.Poisson(lambda_)

    def phi(z):
        return z[:, 0] > 1

    num_samples = 10000
    S, alpha = helpers.generate_truncated_distribution_dataset(dist, phi, num_samples)

    emp_lambda = S.mean(0)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_lambda: {emp_lambda}")

    args = {
        "iterations": 2500,
        "trials": 1,
        "batch_size": 10,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-2,
    }

    truncated = truncated_poisson.TruncatedPoisson(args, phi, alpha, dims)
    truncated.fit(S)

    best_lambda = truncated.best_lambda_
    print(f"best lambda:\n {best_lambda.T}")
    best_m = distributions.Poisson(best_lambda)

    ema_lambda = truncated.ema_lambda_
    print(f"ema lambda:\n {ema_lambda.T}")
    ema_m = distributions.Poisson(ema_lambda)
    ema_kl_div = kl.kl_divergence(ema_m, dist).sum()
    print(f"ema kl divergence: {ema_kl_div}")

    avg_lambda = truncated.avg_lambda_
    print(f"avg lambda:\n {avg_lambda.T}")
    avg_m = distributions.Poisson(avg_lambda)
    avg_kl_div = kl.kl_divergence(avg_m, dist).sum()
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_truncated = kl.kl_divergence(best_m, dist).sum()
    kl_emp = kl.kl_divergence(distributions.Poisson(emp_lambda), dist).sum()
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


def test_truncated_weibull():
    """1D truncated Weibull with known shape recovers true scale."""
    ch.manual_seed(0)
    dims = 1
    k = ch.Tensor([2.0])
    lambda_ = ch.Tensor([1])
    print(f"known k: {k}")
    print(f"true lambda: {lambda_}")

    dist = distributions.Weibull(lambda_, k)

    def phi(z):
        return z > 1.5

    num_samples = 10000
    S, alpha = helpers.generate_truncated_distribution_dataset(dist, phi, num_samples)

    emp_lambda_ = S.pow(k).mean(0).pow(1.0 / k)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_lambda: {emp_lambda_}")

    args = {
        "iterations": 1500,
        "trials": 1,
        "batch_size": 1,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-2,
        "max_phases": 1000000,
        "rate": 1.5,
    }

    truncated = truncated_weibull.TruncatedWeibull(args, phi, alpha, dims, k)
    truncated.fit(S)

    best_lambda = truncated.best_lambda_
    print(f"best lambda:\n {best_lambda.T}")
    best_l2_err = (best_lambda - lambda_).norm(p=2)
    print(f"truncated l2 error: {best_l2_err.item():.3f}")

    ema_lambda = truncated.ema_lambda_
    print(f"ema lambda:\n {ema_lambda.T}")
    ema_l2_err = (ema_lambda - lambda_).norm(p=2)
    print(f"ema l2 error: {ema_l2_err}")

    avg_lambda = truncated.avg_lambda_
    print(f"avg lambda:\n {avg_lambda.T}")
    avg_l2_err = (avg_lambda - lambda_).norm(p=2)
    print(f"avg l2 error: {avg_l2_err}")

    # check performance
    emp_l2_err = (emp_lambda_ - lambda_).norm(p=2)
    print(f"empirical l2 error: {emp_l2_err.item():.3f}")
    assert best_l2_err <= 1e-1, (
        f"L2 error to true distribution exceeds 0.1: {best_l2_err}"
    )


def test_truncated_weibull_2_dims():
    """2D truncated Weibull with uniform shape recovers true scales."""
    ch.manual_seed(0)
    dims = 2
    k = ch.Tensor([1.0, 1.0])
    lambda_ = ch.Tensor([1.0, 2.0])
    print(f"known k: {k}")
    print(f"true lambda: {lambda_}")

    dist = distributions.Weibull(lambda_, k)

    def phi(z):
        return z[:, 0] > 1.0

    num_samples = 10000
    S, alpha = helpers.generate_truncated_distribution_dataset(dist, phi, num_samples)

    emp_lambda_ = S.pow(k).mean(0).pow(1.0 / k)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_lambda: {emp_lambda_}")

    args = {
        "iterations": 2500,
        "trials": 1,
        "batch_size": 10,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-2,
    }

    truncated = truncated_weibull.TruncatedWeibull(args, phi, alpha, dims, k)
    truncated.fit(S)

    best_lambda = truncated.best_lambda_
    print(f"best lambda:\n {best_lambda.T}")
    best_l2_err = (best_lambda - lambda_).norm(p=2)
    print(f"truncated l2 error: {best_l2_err.item():.3f}")

    ema_lambda = truncated.ema_lambda_
    print(f"ema lambda:\n {ema_lambda.T}")
    ema_l2_err = (ema_lambda - lambda_).norm(p=2)
    print(f"ema l2 error: {ema_l2_err}")

    avg_lambda = truncated.avg_lambda_
    print(f"avg lambda:\n {avg_lambda.T}")
    avg_l2_err = (avg_lambda - lambda_).norm(p=2)
    print(f"avg l2 error: {avg_l2_err}")

    # check performance
    emp_l2_err = (emp_lambda_ - lambda_).norm(p=2)
    print(f"empirical l2 error: {emp_l2_err.item():.3f}")
    assert best_l2_err <= 1e-1, (
        f"L2 error to true distribution exceeds 0.1: {best_l2_err}"
    )


def test_truncated_weibull_2_dims_diff_scale():
    """2D truncated Weibull with mixed shapes recovers true scales."""
    ch.manual_seed(0)
    dims = 2
    k = ch.Tensor([2.0, 3.0])
    lambda_ = ch.Tensor([1.0, 2.0])
    print(f"known k: {k}")
    print(f"true lambda: {lambda_}")

    dist = distributions.Weibull(lambda_, k)

    def phi(z):
        return z[:, 0] > 1.0

    num_samples = 10000
    S, alpha = helpers.generate_truncated_distribution_dataset(dist, phi, num_samples)

    emp_lambda_ = S.pow(k).mean(0).pow(1.0 / k)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_lambda: {emp_lambda_}")

    args = {
        "iterations": 2500,
        "trials": 1,
        "batch_size": 100,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-2,
    }

    truncated = truncated_weibull.TruncatedWeibull(args, phi, alpha, dims, k)
    truncated.fit(S)

    best_lambda = truncated.best_lambda_
    print(f"best lambda:\n {best_lambda.T}")
    best_l2_err = (best_lambda - lambda_).norm(p=2)
    print(f"truncated l2 error: {best_l2_err.item():.3f}")

    ema_lambda = truncated.ema_lambda_
    print(f"ema lambda:\n {ema_lambda.T}")
    ema_l2_err = (ema_lambda - lambda_).norm(p=2)
    print(f"ema l2 error: {ema_l2_err}")

    avg_lambda = truncated.avg_lambda_
    print(f"avg lambda:\n {avg_lambda.T}")
    avg_l2_err = (avg_lambda - lambda_).norm(p=2)
    print(f"avg l2 error: {avg_l2_err}")

    # check performance
    emp_l2_err = (emp_lambda_ - lambda_).norm(p=2)
    print(f"empirical l2 error: {emp_l2_err.item():.3f}")
    assert best_l2_err <= 1e-1, (
        f"L2 error to true distribution exceeds 0.1: {best_l2_err}"
    )


def test_truncated_weibull_20_dims_diff_scale():
    """20D truncated Weibull with mixed shapes recovers true scales."""
    ch.manual_seed(0)
    dims = 20
    k = ch.Tensor([2.0, 3.0, 1.0, 2.0, 1.0]).repeat(4)
    lambda_ = ch.Tensor([1.0, 2.0, 3.0, 1.0, 5.0]).repeat(4)
    print(f"known k: {k}")
    print(f"true lambda: {lambda_}")

    dist = distributions.Weibull(lambda_, k)

    def phi(z):
        return z[:, 0] > 1.0

    num_samples = 10000
    S, alpha = helpers.generate_truncated_distribution_dataset(dist, phi, num_samples)

    emp_lambda_ = S.pow(k).mean(0).pow(1.0 / k)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_lambda: {emp_lambda_}")

    args = {
        "iterations": 2500,
        "trials": 1,
        "batch_size": 10,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-2,
    }

    truncated = truncated_weibull.TruncatedWeibull(args, phi, alpha, dims, k)
    truncated.fit(S)

    best_lambda = truncated.best_lambda_
    print(f"best lambda:\n {best_lambda.T}")
    best_l2_err = (best_lambda - lambda_).norm(p=2)
    print(f"truncated l2 error: {best_l2_err.item():.3f}")

    ema_lambda = truncated.ema_lambda_
    print(f"ema lambda:\n {ema_lambda.T}")
    ema_l2_err = (ema_lambda - lambda_).norm(p=2)
    print(f"ema l2 error: {ema_l2_err}")

    avg_lambda = truncated.avg_lambda_
    print(f"avg lambda:\n {avg_lambda.T}")
    avg_l2_err = (avg_lambda - lambda_).norm(p=2)
    print(f"avg l2 error: {avg_l2_err}")

    # check performance
    emp_l2_err = (emp_lambda_ - lambda_).norm(p=2)
    print(f"empirical l2 error: {emp_l2_err.item():.3f}")
    assert best_l2_err <= 1e-1, (
        f"L2 error to true distribution exceeds 0.1: {best_l2_err}"
    )
