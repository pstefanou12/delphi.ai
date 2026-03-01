# Author: pstefanou12@
"""Tests for truncated exponential family distributions."""

import torch as ch
from torch import Tensor
from torch.distributions import (
    MultivariateNormal,
    Uniform,
    Bernoulli,
    Exponential,
    Poisson,
    Weibull,
)
from torch.distributions.kl import kl_divergence

from delphi import oracle
from delphi.truncated.distributions.truncated_normal import TruncatedNormal
from delphi.truncated.distributions.truncated_multivariate_normal_known_covariance import (
    TruncatedMultivariateNormalKnownCovariance,
)
from delphi.truncated.distributions.truncated_multivariate_normal import (
    TruncatedMultivariateNormal,
)
from delphi.truncated.distributions.unknown_truncated_normal import (
    UnknownTruncationNormal,
)
from delphi.truncated.distributions.unknown_truncated_multivariate_normal import (
    UnknownTruncationMultivariateNormal,
)
from delphi.truncated.distributions.truncated_boolean_product import (
    TruncatedBooleanProduct,
)
from delphi.truncated.distributions.truncated_exponential import TruncatedExponential
from delphi.truncated.distributions.truncated_poisson import TruncatedPoisson
from delphi.truncated.distributions.truncated_weibull import TruncatedWeibull
from delphi.utils.helpers import cov

ch.set_printoptions(precision=4, sci_mode=False)


def generate_truncated_dataset(dist, phi, num_samples: int):
    """Draw samples from dist that satisfy phi.

    Repeatedly draws batches from dist, keeps samples where phi returns
    True, and stops once num_samples accepted samples have been collected.

    Args:
        dist: Source distribution with a .sample() method.
        phi: Truncation oracle; returns a boolean mask over a batch.
        num_samples: Number of accepted samples to return.

    Returns:
        Tuple of (S, alpha) where S is a tensor of shape (num_samples, ...)
        and alpha is the empirical acceptance rate.
    """
    num_accepted, num_sampled = 0, 0

    S = []
    while num_accepted < num_samples:
        samples = dist.sample([num_samples])
        indices = phi(samples).nonzero()[:, 0]
        S.append(samples[indices])
        num_accepted += indices.size(0)
        num_sampled += num_samples
    S = ch.cat(S)[:num_samples]
    alpha = num_accepted / num_sampled

    return S, alpha


def test_truncated_normal_known_variance():
    """Right-truncated 1D normal with known variance converges to true mean."""
    ch.manual_seed(0)
    M = MultivariateNormal(ch.zeros(1), ch.eye(1))
    phi = oracle.Left_Distribution(Tensor([0.0]))
    num_samples = 1000
    S, alpha = generate_truncated_dataset(M, phi, num_samples)
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
    truncated = TruncatedNormal(args, phi_std_norm, alpha, 1, variance=ch.eye(1))
    truncated.fit(S_std_norm)

    best_loc = truncated.best_loc_ + emp_loc
    print(f"best loc:\n {best_loc}")
    best_m = MultivariateNormal(best_loc, ch.eye(1))
    kl_truncated = kl_divergence(best_m, M)
    print(f"truncated kl divergence: {kl_truncated.item():.3f}")

    ema_loc = truncated.ema_loc_ + emp_loc
    print(f"ema loc:\n {ema_loc}")
    ema_m = MultivariateNormal(ema_loc, ch.eye(1))
    ema_kl_div = kl_divergence(ema_m, M)
    print(f"ema kl divergence: {ema_kl_div}")

    avg_loc = truncated.avg_loc_ + emp_loc
    print(f"avg loc:\n {avg_loc}")
    avg_m = MultivariateNormal(avg_loc, ch.eye(1))
    avg_kl_div = kl_divergence(avg_m, M)
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_emp = kl_divergence(MultivariateNormal(emp_loc, M.covariance_matrix), M)
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
    M = MultivariateNormal(ch.zeros(1), ch.eye(1))

    phi = oracle.Left_Distribution(Tensor([0.0]))
    num_samples = 1000
    S, alpha = generate_truncated_dataset(M, phi, num_samples)
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
    truncated = TruncatedNormal(args, phi_std_norm, alpha, 1)
    truncated.fit(S_std_norm)

    best_loc = truncated.best_loc_ * emp_scale + emp_loc
    best_covariance_matrix = truncated.best_covariance_matrix_ * emp_var
    print(f"best loc:\n {best_loc}")
    print(f"best covariance matrix:\n {best_covariance_matrix}")
    best_m = MultivariateNormal(best_loc, best_covariance_matrix)
    kl_truncated = kl_divergence(best_m, M)
    print(f"truncated kl divergence: {kl_truncated.item():.3f}")

    ema_loc = truncated.ema_loc_ * emp_scale + emp_loc
    ema_covariance_matrix = truncated.ema_covariance_matrix_ * emp_var
    print(f"ema loc:\n {ema_loc}")
    print(f"ema covariance matrix:\n {ema_covariance_matrix}")
    ema_m = MultivariateNormal(ema_loc, ema_covariance_matrix)
    ema_kl_div = kl_divergence(ema_m, M)
    print(f"ema kl divergence: {ema_kl_div}")

    avg_loc = truncated.avg_loc_ * emp_scale + emp_loc
    avg_covariance_matrix = truncated.avg_covariance_matrix_ * emp_var
    print(f"avg loc:\n {avg_loc}")
    print(f"avg covariance matrix:\n {avg_covariance_matrix}")
    avg_m = MultivariateNormal(avg_loc, avg_covariance_matrix)
    avg_kl_div = kl_divergence(avg_m, M)
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_emp = kl_divergence(MultivariateNormal(emp_loc, M.covariance_matrix), M)
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
    M = MultivariateNormal(ch.zeros(dims), 2 * ch.eye(dims))

    num_samples = 5000

    def phi(x):
        return x[:, 0] > 0

    S, alpha = generate_truncated_dataset(M, phi, num_samples)

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
    truncated = TruncatedMultivariateNormalKnownCovariance(
        args, phi, alpha, dims, M.covariance_matrix
    )
    truncated.fit(S)

    best_loc = truncated.best_loc_
    print(f"best loc:\n {best_loc.T}")
    best_m = MultivariateNormal(best_loc, M.covariance_matrix)
    kl_truncated = kl_divergence(best_m, M)
    print(f"truncated kl divergence: {kl_truncated.item():.3f}")

    ema_loc = truncated.ema_loc_
    print(f"ema loc:\n {ema_loc.T}")
    ema_m = MultivariateNormal(ema_loc, M.covariance_matrix)
    ema_kl_div = kl_divergence(ema_m, M)
    print(f"ema kl divergence: {ema_kl_div}")

    avg_loc = truncated.avg_loc_
    print(f"avg loc:\n {avg_loc.T}")
    avg_m = MultivariateNormal(avg_loc, M.covariance_matrix)
    avg_kl_div = kl_divergence(avg_m, M)
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_emp = kl_divergence(MultivariateNormal(emp_loc, M.covariance_matrix), M)
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
    M = MultivariateNormal(ch.zeros(dims), 2 * ch.eye(dims))
    num_samples = 1000

    def phi(x):
        return x[:, 0] > 0

    S, alpha = generate_truncated_dataset(M, phi, num_samples)

    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")

    emp_loc = S.mean(0, keepdim=True)
    emp_covariance_matrix = cov(S)
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
    truncated = TruncatedMultivariateNormal(args, phi_std_norm, alpha, dims)
    truncated.fit(S_std_norm)
    best_loc = truncated.best_loc_ * ch.sqrt(emp_var) + emp_loc
    best_covariance_matrix = (
        emp_sigma_diag @ truncated.best_covariance_matrix_ @ emp_sigma_diag
    )
    print(f"best loc:\n {best_loc.T}")
    print(f"best covariance matrix:\n {best_covariance_matrix}")
    best_m = MultivariateNormal(best_loc, best_covariance_matrix)
    kl_truncated = kl_divergence(best_m, M)
    print(f"truncated kl divergence: {kl_truncated.item():.3f}")

    ema_loc = truncated.ema_loc_ * ch.sqrt(emp_var) + emp_loc
    ema_covariance_matrix = (
        emp_sigma_diag @ truncated.ema_covariance_matrix_ @ emp_sigma_diag
    )
    print(f"ema loc:\n {ema_loc.T}")
    print(f"ema covariance matrix:\n {ema_covariance_matrix}")
    ema_m = MultivariateNormal(ema_loc, ema_covariance_matrix)
    ema_kl_div = kl_divergence(ema_m, M)
    print(f"ema kl divergence: {ema_kl_div}")

    avg_loc = truncated.avg_loc_ * ch.sqrt(emp_var) + emp_loc
    avg_covariance_matrix = (
        emp_sigma_diag @ truncated.avg_covariance_matrix_ @ emp_sigma_diag
    )
    print(f"avg loc:\n {avg_loc.T}")
    print(f"avg covariance matrix:\n {avg_covariance_matrix}")
    avg_m = MultivariateNormal(avg_loc, avg_covariance_matrix)
    avg_kl_div = kl_divergence(avg_m, M)
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_emp = kl_divergence(MultivariateNormal(emp_loc, emp_covariance_matrix), M)
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
    M = MultivariateNormal(ch.zeros(dims), 10 * ch.eye(dims))

    def phi(x):
        return x[:, 0] > 0

    num_samples = 1000
    S, alpha = generate_truncated_dataset(M, phi, num_samples)
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
    truncated = TruncatedMultivariateNormalKnownCovariance(
        args, phi_std_norm, alpha, dims, scaled_cov
    )
    truncated.fit(S_std_norm)
    best_loc = truncated.best_loc_ * ch.sqrt(true_var) + emp_loc
    print(f"best loc:\n {best_loc.T}")
    best_m = MultivariateNormal(best_loc, M.covariance_matrix)
    kl_truncated = kl_divergence(best_m, M)
    print(f"truncated kl divergence: {kl_truncated.item():.3f}")

    ema_loc = truncated.ema_loc_ * ch.sqrt(true_var) + emp_loc
    print(f"ema loc:\n {ema_loc.T}")
    ema_m = MultivariateNormal(ema_loc, M.covariance_matrix)
    ema_kl_div = kl_divergence(ema_m, M)
    print(f"ema kl divergence: {ema_kl_div}")

    avg_loc = truncated.avg_loc_ * ch.sqrt(true_var) + emp_loc
    print(f"avg loc:\n {avg_loc.T}")
    avg_m = MultivariateNormal(avg_loc, M.covariance_matrix)
    avg_kl_div = kl_divergence(avg_m, M)
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_emp = kl_divergence(MultivariateNormal(emp_loc, M.covariance_matrix), M)
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
    M = MultivariateNormal(ch.zeros(dims), 10 * ch.eye(dims))

    def phi(x):
        return x[:, 0] > 0

    num_samples = 5000
    S, alpha = generate_truncated_dataset(M, phi, num_samples)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")

    emp_loc = S.mean(0, keepdim=True)
    emp_covariance_matrix = cov(S)
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
    truncated = TruncatedMultivariateNormal(args, phi_std_norm, alpha, dims)
    truncated.fit(S_std_norm)

    best_loc = truncated.best_loc_ * ch.sqrt(emp_var) + emp_loc
    best_covariance_matrix = (
        emp_sigma_diag @ truncated.best_covariance_matrix_ @ emp_sigma_diag
    )
    print(f"best loc:\n {best_loc.T}")
    print(f"best covariance matrix:\n {best_covariance_matrix}")
    best_m = MultivariateNormal(best_loc, best_covariance_matrix)
    kl_truncated = kl_divergence(best_m, M)
    print(f"truncated kl divergence: {kl_truncated.item():.3f}")

    ema_loc = truncated.ema_loc_ * ch.sqrt(emp_var) + emp_loc
    ema_covariance_matrix = (
        emp_sigma_diag @ truncated.ema_covariance_matrix_ @ emp_sigma_diag
    )
    print(f"ema loc:\n {ema_loc.T}")
    print(f"ema covariance matrix:\n {ema_covariance_matrix}")
    ema_m = MultivariateNormal(ema_loc, ema_covariance_matrix)
    ema_kl_div = kl_divergence(ema_m, M)
    print(f"ema kl divergence: {ema_kl_div}")

    avg_loc = truncated.avg_loc_ * ch.sqrt(emp_var) + emp_loc
    avg_covariance_matrix = (
        emp_sigma_diag @ truncated.avg_covariance_matrix_ @ emp_sigma_diag
    )
    print(f"avg loc:\n {avg_loc.T}")
    print(f"avg covariance matrix:\n {avg_covariance_matrix}")
    avg_m = MultivariateNormal(avg_loc, avg_covariance_matrix)
    avg_kl_div = kl_divergence(avg_m, M)
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_emp = kl_divergence(MultivariateNormal(emp_loc, emp_covariance_matrix), M)
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


def test_unknown_truncation_normal_known_variance():
    """1D normal with unknown truncation and known variance recovers true mean."""
    ch.manual_seed(0)
    dims = 1
    M = MultivariateNormal(ch.zeros(dims), ch.eye(dims))

    phi = oracle.Left_Distribution(Tensor([0.0]))
    num_samples = 10000
    S, alpha = generate_truncated_dataset(M, phi, num_samples)
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
    truncated = UnknownTruncationNormal(args, k, alpha, dims, variance=true_cov_std)
    truncated.fit(S_std_norm)
    best_loc = truncated.best_loc_ * ch.sqrt(emp_var) + emp_loc
    print(f"best loc:\n {best_loc.T}")
    best_m = MultivariateNormal(best_loc, M.covariance_matrix)
    kl_truncated = kl_divergence(best_m, M)
    print(f"truncated kl divergence: {kl_truncated.item():.3f}")

    ema_loc = truncated.ema_loc_ * ch.sqrt(emp_var) + emp_loc
    print(f"ema loc:\n {ema_loc.T}")
    ema_m = MultivariateNormal(ema_loc, M.covariance_matrix)
    ema_kl_div = kl_divergence(ema_m, M)
    print(f"ema kl divergence: {ema_kl_div}")

    avg_loc = truncated.avg_loc_ * ch.sqrt(emp_var) + emp_loc
    print(f"avg loc:\n {avg_loc.T}")
    avg_m = MultivariateNormal(avg_loc, M.covariance_matrix)
    avg_kl_div = kl_divergence(avg_m, M)
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_emp = kl_divergence(MultivariateNormal(emp_loc, M.covariance_matrix), M)
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
    M = MultivariateNormal(ch.zeros(dims), ch.eye(dims))
    phi = oracle.Right_Distribution(Tensor([0.0]))
    num_samples = 10000
    S, alpha = generate_truncated_dataset(M, phi, num_samples)
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
    truncated = UnknownTruncationNormal(args, k, alpha, dims)
    truncated.fit(S_std_norm)
    # rescale distribution
    best_loc = truncated.best_loc_ * emp_scale + emp_loc
    best_variance = truncated.best_variance_ * emp_var
    print(f"best loc:\n {best_loc.T}")
    print(f"best variance:\n {best_variance}")
    best_m = MultivariateNormal(best_loc, best_variance)

    ema_loc = truncated.ema_loc_ * emp_scale + emp_loc
    ema_variance = truncated.ema_variance_ * emp_var
    print(f"ema loc:\n {ema_loc.T}")
    print(f"ema variance:\n {ema_variance}")
    ema_m = MultivariateNormal(ema_loc, ema_variance)
    ema_kl_div = kl_divergence(ema_m, M)
    print(f"ema kl divergence: {ema_kl_div}")

    avg_loc = truncated.avg_loc_ * emp_scale + emp_loc
    avg_variance = truncated.avg_variance_ * emp_var
    print(f"avg loc:\n {avg_loc.T}")
    print(f"avg variance:\n {avg_variance}")
    avg_m = MultivariateNormal(avg_loc, avg_variance)
    avg_kl_div = kl_divergence(avg_m, M)
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_truncated = kl_divergence(best_m, M)
    kl_emp = kl_divergence(MultivariateNormal(emp_loc, emp_var), M)
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
    M = MultivariateNormal(ch.zeros(dims), ch.eye(dims))

    W = Uniform(-0.5, 0.5)
    centroid = W.sample([dims])
    phi = oracle.Sphere(M.covariance_matrix, centroid, 3.5)

    num_samples = 5000
    S, alpha = generate_truncated_dataset(M, phi, num_samples)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")

    emp_loc = S.mean(0)
    emp_cov = cov(S)
    L = ch.linalg.cholesky(emp_cov)  # pylint: disable=not-callable
    L_inv = ch.linalg.inv(L)  # pylint: disable=not-callable
    S_white = (S - emp_loc) @ L_inv.T

    k = 12
    args = {
        "epochs": 25,
        "batch_size": 100,
        "verbose": True,
    }
    truncated = UnknownTruncationMultivariateNormal(args, k, alpha, dims)
    truncated.fit(S_white)

    best_loc = truncated.best_loc_ @ L.T + emp_loc
    best_covariance_matrix = L @ truncated.best_covariance_matrix_ @ L.T
    print(f"best loc:\n {best_loc.T}")
    print(f"best covariance matrix:\n {best_covariance_matrix}")
    best_m = MultivariateNormal(best_loc, best_covariance_matrix)
    kl_truncated = kl_divergence(best_m, M)
    print(f"truncated kl divergence: {kl_truncated.item():.3f}")

    ema_loc = truncated.ema_loc_ @ L.T + emp_loc
    ema_covariance_matrix = L @ truncated.ema_covariance_matrix_ @ L.T
    print(f"ema loc:\n {ema_loc.T}")
    print(f"ema covariance matrix:\n {ema_covariance_matrix}")
    ema_m = MultivariateNormal(ema_loc, ema_covariance_matrix)
    ema_kl_div = kl_divergence(ema_m, M)
    print(f"ema kl divergence: {ema_kl_div}")

    avg_loc = truncated.avg_loc_ @ L.T + emp_loc
    avg_covariance_matrix = L @ truncated.avg_covariance_matrix_ @ L.T
    print(f"avg loc:\n {avg_loc.T}")
    print(f"avg covariance matrix:\n {avg_covariance_matrix}")
    avg_m = MultivariateNormal(avg_loc, avg_covariance_matrix)
    avg_kl_div = kl_divergence(avg_m, M)
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_emp = kl_divergence(MultivariateNormal(emp_loc, emp_cov), M)
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
    dist = Bernoulli(p)

    def phi(z):
        return ~((z[:, 0] == 1) * (z[:, 1] == 1))

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

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
    }

    truncated = TruncatedBooleanProduct(args, phi, alpha, dims)
    truncated.fit(S)

    best_p = truncated.best_p_
    print(f"best p:\n {best_p.T}")
    best_m = Bernoulli(best_p)

    ema_p = truncated.ema_p_
    print(f"ema p:\n {ema_p.T}")
    ema_m = Bernoulli(ema_p)
    ema_kl_div = kl_divergence(ema_m, dist).sum()
    print(f"ema kl divergence: {ema_kl_div}")

    avg_p = truncated.avg_p_
    print(f"avg p:\n {avg_p.T}")
    avg_m = Bernoulli(avg_p)
    avg_kl_div = kl_divergence(avg_m, dist).sum()
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_truncated = kl_divergence(best_m, dist).sum()
    kl_emp = kl_divergence(Bernoulli(emp_p), dist).sum()
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
    dist = Bernoulli(p)

    def phi(z):
        return ~((z[:, 0] == 1) * (z[:, 1] == 1))

    num_samples = 10000

    S, alpha = generate_truncated_dataset(dist, phi, num_samples)
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

    truncated = TruncatedBooleanProduct(args, phi, alpha, dims)
    truncated.fit(S)

    best_p = truncated.best_p_
    print(f"best p:\n {best_p.T}")
    best_m = Bernoulli(best_p)

    ema_p = truncated.ema_p_
    print(f"ema p:\n {ema_p.T}")
    ema_m = Bernoulli(ema_p)
    ema_kl_div = kl_divergence(ema_m, dist).sum()
    print(f"ema kl divergence: {ema_kl_div}")

    avg_p = truncated.avg_p_
    print(f"avg p:\n {avg_p.T}")
    avg_m = Bernoulli(avg_p)
    avg_kl_div = kl_divergence(avg_m, dist).sum()
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_truncated = kl_divergence(best_m, dist).sum()
    kl_emp = kl_divergence(Bernoulli(emp_p), dist).sum()
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
    dist = Exponential(lambda_)

    def phi(z):
        return z > 1

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

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

    truncated = TruncatedExponential(args, phi, alpha, dims)
    truncated.fit(S)

    best_lambda = truncated.best_lambda_
    print(f"best lambda:\n {best_lambda.T}")
    best_m = Exponential(best_lambda)

    ema_lambda = truncated.ema_lambda_
    print(f"ema lambda:\n {ema_lambda.T}")
    ema_m = Exponential(ema_lambda)
    ema_kl_div = kl_divergence(ema_m, dist).sum()
    print(f"ema kl divergence: {ema_kl_div}")

    avg_lambda = truncated.avg_lambda_
    print(f"avg lambda:\n {avg_lambda.T}")
    avg_m = Exponential(avg_lambda)
    avg_kl_div = kl_divergence(avg_m, dist).sum()
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_truncated = kl_divergence(best_m, dist).sum()
    kl_emp = kl_divergence(Exponential(emp_lambda), dist).sum()
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
    dist = Exponential(lambda_)

    def phi(z):
        return z[:, 0] > 0.5

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

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

    truncated = TruncatedExponential(args, phi, alpha, dims)
    truncated.fit(S)

    best_lambda = truncated.best_lambda_
    print(f"best lambda:\n {best_lambda.T}")
    best_m = Exponential(best_lambda)

    ema_lambda = truncated.ema_lambda_
    print(f"ema lambda:\n {ema_lambda.T}")
    ema_m = Exponential(ema_lambda)
    ema_kl_div = kl_divergence(ema_m, dist).sum()
    print(f"ema kl divergence: {ema_kl_div}")

    avg_lambda = truncated.avg_lambda_
    print(f"avg lambda:\n {avg_lambda.T}")
    avg_m = Exponential(avg_lambda)
    avg_kl_div = kl_divergence(avg_m, dist).sum()
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_truncated = kl_divergence(best_m, dist).sum()
    kl_emp = kl_divergence(Exponential(emp_lambda), dist).sum()
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
    dist = Exponential(lambda_)

    def phi(z):
        return z[:, 0] > 0.25

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

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

    truncated = TruncatedExponential(args, phi, alpha, dims)
    truncated.fit(S)

    best_lambda = truncated.best_lambda_
    print(f"best lambda:\n {best_lambda.T}")
    best_m = Exponential(best_lambda)

    ema_lambda = truncated.ema_lambda_
    print(f"ema lambda:\n {ema_lambda.T}")
    ema_m = Exponential(ema_lambda)
    ema_kl_div = kl_divergence(ema_m, dist).sum()
    print(f"ema kl divergence: {ema_kl_div}")

    avg_lambda = truncated.avg_lambda_
    print(f"avg lambda:\n {avg_lambda.T}")
    avg_m = Exponential(avg_lambda)
    avg_kl_div = kl_divergence(avg_m, dist).sum()
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_truncated = kl_divergence(best_m, dist).sum()
    kl_emp = kl_divergence(Exponential(emp_lambda), dist).sum()
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
    dist = Poisson(lambda_)

    def phi(z):
        return z > 1

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

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

    truncated = TruncatedPoisson(args, phi, alpha, dims)
    truncated.fit(S)

    best_lambda = truncated.best_lambda_
    print(f"best lambda:\n {best_lambda.T}")
    best_m = Poisson(best_lambda)

    ema_lambda = truncated.ema_lambda_
    print(f"ema lambda:\n {ema_lambda.T}")
    ema_m = Poisson(ema_lambda)
    ema_kl_div = kl_divergence(ema_m, dist).sum()
    print(f"ema kl divergence: {ema_kl_div}")

    avg_lambda = truncated.avg_lambda_
    print(f"avg lambda:\n {avg_lambda.T}")
    avg_m = Poisson(avg_lambda)
    avg_kl_div = kl_divergence(avg_m, dist).sum()
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_truncated = kl_divergence(best_m, dist).sum()
    kl_emp = kl_divergence(Poisson(emp_lambda), dist).sum()
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
    dist = Poisson(lambda_)

    def phi(z):
        return z[:, 0] > 0.25

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

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

    truncated = TruncatedPoisson(args, phi, alpha, dims)
    truncated.fit(S)

    best_lambda = truncated.best_lambda_
    print(f"best lambda:\n {best_lambda.T}")
    best_m = Poisson(best_lambda)

    ema_lambda = truncated.ema_lambda_
    print(f"ema lambda:\n {ema_lambda.T}")
    ema_m = Poisson(ema_lambda)
    ema_kl_div = kl_divergence(ema_m, dist).sum()
    print(f"ema kl divergence: {ema_kl_div}")

    avg_lambda = truncated.avg_lambda_
    print(f"avg lambda:\n {avg_lambda.T}")
    avg_m = Poisson(avg_lambda)
    avg_kl_div = kl_divergence(avg_m, dist).sum()
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_truncated = kl_divergence(best_m, dist).sum()
    kl_emp = kl_divergence(Poisson(emp_lambda), dist).sum()
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
    dist = Poisson(lambda_)

    def phi(z):
        return z[:, 0] > 1

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

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

    truncated = TruncatedPoisson(args, phi, alpha, dims)
    truncated.fit(S)

    best_lambda = truncated.best_lambda_
    print(f"best lambda:\n {best_lambda.T}")
    best_m = Poisson(best_lambda)

    ema_lambda = truncated.ema_lambda_
    print(f"ema lambda:\n {ema_lambda.T}")
    ema_m = Poisson(ema_lambda)
    ema_kl_div = kl_divergence(ema_m, dist).sum()
    print(f"ema kl divergence: {ema_kl_div}")

    avg_lambda = truncated.avg_lambda_
    print(f"avg lambda:\n {avg_lambda.T}")
    avg_m = Poisson(avg_lambda)
    avg_kl_div = kl_divergence(avg_m, dist).sum()
    print(f"avg kl divergence: {avg_kl_div}")

    # check performance
    kl_truncated = kl_divergence(best_m, dist).sum()
    kl_emp = kl_divergence(Poisson(emp_lambda), dist).sum()
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

    dist = Weibull(lambda_, k)

    def phi(z):
        return z > 1.5

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

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

    truncated = TruncatedWeibull(args, phi, alpha, dims, k)
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

    dist = Weibull(lambda_, k)

    def phi(z):
        return z[:, 0] > 1.0

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

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

    truncated = TruncatedWeibull(args, phi, alpha, dims, k)
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

    dist = Weibull(lambda_, k)

    def phi(z):
        return z[:, 0] > 1.0

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

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

    truncated = TruncatedWeibull(args, phi, alpha, dims, k)
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

    dist = Weibull(lambda_, k)

    def phi(z):
        return z[:, 0] > 1.0

    num_samples = 10000
    S, alpha = generate_truncated_dataset(dist, phi, num_samples)

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

    truncated = TruncatedWeibull(args, phi, alpha, dims, k)
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
