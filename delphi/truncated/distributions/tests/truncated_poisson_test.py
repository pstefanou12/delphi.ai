# Author: pstefanou12@
"""Tests for the truncated Poisson distribution."""

import torch as ch
from torch import distributions
from torch.distributions import kl

from delphi.truncated.distributions import truncated_poisson
from delphi.truncated.distributions.tests import helpers

ch.set_printoptions(precision=4, sci_mode=False)


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
    S, alpha, parent = helpers.generate_truncated_distribution_dataset(
        dist, phi, num_samples
    )

    emp_lambda = S.mean(0)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_lambda: {emp_lambda}")

    hyperparameters = {
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

    truncated = truncated_poisson.TruncatedPoisson(hyperparameters, phi, dims, alpha)
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
    S, alpha, parent = helpers.generate_truncated_distribution_dataset(
        dist, phi, num_samples
    )

    emp_lambda = S.mean(0)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_lambda: {emp_lambda}")

    hyperparameters = {
        "iterations": 2500,
        "trials": 1,
        "batch_size": 1,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-2,
    }

    truncated = truncated_poisson.TruncatedPoisson(hyperparameters, phi, dims, alpha)
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
    S, alpha, parent = helpers.generate_truncated_distribution_dataset(
        dist, phi, num_samples
    )

    emp_lambda = S.mean(0)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_lambda: {emp_lambda}")

    hyperparameters = {
        "iterations": 2500,
        "trials": 1,
        "batch_size": 10,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-2,
    }

    truncated = truncated_poisson.TruncatedPoisson(hyperparameters, phi, dims, alpha)
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
