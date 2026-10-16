# Author: pstefanou12@
"""Tests for the truncated boolean product distribution."""

import torch as ch
from torch import distributions
from torch.distributions import kl

from delphi.truncated.distributions import truncated_boolean_product
from delphi.truncated.distributions.tests import helpers

ch.set_printoptions(precision=4, sci_mode=False)


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
    S, alpha, parent = helpers.generate_truncated_distribution_dataset(
        dist, phi, num_samples
    )

    emp_p = S.mean(0)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_p: {emp_p}")

    hyperparameters = {
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

    truncated = truncated_boolean_product.TruncatedBooleanProduct(
        hyperparameters, phi, dims, alpha
    )
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

    S, alpha, parent = helpers.generate_truncated_distribution_dataset(
        dist, phi, num_samples
    )
    emp_p = S.mean(0)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_p: {emp_p}")

    hyperparameters = {
        "iterations": 2500,
        "trials": 1,
        "batch_size": 1,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-1,
    }

    truncated = truncated_boolean_product.TruncatedBooleanProduct(
        hyperparameters, phi, dims, alpha
    )
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
