# Author: pstefanou12@
"""Tests for the truncated Weibull distribution."""

import functools

import torch as ch
from torch import distributions

from delphi.truncated.distributions import truncated_weibull
from delphi.truncated.distributions.tests import helpers

ch.set_printoptions(precision=4, sci_mode=False)


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
    S, alpha, parent = helpers.generate_truncated_distribution_dataset(
        dist, phi, num_samples
    )

    emp_lambda_ = S.pow(k).mean(0).pow(1.0 / k)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_lambda: {emp_lambda_}")

    hyperparameters = {
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

    truncated = truncated_weibull.TruncatedWeibull(hyperparameters, phi, dims, k, alpha)
    truncated.fit(S)

    # The untruncated MLE is what the estimator is asked to recover; comparing
    # against the true lambda instead would charge it for sampling error that
    # no estimator can remove.
    mle = parent.pow(k).mean(0).pow(1.0 / k)
    kl_of = functools.partial(
        helpers.weibull_kl_divergence, other_scale=mle, concentration=k
    )

    best_kl = kl_of(truncated.best_lambda_.flatten()).sum()
    ema_kl = kl_of(truncated.ema_lambda_.flatten()).sum()
    avg_kl = kl_of(truncated.avg_lambda_.flatten()).sum()
    emp_kl = kl_of(emp_lambda_.flatten()).sum()
    print(f"empirical kl divergence: {emp_kl.item():.4f}")
    print(f"best kl divergence: {best_kl.item():.4f}")
    print(f"ema kl divergence: {ema_kl.item():.4f}")
    print(f"avg kl divergence: {avg_kl.item():.4f}")
    assert ema_kl <= 1e-1, (
        f"EMA KL divergence to the untruncated MLE exceeds 0.1: {ema_kl}"
    )
    assert avg_kl <= 0.15, (
        f"Average KL divergence to the untruncated MLE exceeds 0.15: {avg_kl}"
    )
    assert best_kl <= 0.2, (
        f"Best KL divergence to the untruncated MLE exceeds 0.2: {best_kl}"
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
    S, alpha, parent = helpers.generate_truncated_distribution_dataset(
        dist, phi, num_samples
    )

    emp_lambda_ = S.pow(k).mean(0).pow(1.0 / k)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_lambda: {emp_lambda_}")

    hyperparameters = {
        "iterations": 2500,
        "trials": 1,
        "batch_size": 10,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-2,
    }

    truncated = truncated_weibull.TruncatedWeibull(hyperparameters, phi, dims, k, alpha)
    truncated.fit(S)

    # The untruncated MLE is what the estimator is asked to recover; comparing
    # against the true lambda instead would charge it for sampling error that
    # no estimator can remove.
    mle = parent.pow(k).mean(0).pow(1.0 / k)
    kl_of = functools.partial(
        helpers.weibull_kl_divergence, other_scale=mle, concentration=k
    )

    best_kl = kl_of(truncated.best_lambda_.flatten()).sum()
    ema_kl = kl_of(truncated.ema_lambda_.flatten()).sum()
    avg_kl = kl_of(truncated.avg_lambda_.flatten()).sum()
    emp_kl = kl_of(emp_lambda_.flatten()).sum()
    print(f"empirical kl divergence: {emp_kl.item():.4f}")
    print(f"best kl divergence: {best_kl.item():.4f}")
    print(f"ema kl divergence: {ema_kl.item():.4f}")
    print(f"avg kl divergence: {avg_kl.item():.4f}")
    assert ema_kl <= 1e-1, (
        f"EMA KL divergence to the untruncated MLE exceeds 0.1: {ema_kl}"
    )
    assert avg_kl <= 0.15, (
        f"Average KL divergence to the untruncated MLE exceeds 0.15: {avg_kl}"
    )
    assert best_kl <= 0.2, (
        f"Best KL divergence to the untruncated MLE exceeds 0.2: {best_kl}"
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
    S, alpha, parent = helpers.generate_truncated_distribution_dataset(
        dist, phi, num_samples
    )

    emp_lambda_ = S.pow(k).mean(0).pow(1.0 / k)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_lambda: {emp_lambda_}")

    hyperparameters = {
        "iterations": 2500,
        "trials": 1,
        "batch_size": 100,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-2,
    }

    truncated = truncated_weibull.TruncatedWeibull(hyperparameters, phi, dims, k, alpha)
    truncated.fit(S)

    # The untruncated MLE is what the estimator is asked to recover; comparing
    # against the true lambda instead would charge it for sampling error that
    # no estimator can remove.
    mle = parent.pow(k).mean(0).pow(1.0 / k)
    kl_of = functools.partial(
        helpers.weibull_kl_divergence, other_scale=mle, concentration=k
    )

    best_kl = kl_of(truncated.best_lambda_.flatten()).sum()
    ema_kl = kl_of(truncated.ema_lambda_.flatten()).sum()
    avg_kl = kl_of(truncated.avg_lambda_.flatten()).sum()
    emp_kl = kl_of(emp_lambda_.flatten()).sum()
    print(f"empirical kl divergence: {emp_kl.item():.4f}")
    print(f"best kl divergence: {best_kl.item():.4f}")
    print(f"ema kl divergence: {ema_kl.item():.4f}")
    print(f"avg kl divergence: {avg_kl.item():.4f}")
    assert ema_kl <= 1e-1, (
        f"EMA KL divergence to the untruncated MLE exceeds 0.1: {ema_kl}"
    )
    assert avg_kl <= 0.15, (
        f"Average KL divergence to the untruncated MLE exceeds 0.15: {avg_kl}"
    )
    assert best_kl <= 0.2, (
        f"Best KL divergence to the untruncated MLE exceeds 0.2: {best_kl}"
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
    S, alpha, parent = helpers.generate_truncated_distribution_dataset(
        dist, phi, num_samples
    )

    emp_lambda_ = S.pow(k).mean(0).pow(1.0 / k)
    print(f"alpha: {alpha}")
    print(f"num truncated samples: {S.size(0)}")
    print(f"emp_lambda: {emp_lambda_}")

    hyperparameters = {
        "iterations": 2500,
        "trials": 1,
        "batch_size": 10,
        "num_samples": 1000,
        "verbose": True,
        "optimizer": "sgd",
        "lr": 1e-2,
    }

    truncated = truncated_weibull.TruncatedWeibull(hyperparameters, phi, dims, k, alpha)
    truncated.fit(S)

    # The untruncated MLE is what the estimator is asked to recover; comparing
    # against the true lambda instead would charge it for sampling error that
    # no estimator can remove.
    mle = parent.pow(k).mean(0).pow(1.0 / k)
    kl_of = functools.partial(
        helpers.weibull_kl_divergence, other_scale=mle, concentration=k
    )

    best_kl = kl_of(truncated.best_lambda_.flatten()).sum()
    ema_kl = kl_of(truncated.ema_lambda_.flatten()).sum()
    avg_kl = kl_of(truncated.avg_lambda_.flatten()).sum()
    emp_kl = kl_of(emp_lambda_.flatten()).sum()
    print(f"empirical kl divergence: {emp_kl.item():.4f}")
    print(f"best kl divergence: {best_kl.item():.4f}")
    print(f"ema kl divergence: {ema_kl.item():.4f}")
    print(f"avg kl divergence: {avg_kl.item():.4f}")
    assert ema_kl <= 1e-1, (
        f"EMA KL divergence to the untruncated MLE exceeds 0.1: {ema_kl}"
    )
    assert avg_kl <= 0.15, (
        f"Average KL divergence to the untruncated MLE exceeds 0.15: {avg_kl}"
    )
    assert best_kl <= 0.2, (
        f"Best KL divergence to the untruncated MLE exceeds 0.2: {best_kl}"
    )
