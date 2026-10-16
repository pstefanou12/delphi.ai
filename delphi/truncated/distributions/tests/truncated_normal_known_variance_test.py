# Author: pstefanou12@
"""Tests for the truncated normal with known variance."""

import torch as ch
from torch import distributions
from torch.distributions import kl

from delphi.truncated.distributions import truncated_normal_known_variance
from delphi.truncated.distributions.tests import helpers

ch.set_printoptions(precision=4, sci_mode=False)


def test_truncated_normal_known_variance():
    """Right-truncated 1D normal with known variance converges to true mean."""
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
    print(f"emp loc:\n {emp_loc}")
    print(f"known variance:\n {ch.eye(1)}")
    S_std_norm = S - emp_loc
    left_std_norm = (left - emp_loc).flatten()

    def phi_std_norm(x):
        return (x > left_std_norm).prod(dim=-1, keepdim=True)

    hyperparameters = {
        "batch_size": 10,
        "trials": 1,
        "verbose": True,
        "lr": 1e-1,
        "num_samples": 1000,
        "early_stopping": True,
        "tol": 5e-2,
        "iterations": 1500,
    }
    truncated = truncated_normal_known_variance.TruncatedNormalKnownVariance(
        hyperparameters, phi_std_norm, ch.eye(1), alpha
    )
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
