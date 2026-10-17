# Author: pstefanou12@
"""Tests for the multivariate normal with unknown truncation."""

import pytest
import torch as ch
from torch import distributions
from torch.distributions import kl
from torch.distributions.multivariate_normal import _batch_mahalanobis

from delphi.truncated.distributions import unknown_truncated_multivariate_normal
from delphi.truncated.distributions.tests import helpers as test_helpers
from delphi.utils import helpers

ch.set_printoptions(precision=4, sci_mode=False)


# TODO: pstefanou12@ - re-enable with the unknown-truncation rework. The
# Hermite oracle gets k=12 in 10 dims, i.e. 646,646 multi-indices, so h_v
# materialises a ~13GB (5000, 646646) tensor on every call. k also changed
# meaning from a neighbor count to a polynomial degree without its callers
# being updated.
@pytest.mark.skip(
    reason="unknown-truncation rework: hangs building a ~13GB Hermite tensor"
)
def test_unknown_truncation_multivariate_normal():
    """10D sphere-truncated MVN with unknown truncation recovers true params."""
    ch.manual_seed(0)
    dims = 10
    M = distributions.MultivariateNormal(ch.zeros(dims), ch.eye(dims))

    W = distributions.Uniform(-0.5, 0.5)
    centroid = W.sample([dims])
    scale_tril = ch.linalg.cholesky(M.covariance_matrix)  # pylint: disable=not-callable

    def phi(x):
        diff = x - centroid
        dist = ch.sqrt(_batch_mahalanobis(scale_tril, diff))
        return (dist < 3.5).float().flatten()

    num_samples = 5000
    S, alpha, parent = test_helpers.generate_truncated_distribution_dataset(
        M, phi, num_samples
    )
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
    }
    truncated = (
        unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormal(
            args, k, alpha, dims
        )
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
    kl_emp = kl.kl_divergence(distributions.MultivariateNormal(emp_loc, emp_cov), M)
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
