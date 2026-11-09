# Author: pstefanou12@
"""Tests for detecting whether a sample was truncated."""

from collections.abc import Callable

import pytest
import torch as ch
import torch.distributions

from delphi.torchnomials import empirical, hermite, laguerre, polynomial
import delphi.truncation_detection

BETA = 0.2


def _normal() -> torch.distributions.Distribution:
    return torch.distributions.Independent(
        torch.distributions.Normal(
            ch.zeros(2, dtype=ch.float64), ch.ones(2, dtype=ch.float64)
        ),
        1,
    )


def _poisson() -> torch.distributions.Distribution:
    return torch.distributions.Independent(
        torch.distributions.Poisson(ch.full((2,), 3.0, dtype=ch.float64)), 1
    )


def _gaussian_basis() -> polynomial.Polynomial:
    return hermite.Hermite(4, 2, prob=True, norm=True)


def _poisson_basis() -> polynomial.Polynomial:
    return empirical.EmpiricalOrthonormal(
        laguerre.Laguerre(4, 2, norm=True), _poisson()
    )


def _everything(x: ch.Tensor) -> ch.Tensor:
    return ch.ones(x.size(0), dtype=ch.bool)


def _detect(
    distribution: torch.distributions.Distribution,
    phi: Callable[[ch.Tensor], ch.Tensor],
    basis: polynomial.Polynomial,
    seed: int = 0,
) -> delphi.truncation_detection.TruncationTest:
    """Test 5000 samples from the distribution truncated to phi."""
    ch.manual_seed(seed)
    population = distribution.sample((50000,))
    sample = population[phi(population)][:5000]
    return delphi.truncation_detection.detect_truncation(
        sample, lambda count: distribution.sample((count,)), basis, beta=BETA
    )


# Over seeds 0 to 2 the learned set's mass was 1.0 untruncated, and 0.673 to
# 0.679 and 0.742 truncated, against the threshold 1 - BETA / 2 = 0.9.
@pytest.mark.parametrize(
    "distribution,phi,basis,truncated",
    [
        (_normal(), _everything, _gaussian_basis(), False),
        (_normal(), lambda x: x[:, 0] + 0.5 * x[:, 1] >= -0.5, _gaussian_basis(), True),
        (_poisson(), _everything, _poisson_basis(), False),
        (_poisson(), lambda x: x[:, 0] + x[:, 1] <= 7, _poisson_basis(), True),
    ],
    ids=[
        "gaussian_untruncated",
        "gaussian_half_plane",
        "poisson_untruncated",
        "poisson_half_plane",
    ],
)
def test_detects_truncation(distribution, phi, basis, truncated):
    result = _detect(distribution, phi, basis)
    assert result.truncated == truncated
    assert result.truncation_set is not None


def test_beta_outside_the_unit_interval_raises():
    with pytest.raises(ValueError, match="beta"):
        delphi.truncation_detection.detect_truncation(
            ch.zeros(10, 2),
            lambda count: ch.zeros(count, 2),
            _gaussian_basis(),
            beta=1.0,
        )
