# Author: pstefanou12@
"""Shared fixtures and checks for the estimators under unknown truncation."""

from collections.abc import Callable
from typing import Any, Optional

import torch as ch
from torch import distributions
from torch.distributions import kl

from delphi.truncated.distributions.tests import helpers

CORRELATED = ch.tensor([[2.0, 0.5], [0.5, 1.0]])
CORRELATED_3D = ch.tensor([[1.5, 0.4, -0.3], [0.4, 1.0, 0.2], [-0.3, 0.2, 0.8]])

# Population mass on which the recovered and the true set may disagree. The
# fits measured 0.005 to 0.073, the worst a ten-dimensional ball and a
# Sierpinski triangle at 2% to 6% survival mass.
SET_TOLERANCE = 0.1


def correlated(dims: int, seed: int) -> ch.Tensor:
    """Return a fixed random covariance with correlated coordinates."""
    generator = ch.Generator().manual_seed(seed)
    factor = 0.4 * ch.randn(dims, dims, generator=generator)
    return factor @ factor.T + 0.6 * ch.eye(dims)


def two_half_spaces(z: ch.Tensor) -> ch.Tensor:
    return (z[:, 0] > -0.5) & (z[:, 1] > -0.5)


def assert_recovers(  # pylint: disable=too-many-arguments
    model: Any,
    target: distributions.Distribution,
    phi: helpers.Oracle,
    estimate: Callable[[str], distributions.Distribution],
    baseline: Callable[[ch.Tensor], distributions.Distribution],
    dataset_size: int = 20000,
    tolerances: Optional[dict[str, float]] = None,
    set_tolerance: float = SET_TOLERANCE,
) -> None:
    """Fit on a truncated sample; assert it recovers the truth and the set.

    Args:
        model: The unfit estimator.
        target: The untruncated truth.
        phi: The truncation set; only the sampler and the set check see it.
        estimate: Maps an estimator's name to its fitted distribution.
        baseline: Maps the sample to the naive estimate.
        dataset_size: Truncated samples to draw.
        tolerances: Per-estimator bounds; the suite's usual ones by default.
        set_tolerance: Bound on the population mass where the recovered and
            the true set disagree.
    """
    ch.manual_seed(0)
    sample = helpers.generate_truncated_distribution_dataset(target, phi, dataset_size)
    model.fit(sample.S)
    helpers.assert_recovers(
        estimate=estimate,
        divergence=lambda fitted: kl.kl_divergence(fitted, target),
        baseline=baseline(sample.S),
        tolerances=tolerances,
    )
    ch.manual_seed(1)
    population = target.sample([20000])
    inside = phi(population)
    error = float((model.truncation_set()(population) != inside).float().mean())
    print(f"set error: {error:.4f} (true set mass {float(inside.float().mean()):.4f})")
    assert error <= set_tolerance, f"set error {error:.4f} exceeds {set_tolerance}"
