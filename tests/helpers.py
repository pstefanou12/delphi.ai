# Author: pstefanou12@
"""Shared helper utilities for delphi test modules."""

from collections.abc import Callable
from typing import NamedTuple

import torch as ch
from torch.distributions import Distribution


class TruncatedSample(NamedTuple):
    """A truncated sample together with the draw it came from.

    Attributes:
        S: The accepted samples, the only data an estimator sees.
        alpha: Empirical acceptance rate.
        parent: Every sample drawn, accepted or not. Tests compare a fit
            against the estimate this would have given, which isolates the
            estimator's error from the sampling error both share.
    """

    S: ch.Tensor  # pylint: disable=invalid-name
    alpha: float
    parent: ch.Tensor


def generate_truncated_distribution_dataset(
    dist: Distribution, phi: Callable, num_samples: int
) -> TruncatedSample:
    """Draw samples from dist that satisfy phi.

    Repeatedly draws batches from dist, keeps samples where phi returns
    True, and stops once num_samples accepted samples have been collected.

    Args:
        dist: Source distribution with a .sample() method.
        phi: Truncation oracle; returns a boolean mask over a batch.
        num_samples: Number of accepted samples to return.

    Returns:
        The accepted samples, the acceptance rate, and the full draw.
    """
    num_accepted, num_sampled = 0, 0

    accepted: list[ch.Tensor] = []
    drawn: list[ch.Tensor] = []
    while num_accepted < num_samples:
        samples = dist.sample([num_samples])
        drawn.append(samples)
        indices = phi(samples).nonzero()[:, 0]
        accepted.append(samples[indices])
        num_accepted += indices.size(0)
        num_sampled += num_samples
    S = ch.cat(accepted)[:num_samples]  # pylint: disable=invalid-name
    alpha = num_accepted / num_sampled

    return TruncatedSample(S, alpha, ch.cat(drawn))


def weibull_kl_divergence(
    scale: ch.Tensor, other_scale: ch.Tensor, concentration: ch.Tensor
) -> ch.Tensor:
    """Return KL(Weibull(scale, k) || Weibull(other_scale, k)) per coordinate.

    torch registers no KL for Weibull, and the shared-shape case has a closed
    form: ``k log(l2/l1) + (l1/l2)^k - 1``. Checked against 4e6 Monte Carlo
    draws to 3e-4.

    Args:
        scale: Scale of the first distribution.
        other_scale: Scale of the second distribution.
        concentration: Shape parameter, shared by both.

    Returns:
        Per-coordinate KL divergence.
    """
    ratio = scale / other_scale
    return concentration * ch.log(1.0 / ratio) + ratio.pow(concentration) - 1.0
