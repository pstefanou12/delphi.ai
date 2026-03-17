# Author: pstefanou12@
"""Shared helper utilities for delphi test modules."""

from collections.abc import Callable

import torch as ch
from torch.distributions import Distribution


def generate_truncated_distribution_dataset(
    dist: Distribution, phi: Callable, num_samples: int
) -> tuple[ch.Tensor, float]:
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
