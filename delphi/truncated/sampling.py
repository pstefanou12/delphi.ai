# Author: pstefanou12@
"""Drawing samples that fall inside a truncation set."""

from collections.abc import Callable

import torch as ch
from torch.distributions import Distribution

# Floors the block estimate so a run of rejections cannot ask for an
# unbounded draw, and caps any single block.
_MIN_ACCEPTANCE_RATE = 1e-3
_MAX_BLOCK = 100


def rejection_sample(
    dist: Distribution, phi: Callable, num_samples: int
) -> tuple[ch.Tensor, ch.Tensor]:
    """Draw from dist and keep whatever falls inside the truncation set.

    The expected cost is one draw per ``p(S)`` accepted samples, so this
    degrades exactly where truncation is severe. Each block is sized from the
    acceptance rate seen so far: a fixed multiple wastes most of a draw when
    the survival mass is large, since at ``alpha = 0.36`` collecting 1000
    accepted samples needs roughly 2700 draws rather than 10000.

    Args:
        dist: Distribution to draw from.
        phi: Truncation set oracle, returning a boolean mask per sample.
        num_samples: Minimum number of samples to return.

    Returns:
        Tuple of (samples, survival mass), where the mass estimates
        ``P(x in S)`` under dist.
    """
    accepted_samples = []
    num_sampled = 0
    num_accepted = 0
    block = num_samples
    while num_accepted < num_samples:
        drawn = dist.sample([block])
        accepted = drawn[phi(drawn).nonzero()[:, 0]]
        accepted_samples.append(accepted)
        num_accepted += accepted.size(0)
        num_sampled += block
        if num_accepted < num_samples:
            rate = max(num_accepted / num_sampled, _MIN_ACCEPTANCE_RATE)
            block = min(
                int((num_samples - num_accepted) / rate) + 1,
                _MAX_BLOCK * num_samples,
            )
    return ch.cat(accepted_samples), ch.Tensor([num_accepted / num_sampled])
