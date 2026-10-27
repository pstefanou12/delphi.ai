# Author: pstefanou12@
"""Drawing samples that fall inside a truncation set."""

import abc
from collections.abc import Callable
import dataclasses
from typing import Optional

import torch as ch
from torch.distributions import Distribution

# Floors the block estimate so a run of rejections cannot ask for an
# unbounded draw, and caps any single block.
_MIN_ACCEPTANCE_RATE = 1e-3
_MAX_BLOCK = 100

# Below this acceptance rate the sampler gives up: a distribution putting so
# little mass on the set would otherwise run for hours instead of failing.
MIN_ACCEPTANCE = 1e-4

# Draws per sample a GaussianRejectionSampler may take to collect
# min_accepted; a sample whose survival mass is below
# min_accepted / MAX_DRAWS_PER_SAMPLE keeps what it has.
MAX_DRAWS_PER_SAMPLE = 1_000_000
# Caps one round's draws across the batch, bounding its memory.
MAX_DRAWS_PER_ROUND = 10_000_000


class SamplingBudgetExceeded(ValueError):
    """Raised when rejection sampling accepts too rarely to finish."""


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

    Raises:
        SamplingBudgetExceeded: If ``num_samples / MIN_ACCEPTANCE`` draws
            pass without collecting ``num_samples``.
    """
    budget = num_samples / MIN_ACCEPTANCE
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
        if num_accepted < num_samples and num_sampled >= budget:
            raise SamplingBudgetExceeded(
                f"accepted {num_accepted} of {num_sampled} draws, a rate of "
                f"{num_accepted / num_sampled:.2e}, below {MIN_ACCEPTANCE:g}: "
                "the distribution puts almost no mass on the truncation set."
            )
        if num_accepted < num_samples:
            rate = max(num_accepted / num_sampled, _MIN_ACCEPTANCE_RATE)
            block = min(
                int((num_samples - num_accepted) / rate) + 1,
                _MAX_BLOCK * num_samples,
            )
    return ch.cat(accepted_samples), ch.Tensor([num_accepted / num_sampled])


@dataclasses.dataclass(frozen=True)
class ConditionalMoments:
    """Per-sample moments of Gaussian latents given the truncation set.

    Attributes:
        mass: Estimated ``P(z in S)``, shaped (n, 1).
        mean: ``E[z | S]`` in float64, shaped (n, k); zero where no draw was
            kept.
        second_moment: ``E[z^2 | S]`` per coordinate in float64, or None if
            not requested.
    """

    mass: ch.Tensor
    mean: ch.Tensor
    second_moment: Optional[ch.Tensor] = None


class GaussianSampler(abc.ABC):
    """Estimates moments of ``z ~ N(loc, noise_var)`` conditioned on ``z in S``.

    A truncated regression loss reads its gradient and Hessian from these
    moments, so another estimator, such as importance sampling or a closed
    form for intervals, replaces rejection sampling without changing the loss.
    """

    @abc.abstractmethod
    def moments(
        self,
        loc: ch.Tensor,
        noise_var: ch.Tensor,
        phi: Callable,
        second_moment: bool = False,
    ) -> ConditionalMoments:
        """Return the conditional moments at each row of loc.

        Args:
            loc: Means, shaped (n, k).
            noise_var: Noise variance, broadcastable to (k,).
            phi: Truncation oracle over latents shaped (n, m, k), returning
                a mask or a survival probability.
            second_moment: Also estimate ``E[z^2 | S]``.
        """


class GaussianRejectionSampler(GaussianSampler):
    """Rejection-samples each row until phi keeps enough draws.

    A fixed number of draws leaves a sample phi rarely admits with no kept
    draw, whose conditional mean then reads as zero and biases a gradient.
    Rows short of ``min_accepted`` draw again, each round sized from the
    slowest row's acceptance rate. A phi returning a survival probability
    keeps each draw with that probability.

    Attributes:
        num_samples: Draws per row in the first round.
        min_accepted: Kept draws each row collects, up to
            ``MAX_DRAWS_PER_SAMPLE`` draws.
    """

    def __init__(self, num_samples: int = 1000, min_accepted: int = 10):
        """Initialize the sampler.

        Args:
            num_samples: Draws per row in the first round.
            min_accepted: Kept draws each row collects.
        """
        self.num_samples = num_samples
        self.min_accepted = min_accepted

    def moments(
        self,
        loc: ch.Tensor,
        noise_var: ch.Tensor,
        phi: Callable,
        second_moment: bool = False,
    ) -> ConditionalMoments:
        n = loc.size(0)
        accepted, drawn = ch.zeros(n, 1), ch.zeros(n, 1)
        total = ch.zeros_like(loc, dtype=ch.float64)
        total_sq = ch.zeros_like(total) if second_moment else None
        rows, block = ch.arange(n), self.num_samples
        while rows.numel() > 0:
            z = loc[rows, None, :] + noise_var.sqrt() * ch.randn(
                rows.numel(), block, loc.size(1)
            )
            kept = phi(z).reshape(rows.numel(), block, 1)
            kept = ch.bernoulli(kept) if kept.is_floating_point() else kept.float()
            accepted[rows] += kept.sum(1)
            drawn[rows] += block
            total[rows] += (kept * z).sum(1).double()
            if total_sq is not None:
                total_sq[rows] += (kept * z.pow(2)).sum(1).double()
            short = (accepted[rows, 0] < self.min_accepted) & (
                drawn[rows, 0] < MAX_DRAWS_PER_SAMPLE
            )
            rows = rows[short]
            if rows.numel() > 0:
                rate = float((accepted[rows] / drawn[rows]).min().clamp_min(1e-4))
                needed = float((self.min_accepted - accepted[rows]).max()) / rate
                block = int(
                    min(
                        needed + 1,
                        max(MAX_DRAWS_PER_ROUND // rows.numel(), 1),
                        MAX_DRAWS_PER_SAMPLE - float(drawn[rows].max()),
                    )
                )
        count = accepted.double().clamp_min(1.0)
        return ConditionalMoments(
            mass=accepted / drawn,
            mean=total / count,
            second_moment=None if total_sq is None else total_sq / count,
        )
