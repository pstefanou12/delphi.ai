# Author: pstefanou12@
"""Shared helper utilities for delphi test modules."""

from collections.abc import Callable
from typing import NamedTuple, Optional, Protocol, TypeVar, cast

import torch as ch
from torch.distributions import Distribution, kl


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
    dist: Distribution, phi: Callable, dataset_size: int
) -> TruncatedSample:
    """Draw samples from dist that satisfy phi.

    Repeatedly draws batches from dist, keeps samples where phi returns
    True, and stops once dataset_size accepted samples have been collected.

    Args:
        dist: Source distribution with a .sample() method.
        phi: Truncation oracle; returns a boolean mask over a batch.
        dataset_size: Number of samples inside the truncation set to return.
            Unrelated to the ``num_samples`` hyperparameter, which sets the
            rejection draws backing a single gradient step.

    Returns:
        The accepted samples, the acceptance rate, and the full draw.
    """
    num_accepted, num_sampled = 0, 0

    accepted: list[ch.Tensor] = []
    drawn: list[ch.Tensor] = []
    while num_accepted < dataset_size:
        samples = dist.sample([dataset_size])
        drawn.append(samples)
        indices = phi(samples).nonzero()[:, 0]
        accepted.append(samples[indices])
        num_accepted += indices.size(0)
        num_sampled += dataset_size
    S = ch.cat(accepted)[:dataset_size]  # pylint: disable=invalid-name
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


# A fitted value, scored by whatever divergence the caller pairs with it: a
# torch distribution under a registered KL, or a bare parameter tensor where
# torch registers none.
_FittedT = TypeVar("_FittedT")


class TruncatedModel(Protocol):
    """The surface assert_family_recovers needs from a truncated model."""

    def fit(self, S: ch.Tensor) -> object:  # pylint: disable=invalid-name
        """Fit the model to truncated data."""


# The averaged iterate is the least noisy estimate and the one selected by a
# noisy validation loss is the most, so each carries its own bound.
TOLERANCES = {"ema": 1e-1, "avg": 0.15, "best": 0.2}


def assert_recovers(
    estimate: Callable[[str], _FittedT],
    divergence: Callable[[_FittedT], ch.Tensor],
    baseline: Optional[_FittedT] = None,
    tolerances: Optional[dict[str, float]] = None,
) -> None:
    """Assert every estimator lands within its divergence tolerance.

    Args:
        estimate: Maps an estimator name -- ``ema``, ``avg`` or ``best`` -- to
            the fitted value, undoing any standardization the caller applied.
        divergence: Maps a fitted value to a scalar divergence from the target.
        baseline: The naive estimate, printed for comparison so a test that
            stops discriminating is visible.
        tolerances: Per-estimator bounds, defaulting to ``TOLERANCES``.

    Raises:
        AssertionError: If any estimator exceeds its bound.
    """
    bounds = TOLERANCES if tolerances is None else tolerances
    if baseline is not None:
        print(f"empirical divergence: {float(divergence(baseline)):.4f}")
    for name, bound in bounds.items():
        value = float(divergence(estimate(name)))
        print(f"{name} divergence: {value:.4f}")
        assert value <= bound, f"{name} divergence {value:.4f} exceeds {bound}"


def assert_family_recovers(  # pylint: disable=too-many-arguments
    model_class: Callable[..., TruncatedModel],
    target: Distribution,
    phi: Callable,
    dims: int,
    parameter: str,
    empirical: Callable[[ch.Tensor], ch.Tensor],
    hyperparameters: dict,
    dataset_size: int = 10000,
) -> None:
    """Fit a family on truncated data and assert it recovers the target.

    Covers families whose canonical parameter reconstructs the distribution on
    its own, so ``type(target)(estimate)`` is the fitted distribution and
    torch's registered KL scores it. A family needing a transform on the way
    in, or carrying a second parameter, keeps its own fit in its test module.

    Args:
        model_class: Truncated model, called as
            ``model_class(hyperparameters, phi, dims, alpha)``.
        target: Untruncated distribution the data is drawn from.
        phi: Truncation set oracle.
        dims: Number of dimensions.
        parameter: Attribute suffix holding the canonical estimate, so that
            ``f"avg_{parameter}"`` names the averaged iterate.
        empirical: Maps the truncated data to the naive estimate.
        hyperparameters: Hyperparameter dict for the procedure.
        dataset_size: Samples inside the truncation set to fit on.

    Raises:
        AssertionError: If any estimator exceeds its bound.
    """
    ch.manual_seed(0)
    sample = generate_truncated_distribution_dataset(target, phi, dataset_size)
    truncated = model_class(hyperparameters, phi, dims, sample.alpha)
    truncated.fit(sample.S)

    # Each family here builds from its canonical parameter alone, which the
    # Distribution base signature does not express.
    family = cast(Callable[[ch.Tensor], Distribution], type(target))
    assert_recovers(
        estimate=lambda name: family(getattr(truncated, f"{name}_{parameter}")),
        divergence=lambda fitted: kl.kl_divergence(fitted, target).sum(),
        baseline=family(empirical(sample.S)),
    )
