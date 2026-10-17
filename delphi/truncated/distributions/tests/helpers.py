# Author: pstefanou12@
"""Shared helper utilities for delphi test modules."""

from collections.abc import Callable
from typing import NamedTuple, Optional, Protocol, TypeVar, cast

import torch as ch
from torch.distributions import Distribution, kl

# Read through .nonzero(), so a bool, int or float return all work.
Oracle = Callable[[ch.Tensor], ch.Tensor]


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
    dist: Distribution, phi: Oracle, dataset_size: int
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


# Generic because Weibull has no registered KL, so it scores bare tensors.
_FittedT = TypeVar("_FittedT")


class TruncatedModel(Protocol):
    """The surface assert_family_recovers needs from a truncated model."""

    def fit(self, S: ch.Tensor) -> object:  # pylint: disable=invalid-name
        """Fit the model to truncated data."""


# The averaged iterate is the least noisy estimate and the one selected by a
# noisy validation loss is the most, so each carries its own bound.
TOLERANCES = {"ema": 1e-1, "avg": 0.15, "best": 0.2}

# A summed bound lets one coordinate absorb the whole budget while the rest
# are perfect. Measured worst coordinate across the suite is 0.0809.
MAX_COORDINATE = 0.1


def assert_recovers(  # pylint: disable=too-many-arguments
    estimate: Callable[[str], _FittedT],
    divergence: Callable[[_FittedT], ch.Tensor],
    baseline: Optional[_FittedT] = None,
    tolerances: Optional[dict[str, float]] = None,
    baseline_must_fail: bool = True,
    max_coordinate: Optional[float] = None,
) -> None:
    """Assert every estimator lands within its divergence tolerance.

    Args:
        estimate: Maps an estimator name -- ``ema``, ``avg`` or ``best`` -- to
            the fitted value, undoing any standardization the caller applied.
        divergence: Maps a fitted value to its divergence from the target,
            unreduced, so a product family returns one entry per coordinate.
        baseline: The naive estimate, scored for comparison.
        tolerances: Per-estimator bounds on the summed divergence, defaulting
            to ``TOLERANCES``.
        baseline_must_fail: Require the naive estimate to breach every bound.
            False only where it is the right answer, as under no truncation.
        max_coordinate: Bound on the worst single coordinate, defaulting to
            ``MAX_COORDINATE``. Applied only where the divergence decomposes,
            so a one-dimensional or joint score is scored by the sum alone.

    Raises:
        AssertionError: If an estimator exceeds a bound, or if the naive
            estimate passes while ``baseline_must_fail`` is set.
    """
    bounds = TOLERANCES if tolerances is None else tolerances
    if baseline is not None:
        naive = float(divergence(baseline).sum())
        print(f"empirical divergence: {naive:.4f}")
        if baseline_must_fail:
            assert naive > max(bounds.values()), (
                f"the naive estimate scores {naive:.4f}, inside the loosest "
                f"bound {max(bounds.values())}: this test would pass without "
                "an estimator, so it measures nothing"
            )

    for name, bound in bounds.items():
        per_coordinate = divergence(estimate(name))
        value = float(per_coordinate.sum())
        worst = float(per_coordinate.max())
        print(f"{name} divergence: {value:.4f} (worst coordinate {worst:.4f})")
        assert value <= bound, f"{name} divergence {value:.4f} exceeds {bound}"
        if per_coordinate.numel() > 1:
            ceiling = MAX_COORDINATE if max_coordinate is None else max_coordinate
            assert worst <= ceiling, (
                f"{name} worst coordinate {worst:.4f} exceeds {ceiling}, so the "
                "error is concentrated rather than spread"
            )


def assert_family_recovers(  # pylint: disable=too-many-arguments
    model_class: Callable[..., TruncatedModel],
    target: Distribution,
    phi: Oracle,
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
        divergence=lambda fitted: kl.kl_divergence(fitted, target),
        baseline=family(empirical(sample.S)),
    )
