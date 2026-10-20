# Author: pstefanou12@
"""Shared helper utilities for delphi test modules."""

from collections.abc import Callable
from typing import Any, NamedTuple, Optional, Protocol, TypeVar, Union, cast

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


# Generic because Weibull has no registered KL, so it scores bare tensors.
_FittedT = TypeVar("_FittedT")


class TruncatedModel(Protocol):
    """The surface assert_family_recovers needs from a truncated model."""

    def fit(self, S: ch.Tensor) -> object:  # pylint: disable=invalid-name
        """Fit the model to truncated data."""


TOLERANCES = {"ema": 1e-1, "avg": 0.15}

# With nothing truncated the procedure starts at the MLE and should stay
# there, so it is held far tighter than a fit that has to travel.
UNTRUNCATED_TOLERANCES = {"ema": 0.02, "avg": 0.02}

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
            to ``TOLERANCES``, or to ``UNTRUNCATED_TOLERANCES`` when
            ``baseline_must_fail`` is off.
        baseline_must_fail: Require the naive estimate to breach every bound.
            False only where it is the right answer, as under no truncation.
        max_coordinate: Bound on the worst single coordinate, defaulting to
            ``MAX_COORDINATE``. Applied only where the divergence decomposes,
            so a one-dimensional or joint score is scored by the sum alone.

    Raises:
        AssertionError: If an estimator exceeds a bound, or if the naive
            estimate passes while ``baseline_must_fail`` is set.
    """
    if tolerances is not None:
        bounds = tolerances
    elif baseline_must_fail:
        bounds = TOLERANCES
    else:
        bounds = UNTRUNCATED_TOLERANCES
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


class _Family(NamedTuple):
    """How a family's estimates and naive estimate become distributions."""

    suffixes: tuple[str, ...]
    rebuild: Callable[..., Distribution]

    def fitted(self, model: object, name: str) -> Distribution:
        """Return the distribution an estimator of a fitted model describes."""
        return self.rebuild(*(getattr(model, f"{name}_{s}") for s in self.suffixes))

    def naive(self, estimate) -> Distribution:
        """Return the distribution the naive estimate describes."""
        return (
            self.rebuild(*estimate)
            if isinstance(estimate, tuple)
            else self.rebuild(estimate)
        )


def _family(
    target: Distribution,
    parameter: Union[str, tuple[str, ...]],
    rebuild: Optional[Callable[..., Distribution]],
) -> _Family:
    # By default the canonical parameter alone rebuilds the distribution,
    # which the Distribution base signature does not express.
    default = cast(Callable[..., Distribution], type(target))
    suffixes = (parameter,) if isinstance(parameter, str) else parameter
    return _Family(suffixes, rebuild or default)


def assert_family_recovers(  # pylint: disable=too-many-arguments
    model_class: Callable[..., TruncatedModel],
    target: Distribution,
    phi: Oracle,
    dims: int,
    parameter: Union[str, tuple[str, ...]],
    empirical: Callable[[ch.Tensor], Any],
    hyperparameters: dict,
    dataset_size: int = 10000,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    model_args: tuple = (),
    rebuild: Optional[Callable[..., Distribution]] = None,
    divergence: Callable[[Distribution, Distribution], ch.Tensor] = kl.kl_divergence,
) -> None:
    """Fit a family on truncated data and assert it recovers the target.

    By default ``type(target)(estimate)`` is the fitted distribution and
    torch's registered KL scores it. ``rebuild`` and ``divergence`` cover a
    family with known parameters, several estimates, or no registered KL. A
    family needing a transform on the way in keeps its own fit.

    Args:
        model_class: Truncated model, called as
            ``model_class(hyperparameters, phi, dims, alpha)``.
        target: Untruncated distribution the data is drawn from.
        phi: Truncation set oracle.
        dims: Number of dimensions.
        parameter: Attribute suffix holding the canonical estimate, so that
            ``f"avg_{parameter}"`` names the averaged iterate, or several.
        empirical: Maps the truncated data to the naive estimate, a tuple
            when there are several.
        hyperparameters: Hyperparameter dict for the procedure.
        dataset_size: Samples inside the truncation set to fit on.
        pass_alpha: Supply the measured survival mass to the model. False
            leaves the model to fall back on ``min_radius``, which is what a
            caller who cannot measure it gets.
        baseline_must_fail: Forwarded to ``assert_recovers``.
        model_args: Known parameters, passed to the model before alpha.
        rebuild: Maps the estimates to a distribution; ``type(target)`` by
            default.
        divergence: Scores a fitted distribution against the target.

    Raises:
        AssertionError: If any estimator exceeds its bound.
    """
    ch.manual_seed(0)
    sample = generate_truncated_distribution_dataset(target, phi, dataset_size)
    args = (sample.alpha,) if pass_alpha else ()
    truncated = model_class(hyperparameters, phi, dims, *model_args, *args)
    truncated.fit(sample.S)

    family = _family(target, parameter, rebuild)
    assert_recovers(
        estimate=lambda name: family.fitted(truncated, name),
        divergence=lambda fitted: divergence(fitted, target),
        baseline=family.naive(empirical(sample.S)),
        baseline_must_fail=baseline_must_fail,
    )


def assert_distinguishes(  # pylint: disable=too-many-arguments
    targets: tuple[_FittedT, _FittedT],
    recovered: tuple[_FittedT, _FittedT],
    naive: tuple[_FittedT, _FittedT],
    divergence: Callable[[_FittedT, _FittedT], ch.Tensor],
    recovery_bound: float = 0.2,
    confusion_bound: float = 0.05,
    separation_floor: float = 0.5,
) -> None:
    """Assert a fit separates two populations a truncated sample cannot.

    Two distributions far apart can produce near-identical samples once
    truncated to the same set, which is the illustration Daskalakis et al.
    open with: N((0,1), I) and N((0,0), 4I) cut to a box. The observed moments
    agree to three decimals, so nothing computed from the sample alone can
    tell them apart. Only an estimator that knows the set can.

    Args:
        targets: The two untruncated distributions the samples came from.
        recovered: What the procedure fitted for each.
        naive: The estimate each truncated sample gives on its own.
        divergence: Maps a pair to a divergence, unreduced.
        recovery_bound: Bound on each fit against its own target.
        confusion_bound: How close the two naive estimates must be. A test
            where they already differ proves nothing.
        separation_floor: Fraction of the true separation the fits must
            reproduce.

    Raises:
        AssertionError: If a fit misses its target, if the naive estimates
            are already separated, or if the fits are not separated.
    """
    separation = float(divergence(*targets).sum())
    confusion = float(divergence(*naive).sum())
    print(f"true separation: {separation:.4f}")
    print(f"naive separation: {confusion:.4f} (the samples look alike)")

    assert confusion <= confusion_bound, (
        f"the naive estimates are already {confusion:.4f} apart, inside "
        f"{confusion_bound}: the two truncated samples are distinguishable "
        "without an estimator, so this pair tests nothing"
    )

    for index, name in enumerate(("first", "second")):
        value = float(divergence(recovered[index], targets[index]).sum())
        print(f"{name} recovered: {value:.4f}")
        assert value <= recovery_bound, (
            f"{name} fit is {value:.4f} from its target, over {recovery_bound}"
        )

    recovered_separation = float(divergence(*recovered).sum())
    floor = separation_floor * separation
    print(f"recovered separation: {recovered_separation:.4f} (floor {floor:.4f})")
    assert recovered_separation >= floor, (
        f"the fits are only {recovered_separation:.4f} apart against a true "
        f"{separation:.4f}: the procedure did not separate them"
    )


def assert_family_distinguishes(  # pylint: disable=too-many-arguments
    model_class: Callable[..., TruncatedModel],
    targets: tuple[Distribution, Distribution],
    phi: Oracle,
    dims: int,
    parameter: Union[str, tuple[str, ...]],
    empirical: Callable[[ch.Tensor], Any],
    hyperparameters: dict,
    dataset_size: int = 10000,
    model_args: tuple = (),
    rebuild: Optional[Callable[..., Distribution]] = None,
    divergence: Callable[[Distribution, Distribution], ch.Tensor] = kl.kl_divergence,
) -> None:
    """Fit two families truncated to one set and assert they come apart.

    The families covered are those ``assert_family_recovers`` covers: the
    canonical parameter alone rebuilds the distribution, and torch registers
    its KL. Every estimator is checked.

    Args:
        model_class: Truncated model, called as
            ``model_class(hyperparameters, phi, dims, alpha)``.
        targets: The two untruncated distributions.
        phi: The one truncation set both are cut to.
        dims: Number of dimensions.
        parameter: Attribute suffix holding the canonical estimate.
        empirical: Maps a truncated sample to the naive estimate.
        hyperparameters: Hyperparameter dict for the procedure.
        dataset_size: Samples inside the truncation set to fit on.
        model_args: As ``assert_family_recovers``.
        rebuild: As ``assert_family_recovers``.
        divergence: As ``assert_family_recovers``.

    Raises:
        AssertionError: As ``assert_distinguishes``.
    """
    family = _family(targets[0], parameter, rebuild)
    fits, naive = [], []
    for target in targets:
        ch.manual_seed(0)
        sample = generate_truncated_distribution_dataset(target, phi, dataset_size)
        truncated = model_class(hyperparameters, phi, dims, *model_args, sample.alpha)
        truncated.fit(sample.S)
        fits.append(truncated)
        naive.append(family.naive(empirical(sample.S)))

    for name in TOLERANCES:
        print(f"-- {name}")
        assert_distinguishes(
            targets=targets,
            recovered=(family.fitted(fits[0], name), family.fitted(fits[1], name)),
            naive=(naive[0], naive[1]),
            divergence=divergence,
        )
