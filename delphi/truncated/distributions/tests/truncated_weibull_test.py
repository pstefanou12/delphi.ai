# Author: pstefanou12@
"""Tests for the truncated Weibull distribution."""

import functools
from typing import Optional

import pytest
import torch as ch
import torch.distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_weibull
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


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


def _recovers(
    dims: int,
    concentration: ch.Tensor,
    scale: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: bool = True,
    unknown_tolerances: Optional[dict[str, float]] = None,
    set_tolerance: float = helpers.SET_TOLERANCE,
) -> None:
    """Fit a truncated Weibull and assert it recovers the untruncated MLE.

    The comparison is against the MLE of the draw before truncation, not the
    true scale: the latter would charge the estimator for sampling error that
    no estimator can remove.

    Args:
        dims: Number of dimensions.
        concentration: Known shape per dimension.
        scale: True scale per dimension.
        phi: Truncation set oracle.
        pass_alpha: Supply the measured survival mass to the model.
        baseline_must_fail: Forwarded to ``assert_recovers``.
        hyperparameters: Hyperparameter dict for the procedure.
        known: Give the model phi; else it learns the set, scored against phi.
        unknown_tolerances: Bounds for the unknown variant, where it misses
            the shared ones.
        set_tolerance: Bound on the learned set's error.
    """
    ch.manual_seed(0)
    target = torch.distributions.Weibull(scale, concentration)
    S, alpha, parent = helpers.generate_truncated_distribution_dataset(  # pylint: disable=invalid-name
        target, phi, 10000
    )
    args = (alpha,) if pass_alpha else ()
    truncated = helpers.build(
        truncated_weibull.TruncatedWeibull,
        phi,
        known,
        dims,
        concentration,
        *args,
        hyperparameters=hyperparameters,
    )
    truncated.fit(S)
    if not known:
        helpers.assert_learned_set(truncated, phi, target, set_tolerance)

    mle = parent.pow(concentration).mean(0).pow(1.0 / concentration)
    helpers.assert_recovers(
        estimate=lambda name: getattr(truncated, f"{name}_scale_").flatten(),
        divergence=lambda fitted: weibull_kl_divergence(fitted, mle, concentration),
        baseline=S.pow(concentration).mean(0).pow(1.0 / concentration).flatten(),
        baseline_must_fail=baseline_must_fail,
        **helpers.unknown_bounds(known, unknown_tolerances),
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension(known):
    _recovers(
        dims=1,
        concentration=ch.tensor([2.0]),
        scale=ch.tensor([1.0]),
        phi=lambda z: z > 1.5,
        hyperparameters={"iterations": 1500, "batch_size": 1},
        known=known,
    )


@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN_CI
def test_two_dimensions(known):
    _recovers(
        dims=2,
        concentration=ch.tensor([1.0, 1.0]),
        scale=ch.tensor([1.0, 2.0]),
        phi=lambda z: z[:, 0] > 1.0,
        known=known,
    )


@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_differing_shapes(known):
    _recovers(
        dims=2,
        concentration=ch.tensor([2.0, 3.0]),
        scale=ch.tensor([1.0, 2.0]),
        phi=lambda z: z[:, 0] > 1.0,
        hyperparameters={"batch_size": 100},
        known=known,
    )


@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN
def test_twenty_dimensions_differing_shapes(known):
    _recovers(
        dims=20,
        concentration=ch.tensor([2.0, 3.0, 1.0, 2.0, 1.0]).repeat(4),
        scale=ch.tensor([1.0, 2.0, 3.0, 1.0, 5.0]).repeat(4),
        phi=lambda z: z[:, 0] > 1.0,
        known=known,
    )


# Unknown truncation, rho 0.3: over seeds 0,1,2 the estimate's KL measured
# 0.264 to 0.289 (ema), 0.265 to 0.290 (avg).
@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_half_space(known):
    concentration, scale = ch.tensor([2.0, 2.0]), ch.tensor([1.0, 2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 2, 2),
        torch.distributions.Weibull(scale, concentration),
        bounds=(-25.0, 25.0),
    )
    _recovers(
        dims=2,
        concentration=concentration,
        scale=scale,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.58, "avg": 0.58},
        hyperparameters={"truncation_set_rho": 0.3},
    )


# Unknown truncation, rho 0.3: over seeds 0,1,2 the estimate's KL measured
# 0.097 to 0.157 (ema), 0.100 to 0.156 (avg).
@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_disc_union(known):
    concentration, scale = ch.tensor([2.0, 2.0]), ch.tensor([1.0, 2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_disc_union, 1, 2, 6),
        torch.distributions.Weibull(scale, concentration),
        bounds=(1e-4, 1e3),
    )
    _recovers(
        dims=2,
        concentration=concentration,
        scale=scale,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.32, "avg": 0.32},
        hyperparameters={"truncation_set_rho": 0.3},
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_identity(known):
    _recovers(
        dims=1,
        concentration=ch.tensor([2.0]),
        scale=ch.tensor([1.0]),
        phi=delphi.oracle.Identity(),
        hyperparameters={"iterations": 1500},
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    _recovers(
        dims=10,
        concentration=ch.tensor([2.0, 3.0, 1.0, 2.0, 1.0]).repeat(2),
        scale=ch.tensor([1.0, 2.0, 3.0, 1.0, 5.0]).repeat(2),
        phi=delphi.oracle.Identity(),
        hyperparameters={"iterations": 1500},
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_small_mass(known):
    scale, concentration = ch.tensor([1.0]), ch.tensor([2.0])
    # The regime both papers state their guarantees in; nothing else in the
    # suite goes near it.
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.Weibull(scale, concentration),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 50.0),
    )
    _recovers(
        dims=1,
        concentration=concentration,
        scale=scale,
        phi=calibrated.phi,
        known=known,
    )


@pytest.mark.long
@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the learned window is too coarse to separate the pair",
    )
)
def test_indistinguishable_pair(known):
    # Truncated to a narrow window the two samples agree to within sampling
    # error, so only an estimator that knows the window can tell them apart.
    concentration = ch.tensor([2.0])
    scales = (ch.tensor([1.0]), ch.tensor([2.0]))

    def phi(z):
        return (z[:, 0] > 1.0) & (z[:, 0] < 1.2)

    fits, naive = [], []
    for scale in scales:
        ch.manual_seed(0)
        sample = helpers.generate_truncated_distribution_dataset(
            torch.distributions.Weibull(scale, concentration), phi, 10000
        )
        target = torch.distributions.Weibull(scale, concentration)
        truncated = helpers.build(
            truncated_weibull.TruncatedWeibull,
            phi,
            known,
            1,
            concentration,
            sample.alpha,
        )
        truncated.fit(sample.S)
        if not known:
            helpers.assert_learned_set(truncated, phi, target)
        fits.append(truncated)
        naive.append(sample.S.pow(concentration).mean(0).pow(1.0 / concentration))

    for name in helpers.TOLERANCES:
        helpers.assert_distinguishes(
            targets=scales,
            recovered=(
                getattr(fits[0], f"{name}_scale_").flatten(),
                getattr(fits[1], f"{name}_scale_").flatten(),
            ),
            naive=(naive[0], naive[1]),
            divergence=lambda a, b: weibull_kl_divergence(a, b, concentration),
        )
