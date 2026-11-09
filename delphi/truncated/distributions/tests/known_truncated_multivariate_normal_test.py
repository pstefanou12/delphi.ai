# Author: pstefanou12@
"""Tests for the multivariate normal under a known truncation set."""

import functools
import math
from typing import Optional

import pytest
import torch as ch
import torch.distributions
import torch.distributions.kl

import delphi.oracle
from delphi.truncated.distributions import known_truncated_multivariate_normal
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _fit(
    target: torch.distributions.MultivariateNormal,
    phi: helpers.Oracle,
    dataset_size: int,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    known: helpers.Known = True,
    set_tolerance: float = helpers.SET_TOLERANCE,
):
    """Fit on a standardized truncated sample and map the estimates back.

    Returns:
        The estimate of each estimator, by name, and the naive fit: the
        truncated sample's own mean and covariance.
    """
    ch.manual_seed(0)
    S, alpha, _ = helpers.generate_truncated_distribution_dataset(  # pylint: disable=invalid-name
        target, phi, dataset_size
    )
    loc, scale = S.mean(0, keepdim=True), S.std(0)

    def phi_standardised(x):
        # The fit sees standardized points, so phi must undo the transform.
        return phi(x * scale + loc)

    args = (alpha,) if pass_alpha else ()
    model = helpers.build(
        known_truncated_multivariate_normal.KnownTruncationMultivariateNormal,
        phi_standardised,
        known,
        target.loc.numel(),
        *args,
        hyperparameters=hyperparameters,
    )
    model.fit((S - loc) / scale)
    if helpers.learns_set(known):
        helpers.assert_learned_set(
            model, phi, target, set_tolerance, to_model=lambda x: (x - loc) / scale
        )

    def estimate(name: str) -> torch.distributions.MultivariateNormal:
        covariance = getattr(model, f"{name}_covariance_matrix_")
        return torch.distributions.MultivariateNormal(
            getattr(model, f"{name}_loc_") * scale + loc.flatten(),
            ch.diag(scale) @ covariance @ ch.diag(scale),
        )

    naive = torch.distributions.MultivariateNormal(
        loc.flatten(), ch.atleast_2d(ch.cov(S.T))
    )
    return estimate, naive


def _recovers(  # pylint: disable=too-many-arguments
    target: torch.distributions.MultivariateNormal,
    phi: helpers.Oracle,
    dataset_size: int = 5000,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    tolerances: Optional[dict[str, float]] = None,
    known: helpers.Known = True,
    unknown_tolerances: Optional[dict[str, float]] = None,
    set_tolerance: float = helpers.SET_TOLERANCE,
) -> None:
    estimate, naive = _fit(
        target, phi, dataset_size, hyperparameters, pass_alpha, known, set_tolerance
    )
    helpers.assert_recovers(
        estimate=estimate,
        divergence=lambda fitted: torch.distributions.kl.kl_divergence(fitted, target),
        baseline=naive,
        baseline_must_fail=baseline_must_fail,
        tolerances=(
            unknown_tolerances or tolerances
            if helpers.learns_set(known)
            else tolerances
        ),
    )


def _calibrated(
    factory, target: torch.distributions.MultivariateNormal, **kwargs
) -> helpers.Oracle:
    return truncation_sets.calibrate(factory, target, **kwargs).phi


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension(known):
    target = helpers.correlated_normal(1)
    phi = _calibrated(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        target,
        target_alpha=(0.15, 0.25),
        bounds=(-10.0, 10.0),
    )
    _recovers(target, phi, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions(known):
    target = helpers.correlated_normal(2)
    phi = _calibrated(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        target,
        target_alpha=(0.15, 0.25),
        bounds=(-10.0, 10.0),
    )
    _recovers(target, phi, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_half_space(known):
    target = helpers.correlated_normal(2)
    phi = _calibrated(
        functools.partial(truncation_sets.random_half_space, 1, 2),
        target,
        bounds=(-25.0, 25.0),
    )
    _recovers(target, phi, known=known)


# Unknown truncation needs degree 6 here; over seeds 0,1,2 it meets the shared
# bounds.
@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_annulus(known):
    _recovers(
        helpers.correlated_normal(2),
        truncation_sets.annulus(2.5, 5.0),
        known=known,
        hyperparameters={"truncation_set_degree": 6},
    )


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the estimate's KL 0.95 is over half the naive fit's 1.66",
    )
)
def test_two_dimensions_crescent(known):
    _recovers(
        helpers.correlated_normal(2),
        truncation_sets.crescent(2.0, 6.0, 0.0, math.pi),
        known=known,
    )


# Unknown truncation: over seeds 0,1,2 the estimate's KL measured 0.136 to
# 0.200 (ema), 0.135 to 0.199 (avg); the set error 0.117 to 0.147.
@helpers.KNOWN_AND_UNKNOWN_CI
def test_two_dimensions_random_disc_union(known):
    target = helpers.correlated_normal(2)
    phi = _calibrated(
        functools.partial(truncation_sets.random_disc_union, 0, 2, 6),
        target,
        bounds=(1e-4, 1e3),
    )
    _recovers(
        target,
        phi,
        known=known,
        unknown_tolerances={"ema": 0.4, "avg": 0.4},
        set_tolerance=0.3,
    )


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the estimate's KL 0.60 is over half the naive fit's 1.19",
    )
)
def test_ten_dimensions(known):
    target = helpers.correlated_normal(10)
    phi = _calibrated(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        target,
        target_alpha=(0.15, 0.25),
        bounds=(-10.0, 10.0),
    )
    _recovers(target, phi, known=known)


# Unknown truncation, degree 3: over seeds 0,1,2 the estimate's KL measured
# 0.284 to 0.451 (ema), 0.256 to 0.453 (avg).
@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_sum_threshold(known):
    target = helpers.correlated_normal(10)
    # One scalar condition that biases every coordinate, not just one.
    phi = _calibrated(
        truncation_sets.sum_threshold,
        target,
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    # The cut leaves the variance along the sum poorly identified: at 5000
    # samples the fit plateaus near 0.1 however long it runs.
    _recovers(
        target,
        phi,
        dataset_size=20000,
        known=known,
        unknown_tolerances={"ema": 0.91, "avg": 0.91},
        hyperparameters={"truncation_set_degree": 3},
    )


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the estimate's KL reaches 0.37 against the naive fit's 0.83",
    )
)
def test_ten_dimensions_norm_threshold(known):
    target = helpers.correlated_normal(10)
    # Couples every coordinate like the sum does, and also excludes the mode.
    phi = _calibrated(
        truncation_sets.norm_threshold,
        target,
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 100.0),
    )
    _recovers(target, phi, known=known)


@pytest.mark.slow
@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: at degree 2, the default in 20 dims, the set comes back as all of space (error 0.81 on seeds 0-1)",
    )
)
def test_twenty_dimensions(known):
    target = helpers.correlated_normal(20)
    phi = _calibrated(
        truncation_sets.sum_threshold,
        target,
        target_alpha=(0.15, 0.25),
        bounds=(-100.0, 100.0),
    )
    _recovers(target, phi, dataset_size=10000, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_identity(known):
    _recovers(
        helpers.correlated_normal(1),
        delphi.oracle.Identity(),
        hyperparameters={"iterations": 1500, "num_samples": 10000},
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    _recovers(
        helpers.correlated_normal(10),
        delphi.oracle.Identity(),
        hyperparameters={"iterations": 1500, "num_samples": 10000},
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_small_mass(known):
    target = helpers.correlated_normal(1)
    # The regime the paper states its guarantee in; nothing else goes near it.
    phi = _calibrated(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        target,
        target_alpha=(0.008, 0.02),
        bounds=(-10.0, 20.0),
    )
    _recovers(target, phi, dataset_size=10000, known=known)


def test_unit_alpha_is_rejected():
    with pytest.raises(ValueError, match="nothing was truncated"):
        known_truncated_multivariate_normal.KnownTruncationMultivariateNormal(
            delphi.oracle.Identity(), 2, alpha=1.0
        )


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the learned window is too coarse to separate the pair",
    )
)
def test_indistinguishable_pair(known):
    # Daskalakis et al.'s example: cut to the same box, N((0, 1), I) and
    # N((0, 0), 4I) leave samples whose moments agree.
    targets = (
        torch.distributions.MultivariateNormal(ch.tensor([0.0, 1.0]), ch.eye(2)),
        torch.distributions.MultivariateNormal(ch.zeros(2), 4.0 * ch.eye(2)),
    )

    def box(x: ch.Tensor) -> ch.Tensor:
        return ((x > -0.5) & (x < 0.5)).all(dim=-1)

    fits = [_fit(target, box, 10000, known=known) for target in targets]
    helpers.assert_distinguishes(
        targets=targets,
        recovered=(fits[0][0]("avg"), fits[1][0]("avg")),
        naive=(fits[0][1], fits[1][1]),
        divergence=torch.distributions.kl.kl_divergence,
    )
