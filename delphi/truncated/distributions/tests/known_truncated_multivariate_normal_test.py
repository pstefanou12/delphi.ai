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
    model = known_truncated_multivariate_normal.KnownTruncationMultivariateNormal(
        phi_standardised, target.loc.numel(), *args, hyperparameters=hyperparameters
    )
    model.fit((S - loc) / scale)

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
) -> None:
    estimate, naive = _fit(target, phi, dataset_size, hyperparameters, pass_alpha)
    helpers.assert_recovers(
        estimate=estimate,
        divergence=lambda fitted: torch.distributions.kl.kl_divergence(fitted, target),
        baseline=naive,
        baseline_must_fail=baseline_must_fail,
        tolerances=tolerances,
    )


def _calibrated(
    factory, target: torch.distributions.MultivariateNormal, **kwargs
) -> helpers.Oracle:
    return truncation_sets.calibrate(factory, target, **kwargs).phi


def test_one_dimension():
    target = helpers.correlated_normal(1)
    phi = _calibrated(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        target,
        target_alpha=(0.15, 0.25),
        bounds=(-10.0, 10.0),
    )
    _recovers(target, phi)


def test_two_dimensions():
    target = helpers.correlated_normal(2)
    phi = _calibrated(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        target,
        target_alpha=(0.15, 0.25),
        bounds=(-10.0, 10.0),
    )
    _recovers(target, phi)


def test_two_dimensions_random_half_space():
    target = helpers.correlated_normal(2)
    phi = _calibrated(
        functools.partial(truncation_sets.random_half_space, 1, 2),
        target,
        bounds=(-25.0, 25.0),
    )
    _recovers(target, phi)


def test_two_dimensions_annulus():
    _recovers(helpers.correlated_normal(2), truncation_sets.annulus(2.5, 5.0))


def test_two_dimensions_crescent():
    _recovers(
        helpers.correlated_normal(2), truncation_sets.crescent(2.0, 6.0, 0.0, math.pi)
    )


def test_two_dimensions_random_disc_union():
    target = helpers.correlated_normal(2)
    phi = _calibrated(
        functools.partial(truncation_sets.random_disc_union, 0, 2, 6),
        target,
        bounds=(1e-4, 1e3),
    )
    _recovers(target, phi)


def test_ten_dimensions():
    target = helpers.correlated_normal(10)
    phi = _calibrated(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        target,
        target_alpha=(0.15, 0.25),
        bounds=(-10.0, 10.0),
    )
    _recovers(target, phi)


def test_ten_dimensions_sum_threshold():
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
    _recovers(target, phi, dataset_size=20000)


def test_ten_dimensions_norm_threshold():
    target = helpers.correlated_normal(10)
    # Couples every coordinate like the sum does, and also excludes the mode.
    phi = _calibrated(
        truncation_sets.norm_threshold,
        target,
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 100.0),
    )
    _recovers(target, phi)


@pytest.mark.slow
def test_twenty_dimensions():
    target = helpers.correlated_normal(20)
    phi = _calibrated(
        truncation_sets.sum_threshold,
        target,
        target_alpha=(0.15, 0.25),
        bounds=(-100.0, 100.0),
    )
    _recovers(target, phi, dataset_size=10000)


def test_one_dimension_identity():
    _recovers(
        helpers.correlated_normal(1),
        delphi.oracle.Identity(),
        hyperparameters={"iterations": 1500, "num_samples": 10000},
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    _recovers(
        helpers.correlated_normal(10),
        delphi.oracle.Identity(),
        hyperparameters={"iterations": 1500, "num_samples": 10000},
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_one_dimension_small_mass():
    target = helpers.correlated_normal(1)
    # The regime the paper states its guarantee in; nothing else goes near it.
    phi = _calibrated(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        target,
        target_alpha=(0.008, 0.02),
        bounds=(-10.0, 20.0),
    )
    _recovers(target, phi, dataset_size=10000)


def test_unit_alpha_is_rejected():
    with pytest.raises(ValueError, match="nothing was truncated"):
        known_truncated_multivariate_normal.KnownTruncationMultivariateNormal(
            delphi.oracle.Identity(), 2, alpha=1.0
        )


def test_indistinguishable_pair():
    # Daskalakis et al.'s example: cut to the same box, N((0, 1), I) and
    # N((0, 0), 4I) leave samples whose moments agree.
    targets = (
        torch.distributions.MultivariateNormal(ch.tensor([0.0, 1.0]), ch.eye(2)),
        torch.distributions.MultivariateNormal(ch.zeros(2), 4.0 * ch.eye(2)),
    )

    def box(x: ch.Tensor) -> ch.Tensor:
        return ((x > -0.5) & (x < 0.5)).all(dim=-1)

    fits = [_fit(target, box, 10000) for target in targets]
    helpers.assert_distinguishes(
        targets=targets,
        recovered=(fits[0][0]("avg"), fits[1][0]("avg")),
        naive=(fits[0][1], fits[1][1]),
        divergence=torch.distributions.kl.kl_divergence,
    )
