# Author: pstefanou12@
"""Tests for the multivariate normal with known covariance under a known set."""

import functools
import math
from typing import Optional

import pytest
import torch as ch
import torch.distributions
import torch.distributions.kl

import delphi.oracle
from delphi.truncated.distributions import (
    known_truncated_multivariate_normal_known_covariance as known_mvn_known_cov,
)
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _fit(
    target: torch.distributions.MultivariateNormal,
    phi: helpers.Oracle,
    dataset_size: int,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
):
    """Fit the mean on an unstandardized truncated sample, phi as stated.

    Returns:
        The estimate of each estimator, by name, and the naive fit: the
        truncated sample's own mean with the known covariance.
    """
    ch.manual_seed(0)
    S, alpha, _ = helpers.generate_truncated_distribution_dataset(  # pylint: disable=invalid-name
        target, phi, dataset_size
    )
    covariance = target.covariance_matrix
    args = (alpha,) if pass_alpha else ()
    model = known_mvn_known_cov.KnownTruncationMultivariateNormalKnownCovariance(
        phi, target.loc.numel(), covariance, *args, hyperparameters=hyperparameters
    )
    model.fit(S)

    def estimate(name: str) -> torch.distributions.MultivariateNormal:
        return torch.distributions.MultivariateNormal(
            getattr(model, f"{name}_loc_"), covariance
        )

    naive = torch.distributions.MultivariateNormal(S.mean(0), covariance)
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


def test_two_dimensions_crescent():
    _recovers(
        helpers.correlated_normal(2), truncation_sets.crescent(2.0, 6.0, 0.0, math.pi)
    )


def test_two_dimensions_triangle():
    _recovers(
        helpers.correlated_normal(2),
        truncation_sets.triangle([[0.0, 0.0], [4.0, 1.0], [1.0, 4.0]]),
    )


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
    _recovers(target, phi)


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
        known_mvn_known_cov.KnownTruncationMultivariateNormalKnownCovariance(
            delphi.oracle.Identity(), 2, ch.eye(2), alpha=1.0
        )


def test_indistinguishable_pair():
    # Cut to the same box, N((0, 1), I) and N((0, -1), I) leave samples whose
    # means nearly agree; only the set tells them apart.
    targets = (
        torch.distributions.MultivariateNormal(ch.tensor([0.0, 1.0]), ch.eye(2)),
        torch.distributions.MultivariateNormal(ch.tensor([0.0, -1.0]), ch.eye(2)),
    )

    def box(x: ch.Tensor) -> ch.Tensor:
        return ((x > -0.25) & (x < 0.25)).all(dim=-1)

    fits = [_fit(target, box, 10000) for target in targets]
    helpers.assert_distinguishes(
        targets=targets,
        recovered=(fits[0][0]("avg"), fits[1][0]("avg")),
        naive=(fits[0][1], fits[1][1]),
        divergence=torch.distributions.kl.kl_divergence,
    )


def test_unknown_truncation():
    # Over seeds 0-4 the estimate's divergence measured 0.014 to 0.034 and the
    # learned set's error 0.081 to 0.098.
    helpers.assert_family_learns_set(
        model_class=known_mvn_known_cov.KnownTruncationMultivariateNormalKnownCovariance,
        target=torch.distributions.MultivariateNormal(ch.zeros(2), ch.eye(2)),
        phi=lambda z: z[:, 0] + z[:, 1] > 0.0,
        dims=2,
        parameter="loc_",
        empirical=lambda s: s.mean(0),
        model_args=(ch.eye(2),),
        rebuild=lambda loc: torch.distributions.MultivariateNormal(loc, ch.eye(2)),
        tolerance=0.07,
        set_tolerance=0.2,
    )
