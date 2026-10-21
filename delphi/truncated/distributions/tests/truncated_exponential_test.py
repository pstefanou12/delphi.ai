# Author: pstefanou12@
"""Tests for the truncated exponential distribution."""

import functools
from typing import Optional

import pytest
import torch as ch
from torch import distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_exponential
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(
    dims: int,
    rate: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_exponential.TruncatedExponential,
        target=distributions.Exponential(rate),
        phi=phi,
        dims=dims,
        parameter="rate_",
        empirical=lambda s: 1.0 / s.mean(0),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
    )


def test_one_dimension():
    _recovers(
        dims=1,
        rate=ch.Tensor([1.0]),
        phi=lambda z: z > 1,
        hyperparameters={"batch_size": 1},
    )


def test_two_dimensions():
    rate = ch.Tensor([1.0, 2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Exponential(rate),
        target_alpha=(0.15, 0.25),
    )
    _recovers(
        dims=2,
        rate=rate,
        phi=calibrated.phi,
    )


def test_twenty_dimensions():
    rate = 2 * ch.ones(20)
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Exponential(rate),
        target_alpha=(0.15, 0.25),
    )
    _recovers(
        dims=20,
        rate=rate,
        phi=calibrated.phi,
        hyperparameters={"iterations": 1500, "lr": 1e-1},
    )


def test_two_dimensions_random_half_space():
    rate = ch.Tensor([1.0, 2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 1, 2),
        distributions.Exponential(rate),
        bounds=(-25.0, 25.0),
    )
    _recovers(
        dims=2,
        rate=rate,
        phi=calibrated.phi,
    )


def test_two_dimensions_random_ellipsoid():
    rate = ch.Tensor([1.0, 2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_ellipsoid, 0, 2),
        distributions.Exponential(rate),
        bounds=(1e-4, 1e4),
    )
    # A bounded set leaves the empirical rate far from the truth, so the lr
    # the axis-aligned tests use still measures 0.3173 after 2500 steps.
    _recovers(
        dims=2,
        rate=rate,
        phi=calibrated.phi,
        hyperparameters={"iterations": 5000, "lr": 1e-1},
    )


def test_one_dimension_identity():
    _recovers(
        dims=1,
        rate=ch.Tensor([1.0]),
        phi=delphi.oracle.Identity(),
        hyperparameters={"iterations": 1500},
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    _recovers(
        dims=10,
        rate=ch.linspace(1.0, 2.0, 10),
        phi=delphi.oracle.Identity(),
        hyperparameters={"iterations": 1500},
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_sum_threshold():
    rate = ch.linspace(1.0, 2.0, 10)
    # One scalar condition, so sampling stays cheap, but it biases every
    # coordinate instead of one.
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        distributions.Exponential(rate),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 200.0),
    )
    _recovers(
        dims=10,
        rate=rate,
        phi=calibrated.phi,
        hyperparameters={"lr": 1e-1},
    )


def test_one_dimension_small_mass():
    rate = ch.Tensor([1.0])
    # The regime both papers state their guarantees in; nothing else in the
    # suite goes near it.
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Exponential(rate),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 50.0),
    )
    _recovers(
        dims=1,
        rate=rate,
        phi=calibrated.phi,
    )


def test_ten_dimensions_norm_threshold():
    rate = ch.linspace(1.0, 2.0, 10)
    # Couples every coordinate like the sum does, and also excludes the mode.
    calibrated = truncation_sets.calibrate(
        truncation_sets.norm_threshold,
        distributions.Exponential(rate),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 100.0),
    )
    _recovers(
        dims=10,
        rate=rate,
        phi=calibrated.phi,
        hyperparameters={"lr": 1e-1},
    )


def test_unit_alpha_is_rejected():
    with pytest.raises(ValueError, match="nothing was truncated"):
        truncated_exponential.TruncatedExponential(
            {"iterations": 10},
            delphi.oracle.Identity(),
            1,
            alpha=1.0,
        )


def test_indistinguishable_pair():
    # Truncated to a narrow window the two samples agree to within sampling
    # error, so only an estimator that knows the window can tell them apart.
    helpers.assert_family_distinguishes(
        model_class=truncated_exponential.TruncatedExponential,
        targets=(
            distributions.Exponential(ch.Tensor([0.5])),
            distributions.Exponential(ch.Tensor([1.5])),
        ),
        phi=lambda z: (z[:, 0] > 1.0) & (z[:, 0] < 1.5),
        dims=1,
        parameter="rate_",
        empirical=lambda s: 1.0 / s.mean(0),
    )


def test_hessian_refuses_a_theta_with_no_mass_on_the_set():
    ch.manual_seed(0)
    sample = helpers.generate_truncated_distribution_dataset(
        distributions.Exponential(ch.Tensor([1.0])), lambda z: z[:, 0] > 0.5, 2000
    )
    model = truncated_exponential.TruncatedExponential(
        {"iterations": 10},
        lambda z: z[:, 0] > 0.5,
        1,
        sample.alpha,
    )
    model.fit(sample.S)
    # Rate 100 puts exp(-50) of its mass above 0.5.
    with pytest.raises(ValueError, match="almost no mass on the truncation set"):
        model.hessian(ch.Tensor([-100.0]))
