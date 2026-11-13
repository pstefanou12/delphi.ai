# Author: pstefanou12@
"""Tests for the truncated exponential distribution."""

import functools
from typing import Any, Optional

import pytest
import torch as ch
import torch.distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_exponential
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(
    rate: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: bool = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_exponential.TruncatedExponential,
        target=torch.distributions.Exponential(rate),
        phi=phi,
        parameter="rate_",
        empirical=lambda s: 1.0 / s.mean(0),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        known=known,
        **unknown,
    )


@helpers.known_and_unknown(pytest.mark.long, ci=True)
def test_one_dimension(known):
    _recovers(
        rate=ch.tensor([1.0]),
        phi=lambda z: z > 1,
        hyperparameters={"batch_size": 1},
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions(known):
    rate = ch.tensor([1.0, 2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.Exponential(rate),
        target_alpha=(0.15, 0.25),
    )
    _recovers(
        rate=rate,
        phi=calibrated.phi,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_twenty_dimensions(known):
    rate = 2 * ch.ones(20)
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.Exponential(rate),
        target_alpha=(0.15, 0.25),
    )
    _recovers(
        rate=rate,
        phi=calibrated.phi,
        hyperparameters={"iterations": 1500, "lr": 1e-1},
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_half_space(known):
    rate = ch.tensor([1.0, 2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 1, 2),
        torch.distributions.Exponential(rate),
        bounds=(-25.0, 25.0),
    )
    _recovers(
        rate=rate,
        phi=calibrated.phi,
        known=known,
    )


# Unknown truncation, rho 0.25: over seeds 0,1,2 the estimate's KL measured
# 0.143 to 0.175 (ema), 0.143 to 0.173 (avg).
@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_ellipsoid(known):
    rate = ch.tensor([1.0, 2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_ellipsoid, 0, 2),
        torch.distributions.Exponential(rate),
        bounds=(1e-4, 1e4),
    )
    # A bounded set leaves the empirical rate far from the truth, so the lr
    # the axis-aligned tests use still measures 0.3173 after 2500 steps.
    _recovers(
        rate=rate,
        phi=calibrated.phi,
        hyperparameters={"iterations": 5000, "lr": 1e-1, "truncation_set_rho": 0.25},
        known=known,
        unknown_tolerances={"ema": 0.35, "avg": 0.35},
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_identity(known):
    _recovers(
        rate=ch.tensor([1.0]),
        phi=delphi.oracle.Identity(),
        hyperparameters={"iterations": 1500},
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    _recovers(
        rate=ch.linspace(1.0, 2.0, 10),
        phi=delphi.oracle.Identity(),
        hyperparameters={"iterations": 1500},
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_sum_threshold(known):
    rate = ch.linspace(1.0, 2.0, 10)
    # One scalar condition, so sampling stays cheap, but it biases every
    # coordinate instead of one.
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.Exponential(rate),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 200.0),
    )
    _recovers(
        rate=rate,
        phi=calibrated.phi,
        hyperparameters={"lr": 1e-1},
        known=known,
    )


@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_small_mass(known):
    rate = ch.tensor([1.0])
    # The regime both papers state their guarantees in; nothing else in the
    # suite goes near it.
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.Exponential(rate),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 50.0),
    )
    _recovers(
        rate=rate,
        phi=calibrated.phi,
        known=known,
    )


# Unknown truncation needs rho 0.5 here; over seeds 0,1,2 it meets the shared
# bounds.
@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_norm_threshold(known):
    rate = ch.linspace(1.0, 2.0, 10)
    # Couples every coordinate like the sum does, and also excludes the mode.
    calibrated = truncation_sets.calibrate(
        truncation_sets.norm_threshold,
        torch.distributions.Exponential(rate),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 100.0),
    )
    _recovers(
        rate=rate,
        phi=calibrated.phi,
        hyperparameters={"lr": 1e-1, "truncation_set_rho": 0.5},
        known=known,
    )


def test_unit_alpha_is_rejected():
    with pytest.raises(ValueError, match="nothing was truncated"):
        truncated_exponential.TruncatedExponential(
            delphi.oracle.Identity(),
            alpha=1.0,
            hyperparameters={"iterations": 10},
        )


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the learned window is too coarse to separate the pair",
    )
)
def test_indistinguishable_pair(known):
    # Truncated to a narrow window the two samples agree to within sampling
    # error, so only an estimator that knows the window can tell them apart.
    helpers.assert_family_distinguishes(
        model_class=truncated_exponential.TruncatedExponential,
        targets=(
            torch.distributions.Exponential(ch.tensor([0.5])),
            torch.distributions.Exponential(ch.tensor([1.5])),
        ),
        phi=lambda z: (z[:, 0] > 1.0) & (z[:, 0] < 1.5),
        parameter="rate_",
        empirical=lambda s: 1.0 / s.mean(0),
        known=known,
    )


def test_hessian_refuses_a_theta_with_no_mass_on_the_set():
    ch.manual_seed(0)
    sample = helpers.generate_truncated_distribution_dataset(
        torch.distributions.Exponential(ch.tensor([1.0])), lambda z: z[:, 0] > 0.5, 2000
    )
    model = truncated_exponential.TruncatedExponential(
        lambda z: z[:, 0] > 0.5,
        sample.alpha,
        hyperparameters={"iterations": 10},
    )
    model.fit(sample.S)
    # Rate 100 puts exp(-50) of its mass above 0.5.
    with pytest.raises(ValueError, match="almost no mass on the truncation set"):
        model.hessian(ch.tensor([-100.0]))


def test_unknown_truncation_adopts_the_learned_set():
    ch.manual_seed(0)
    target = torch.distributions.Exponential(ch.tensor([1.0]))
    sample = helpers.generate_truncated_distribution_dataset(
        target, lambda z: z[:, 0] > 1.0, 2000
    )
    model = truncated_exponential.TruncatedExponential(
        None,
        hyperparameters={
            "iterations": 1,
            "max_phases": 1,
            "truncation_set_degree": 3,
        },
    ).fit(sample.S)
    assert isinstance(model.truncation_set_, delphi.oracle.PolynomialThreshold)
    assert model.phi is model.truncation_set_
    assert model.truncation_set_.polynomial.degree == 3
    fitted = model.train_loader_.dataset.S.size(0) + model.val_loader_.dataset.S.size(0)
    assert fitted == int(model.truncation_set_(sample.S).sum())
