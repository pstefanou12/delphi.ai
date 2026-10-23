# Author: pstefanou12@
"""Tests for the truncated inverse Gaussian distribution."""

import functools
from typing import Optional

import torch as ch
from torch import distributions

from delphi.exponential_family import inverse_gaussian
import delphi.oracle
from delphi.truncated.distributions import truncated_inverse_gaussian
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)

_FAMILY = inverse_gaussian.InverseGaussian


def _wald(loc: ch.Tensor, concentration: ch.Tensor) -> _FAMILY:
    return _FAMILY(_FAMILY.to_natural(ch.cat([loc, concentration])), loc.numel())


def _kl(
    fitted: distributions.Distribution, target: distributions.Distribution
) -> ch.Tensor:
    """Return the per-coordinate KL, which torch does not register.

    ``(log(l1 / l2) + l2 / l1 - 1) / 2 + l2 (m1 - m2)^2 / (2 m1 m2^2)`` for
    means m and shapes l, checked against 2e6 Monte Carlo draws to 7e-4.
    """
    assert isinstance(fitted, _FAMILY) and isinstance(target, _FAMILY)
    loc, concentration = fitted.loc, fitted.concentration
    other_loc, other_concentration = target.loc, target.concentration
    ratio = other_concentration / concentration
    return 0.5 * (-ch.log(ratio) + ratio - 1.0) + other_concentration * (
        loc - other_loc
    ) ** 2 / (2.0 * loc * other_loc**2)


def _mle(s: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
    loc = s.mean(0)
    return loc, 1.0 / (1.0 / s - 1.0 / loc).mean(0)


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dims: int,
    loc: ch.Tensor,
    concentration: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_inverse_gaussian.TruncatedInverseGaussian,
        target=_wald(loc, concentration),
        phi=phi,
        dims=dims,
        parameter=("loc_", "concentration_"),
        empirical=_mle,
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        rebuild=_wald,
        divergence=_kl,
    )


def test_one_dimension():
    _recovers(
        dims=1,
        loc=ch.Tensor([1.0]),
        concentration=ch.Tensor([2.0]),
        phi=lambda z: z[:, 0] > 1.5,
    )


def test_two_dimensions():
    loc, concentration = ch.Tensor([1.0, 2.0]), ch.Tensor([2.0, 4.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        _wald(loc, concentration),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 50.0),
    )
    _recovers(dims=2, loc=loc, concentration=concentration, phi=calibrated.phi)


def test_two_dimensions_random_half_space():
    loc, concentration = ch.Tensor([1.0, 2.0]), ch.Tensor([2.0, 4.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        _wald(loc, concentration),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(dims=2, loc=loc, concentration=concentration, phi=calibrated.phi)


def test_one_dimension_identity():
    _recovers(
        dims=1,
        loc=ch.Tensor([1.0]),
        concentration=ch.Tensor([2.0]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    _recovers(
        dims=10,
        loc=ch.linspace(0.5, 2.0, 10),
        concentration=ch.linspace(1.0, 5.0, 10),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_one_dimension_small_mass():
    loc, concentration = ch.Tensor([1.0]), ch.Tensor([2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        _wald(loc, concentration),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 50.0),
    )
    _recovers(dims=1, loc=loc, concentration=concentration, phi=calibrated.phi)


def test_ten_dimensions_sum_threshold():
    loc, concentration = ch.linspace(0.5, 2.0, 10), ch.linspace(1.0, 5.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        _wald(loc, concentration),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 200.0),
    )
    _recovers(dims=10, loc=loc, concentration=concentration, phi=calibrated.phi)


def test_indistinguishable_pair():
    helpers.assert_family_distinguishes(
        model_class=truncated_inverse_gaussian.TruncatedInverseGaussian,
        targets=(
            _wald(ch.Tensor([1.0]), ch.Tensor([3.0])),
            _wald(ch.Tensor([3.0]), ch.Tensor([1.0])),
        ),
        phi=lambda z: (z[:, 0] > 0.5) & (z[:, 0] < 1.5),
        dims=1,
        parameter=("loc_", "concentration_"),
        empirical=_mle,
        rebuild=_wald,
        divergence=_kl,
    )
