# Author: pstefanou12@
"""Tests for the truncated von Mises distribution."""

import functools
import math
from typing import Optional

import pytest
import torch as ch
from torch import distributions

from delphi.truncated.distributions import truncated_von_mises
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def von_mises_kl_divergence(
    fitted: distributions.Distribution, target: distributions.Distribution
) -> ch.Tensor:
    """Return KL between two von Mises distributions.

    torch registers none. With ``A(k) = I1(k) / I0(k)``, it is
    ``log I0(k2) - log I0(k1) + A(k1) (k1 - k2 cos(loc1 - loc2))``. Checked
    against 4e5 Monte Carlo draws to 1e-3.

    Args:
        fitted: First distribution.
        target: Second distribution.

    Returns:
        Per-coordinate KL divergence.
    """
    # pylint: disable=not-callable  # torch.special is C-bound; pylint can't see it.
    first, second = fitted.concentration, target.concentration  # type: ignore[attr-defined]
    shift = fitted.loc - target.loc  # type: ignore[attr-defined]

    def log_i0(concentration):
        return ch.log(ch.special.i0e(concentration)) + concentration

    ratio = ch.special.i1e(first) / ch.special.i0e(first)
    return log_i0(second) - log_i0(first) + ratio * (first - second * ch.cos(shift))


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
        model_class=truncated_von_mises.TruncatedVonMises,
        target=distributions.VonMises(loc, concentration),
        phi=phi,
        dims=dims,
        parameter=("loc_", "concentration_"),
        empirical=helpers.untruncated_mle(truncated_von_mises.TruncatedVonMises, dims),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        divergence=von_mises_kl_divergence,
    )


def test_one_dimension():
    _recovers(
        dims=1,
        loc=ch.Tensor([0.0]),
        concentration=ch.Tensor([2.0]),
        phi=lambda z: z[:, 0] > 0.3,
    )


def test_two_dimensions():
    loc, concentration = ch.Tensor([0.0, 1.0]), ch.Tensor([2.0, 1.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.VonMises(loc, concentration),
        target_alpha=(0.15, 0.25),
        bounds=(-math.pi, math.pi),
    )
    _recovers(dims=2, loc=loc, concentration=concentration, phi=calibrated.phi)


def test_two_dimensions_random_half_space():
    loc, concentration = ch.Tensor([0.0, 1.0]), ch.Tensor([2.0, 1.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        distributions.VonMises(loc, concentration),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(dims=2, loc=loc, concentration=concentration, phi=calibrated.phi)


def test_one_dimension_identity():
    _recovers(
        dims=1,
        loc=ch.Tensor([0.0]),
        concentration=ch.Tensor([2.0]),
        phi=truncation_sets.identity,
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    loc, concentration = ch.linspace(-1.0, 1.0, 10), ch.linspace(0.5, 3.0, 10)
    _recovers(
        dims=10,
        loc=loc,
        concentration=concentration,
        phi=truncation_sets.identity,
        pass_alpha=False,
        baseline_must_fail=False,
    )


@pytest.mark.slow
def test_one_dimension_small_mass():
    loc, concentration = ch.Tensor([0.0]), ch.Tensor([2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.VonMises(loc, concentration),
        target_alpha=(0.008, 0.02),
        bounds=(-math.pi, math.pi),
    )
    # At 1% mass the default 2500 steps end at ema KL 0.124; 5000 reach 0.037.
    _recovers(
        dims=1,
        loc=loc,
        concentration=concentration,
        phi=calibrated.phi,
        hyperparameters={"iterations": 5000},
    )


@pytest.mark.slow
def test_ten_dimensions_sum_threshold():
    loc, concentration = ch.linspace(-1.0, 1.0, 10), ch.linspace(0.5, 3.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        distributions.VonMises(loc, concentration),
        target_alpha=(0.15, 0.25),
        bounds=(-10 * math.pi, 10 * math.pi),
    )
    _recovers(dims=10, loc=loc, concentration=concentration, phi=calibrated.phi)


def test_indistinguishable_pair():
    helpers.assert_family_distinguishes(
        model_class=truncated_von_mises.TruncatedVonMises,
        targets=(
            distributions.VonMises(ch.Tensor([0.0]), ch.Tensor([2.0])),
            distributions.VonMises(ch.Tensor([0.0]), ch.Tensor([0.5])),
        ),
        phi=lambda z: (z[:, 0] > -1.0) & (z[:, 0] < 1.0),
        dims=1,
        parameter=("loc_", "concentration_"),
        empirical=helpers.untruncated_mle(truncated_von_mises.TruncatedVonMises, 1),
        divergence=von_mises_kl_divergence,
    )
