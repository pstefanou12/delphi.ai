# Author: pstefanou12@
"""Tests for the truncated von Mises distribution."""

import functools
import math
from typing import Any, Optional

import pytest
import torch as ch
import torch.distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_von_mises
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def von_mises_kl_divergence(
    fitted: torch.distributions.Distribution, target: torch.distributions.Distribution
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
    loc: ch.Tensor,
    concentration: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: bool = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_von_mises.TruncatedVonMises,
        target=torch.distributions.VonMises(loc, concentration),
        phi=phi,
        parameter=("loc_", "concentration_"),
        empirical=helpers.untruncated_mle(truncated_von_mises.TruncatedVonMises),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        divergence=von_mises_kl_divergence,
        known=known,
        **unknown,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension(known):
    _recovers(
        loc=ch.tensor([0.0]),
        concentration=ch.tensor([2.0]),
        phi=lambda z: z[:, 0] > 0.3,
        known=known,
    )


# Unknown truncation needs rho 0.3 here; over seeds 0,1,2 it meets the shared
# bounds.
@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions(known):
    loc, concentration = ch.tensor([0.0, 1.0]), ch.tensor([2.0, 1.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.VonMises(loc, concentration),
        target_alpha=(0.15, 0.25),
        bounds=(-math.pi, math.pi),
    )
    _recovers(
        loc=loc,
        concentration=concentration,
        phi=calibrated.phi,
        known=known,
        hyperparameters={"truncation_set_rho": 0.3},
    )


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the estimate's KL reaches 0.59 against the naive fit's 1.15",
    )
)
def test_two_dimensions_random_half_space(known):
    loc, concentration = ch.tensor([0.0, 1.0]), ch.tensor([2.0, 1.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        torch.distributions.VonMises(loc, concentration),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(loc=loc, concentration=concentration, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN_CI
def test_one_dimension_identity(known):
    _recovers(
        loc=ch.tensor([0.0]),
        concentration=ch.tensor([2.0]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    loc, concentration = ch.linspace(-1.0, 1.0, 10), ch.linspace(0.5, 3.0, 10)
    _recovers(
        loc=loc,
        concentration=concentration,
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@pytest.mark.slow
@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_small_mass(known):
    loc, concentration = ch.tensor([0.0]), ch.tensor([2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.VonMises(loc, concentration),
        target_alpha=(0.008, 0.02),
        bounds=(-math.pi, math.pi),
    )
    # At 1% mass the default 2500 steps end at ema KL 0.124; 5000 reach 0.037.
    _recovers(
        loc=loc,
        concentration=concentration,
        phi=calibrated.phi,
        hyperparameters={"iterations": 5000},
        known=known,
    )


@pytest.mark.slow
@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_sum_threshold(known):
    loc, concentration = ch.linspace(-1.0, 1.0, 10), ch.linspace(0.5, 3.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.VonMises(loc, concentration),
        target_alpha=(0.15, 0.25),
        bounds=(-10 * math.pi, 10 * math.pi),
    )
    _recovers(loc=loc, concentration=concentration, phi=calibrated.phi, known=known)


# Unknown truncation needs rho 0.3 here; over seeds 0,1,2 it meets the shared
# bounds.
@helpers.KNOWN_AND_UNKNOWN
def test_indistinguishable_pair(known):
    helpers.assert_family_distinguishes(
        model_class=truncated_von_mises.TruncatedVonMises,
        targets=(
            torch.distributions.VonMises(ch.tensor([0.0]), ch.tensor([2.0])),
            torch.distributions.VonMises(ch.tensor([0.0]), ch.tensor([0.5])),
        ),
        phi=lambda z: (z[:, 0] > -1.0) & (z[:, 0] < 1.0),
        parameter=("loc_", "concentration_"),
        empirical=helpers.untruncated_mle(truncated_von_mises.TruncatedVonMises),
        divergence=von_mises_kl_divergence,
        known=known,
        hyperparameters={"truncation_set_rho": 0.3},
    )
