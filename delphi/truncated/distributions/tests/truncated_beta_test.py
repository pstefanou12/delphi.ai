# Author: pstefanou12@
"""Tests for the truncated Beta distribution."""

import functools
from typing import Any, Optional

import pytest
import torch as ch
import torch.distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_beta
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _moments(s: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
    mean, var = s.mean(0), s.var(0)
    common = mean * (1 - mean) / var - 1
    return mean * common, (1 - mean) * common


def _recovers(
    dims: int,
    first: ch.Tensor,
    second: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: bool = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_beta.TruncatedBeta,
        target=torch.distributions.Beta(first, second),
        phi=phi,
        dims=dims,
        parameter=("concentration1_", "concentration0_"),
        empirical=_moments,
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        known=known,
        **unknown,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension(known):
    _recovers(
        dims=1,
        first=ch.tensor([2.0]),
        second=ch.tensor([3.0]),
        phi=lambda z: z[:, 0] > 0.5,
        known=known,
    )


# Unknown truncation: over seeds 0,1,2 the estimate's KL measured 0.206 to
# 0.257 (ema), 0.235 to 0.279 (avg).
@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_half_space(known):
    first, second = ch.tensor([2.0, 4.0]), ch.tensor([3.0, 2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        torch.distributions.Beta(first, second),
        bounds=(-25.0, 25.0),
    )
    _recovers(
        dims=2,
        first=first,
        second=second,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.52, "avg": 0.56},
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_identity(known):
    _recovers(
        dims=1,
        first=ch.tensor([2.0]),
        second=ch.tensor([3.0]),
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
        first=ch.linspace(1.5, 3.0, 10),
        second=ch.linspace(2.0, 4.0, 10),
        phi=delphi.oracle.Identity(),
        hyperparameters={"iterations": 1500},
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.known_and_unknown(pytest.mark.slow)
def test_ten_dimensions_sum_threshold(known):
    first, second = ch.linspace(1.5, 3.0, 10), ch.linspace(2.0, 4.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.Beta(first, second),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 10.0),
    )
    _recovers(
        dims=10,
        first=first,
        second=second,
        phi=calibrated.phi,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_indistinguishable_pair(known):
    helpers.assert_family_distinguishes(
        model_class=truncated_beta.TruncatedBeta,
        # Symmetric about the window, so neither slopes across it, and wide
        # enough to carry the curvature a second parameter needs: on
        # (0.45, 0.55) the Cramer-Rao bound on a is 3.35, more than a itself.
        targets=(
            torch.distributions.Beta(ch.tensor([2.0]), ch.tensor([2.0])),
            torch.distributions.Beta(ch.tensor([5.0]), ch.tensor([5.0])),
        ),
        phi=lambda z: (z[:, 0] > 0.25) & (z[:, 0] < 0.75),
        dims=1,
        parameter=("concentration1_", "concentration0_"),
        empirical=_moments,
        known=known,
    )
