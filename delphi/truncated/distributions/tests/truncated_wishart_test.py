# Author: pstefanou12@
"""Tests for the truncated Wishart distribution.

Samples are vech rows, so a set reads ``X_00, X_10, X_11, ...`` and
``coordinate_threshold(0, c)`` cuts the first diagonal entry.
"""

import functools
from typing import Any, Optional

import pytest
import torch as ch

from delphi.exponential_family import wishart
import delphi.oracle
from delphi.truncated.distributions import truncated_wishart
from delphi.truncated.distributions.tests import helpers, truncation_sets
from delphi.utils import helpers as helpers_lib

ch.set_printoptions(precision=4, sci_mode=False)

_SCALE = ch.tensor([[2.0, 0.5], [0.5, 1.0]])


def _wishart(df: ch.Tensor, scale: ch.Tensor) -> wishart.Wishart:
    canonical = ch.cat([ch.atleast_1d(df), helpers_lib.vech(scale)])
    return wishart.Wishart(wishart.Wishart.to_natural(canonical), canonical.numel() - 1)


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    df: ch.Tensor,
    scale: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: bool = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_wishart.TruncatedWishart,
        target=_wishart(df, scale),
        phi=phi,
        dims=scale.size(0),
        parameter=("df_", "covariance_matrix_"),
        empirical=helpers.untruncated_mle(
            truncated_wishart.TruncatedWishart, scale.size(0)
        ),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        rebuild=_wishart,
        known=known,
        **unknown,
    )


def _calibrated(set_family, df: ch.Tensor, scale: ch.Tensor, bounds, **kwargs):
    return truncation_sets.calibrate(
        set_family, _wishart(df, scale), bounds=bounds, **kwargs
    ).phi


# Unknown truncation needs rho 0.3 here; over seeds 0,1,2 it meets the shared
# bounds.
@helpers.KNOWN_AND_UNKNOWN
def test_first_diagonal_threshold(known):
    df = ch.tensor([5.0])
    _recovers(
        df=df,
        scale=_SCALE,
        phi=_calibrated(
            functools.partial(truncation_sets.coordinate_threshold, 0),
            df,
            _SCALE,
            bounds=(0.0, 200.0),
            target_alpha=(0.15, 0.25),
        ),
        known=known,
        hyperparameters={"truncation_set_rho": 0.3},
    )


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the estimate's KL 0.78 is over half the naive fit's 1.44",
    )
)
def test_determinant_threshold(known):
    def phi(z):
        return ch.linalg.det(helpers_lib.unvech(z, 2)) > 60.0  # pylint: disable=not-callable

    _recovers(df=ch.tensor([5.0]), scale=_SCALE, phi=phi, known=known)


# Unknown truncation: over seeds 0,1,2 the estimate's KL measured 0.193 to
# 0.220 (ema), 0.192 to 0.220 (avg).
@helpers.KNOWN_AND_UNKNOWN
def test_random_half_space(known):
    df = ch.tensor([5.0])
    _recovers(
        df=df,
        scale=_SCALE,
        phi=_calibrated(
            functools.partial(truncation_sets.random_half_space, 0, 3),
            df,
            _SCALE,
            bounds=(-200.0, 200.0),
            target_alpha=(0.15, 0.25),
        ),
        known=known,
        unknown_tolerances={"ema": 0.45, "avg": 0.45},
    )


@helpers.KNOWN_AND_UNKNOWN_CI
def test_identity(known):
    _recovers(
        df=ch.tensor([5.0]),
        scale=_SCALE,
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_three_by_three_identity(known):
    _recovers(
        df=ch.tensor([6.0]),
        scale=ch.tensor([[2.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 1.5]]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_three_by_three_sum_threshold(known):
    df = ch.tensor([6.0])
    scale = ch.tensor([[2.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 1.5]])
    _recovers(
        df=df,
        scale=scale,
        phi=_calibrated(
            truncation_sets.sum_threshold,
            df,
            scale,
            bounds=(0.0, 500.0),
            target_alpha=(0.15, 0.25),
        ),
        known=known,
    )


@pytest.mark.slow
@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the estimate's KL reaches 2.37 against the naive fit's 5.18",
    )
)
def test_small_mass(known):
    df = ch.tensor([5.0])
    _recovers(
        df=df,
        scale=_SCALE,
        phi=_calibrated(
            functools.partial(truncation_sets.coordinate_threshold, 0),
            df,
            _SCALE,
            bounds=(0.0, 200.0),
            target_alpha=(0.008, 0.02),
        ),
        known=known,
    )


@pytest.mark.slow
@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the learned window is too coarse to separate the pair",
    )
)
def test_indistinguishable_pair(known):
    # Shifting theta by s (1, -vech(X*^-1)) keeps grad log p at X*, so a
    # small box around X* hides df 5 from df 9.
    first = _wishart(ch.tensor([5.0]), _SCALE)
    center = 5.0 * _SCALE
    shift = 2.0
    second = wishart.Wishart(
        first.theta
        + ch.cat([ch.tensor([shift]), -shift * helpers_lib.vech(center.inverse())]),
        3,
    )
    point = helpers_lib.vech(center)
    # Var(X_ij) = df (scale_ij^2 + scale_ii scale_jj).
    variance = 5.0 * (
        helpers_lib.vech(_SCALE) ** 2
        + helpers_lib.vech(ch.outer(_SCALE.diagonal(), _SCALE.diagonal()))
    )
    half_width = 0.5 * variance.sqrt()

    def phi(z):
        return ((z - point).abs() <= half_width).all(-1)

    helpers.assert_family_distinguishes(
        model_class=truncated_wishart.TruncatedWishart,
        targets=(first, second),
        phi=phi,
        dims=2,
        parameter=("df_", "covariance_matrix_"),
        empirical=helpers.untruncated_mle(truncated_wishart.TruncatedWishart, 2),
        rebuild=_wishart,
        known=known,
    )


@pytest.mark.slow
@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the estimate's KL 2.98 is over half the naive fit's 3.23",
    )
)
def test_six_by_six_small_mass(known):
    # Twenty-one vech entries, so a sample lives in twenty-one dimensions.
    df = ch.tensor([10.0])
    off_diagonal = ch.full((5,), 0.3)
    scale = ch.eye(6) + ch.diag(off_diagonal, 1) + ch.diag(off_diagonal, -1)
    _recovers(
        df=df,
        scale=scale,
        phi=_calibrated(
            truncation_sets.sum_threshold,
            df,
            scale,
            bounds=(0.0, 2000.0),
            target_alpha=(0.008, 0.02),
        ),
        known=known,
    )
