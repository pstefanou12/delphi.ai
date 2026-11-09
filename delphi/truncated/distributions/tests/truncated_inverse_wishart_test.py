# Author: pstefanou12@
"""Tests for the truncated inverse Wishart distribution.

Samples are vech rows, so a set reads ``X_00, X_10, X_11, ...`` and
``coordinate_threshold(0, c)`` cuts the first diagonal entry.
"""

import functools
from typing import Any, Optional

import pytest
import torch as ch

from delphi.exponential_family import inverse_wishart
import delphi.oracle
from delphi.truncated.distributions import truncated_inverse_wishart
from delphi.truncated.distributions.tests import helpers, truncation_sets
from delphi.utils import helpers as helpers_lib

ch.set_printoptions(precision=4, sci_mode=False)

# With df 8 the mean is scale / 5, so this scale has mean [[2, .5], [.5, 1]].
_SCALE = ch.tensor([[10.0, 2.5], [2.5, 5.0]])


def _inverse_wishart(df: ch.Tensor, scale: ch.Tensor) -> inverse_wishart.InverseWishart:
    canonical = ch.cat([ch.atleast_1d(df), helpers_lib.vech(scale)])
    return inverse_wishart.InverseWishart(
        inverse_wishart.InverseWishart.to_natural(canonical), canonical.numel() - 1
    )


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    df: ch.Tensor,
    scale: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: helpers.Known = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_inverse_wishart.TruncatedInverseWishart,
        target=_inverse_wishart(df, scale),
        phi=phi,
        dims=scale.size(0),
        parameter=("df_", "scale_"),
        empirical=helpers.untruncated_mle(
            truncated_inverse_wishart.TruncatedInverseWishart, scale.size(0)
        ),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        rebuild=_inverse_wishart,
        known=known,
        **unknown,
    )


def _calibrated(set_family, df: ch.Tensor, scale: ch.Tensor, bounds, **kwargs):
    return truncation_sets.calibrate(
        set_family, _inverse_wishart(df, scale), bounds=bounds, **kwargs
    ).phi


@helpers.KNOWN_AND_UNKNOWN
def test_first_diagonal_threshold(known):
    df = ch.tensor([8.0])
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
    )


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the estimate's KL 0.71 is over half the naive fit's 1.24",
    )
)
def test_determinant_threshold(known):
    def phi(z):
        return ch.linalg.det(helpers_lib.unvech(z, 2)) < 0.5  # pylint: disable=not-callable

    _recovers(df=ch.tensor([8.0]), scale=_SCALE, phi=phi, known=known)


# Unknown truncation, degree 6: over seeds 0,1,2 the estimate's KL measured
# 0.278 to 0.365 (ema), 0.279 to 0.365 (avg).
@helpers.KNOWN_AND_UNKNOWN
def test_random_half_space(known):
    df = ch.tensor([8.0])
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
        unknown_tolerances={"ema": 0.73, "avg": 0.73},
        hyperparameters={"truncation_set_degree": 6},
    )


@helpers.KNOWN_AND_UNKNOWN_CI
def test_identity(known):
    _recovers(
        df=ch.tensor([8.0]),
        scale=_SCALE,
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_three_by_three_identity(known):
    _recovers(
        df=ch.tensor([9.0]),
        scale=5.0 * ch.tensor([[2.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 1.5]]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


# Unknown truncation: over seeds 0,1,2 the estimate's KL measured 0.236 to
# 0.306 (ema), 0.235 to 0.307 (avg); the set error 0.105 to 0.125.
@helpers.KNOWN_AND_UNKNOWN
def test_three_by_three_sum_threshold(known):
    df = ch.tensor([9.0])
    scale = 5.0 * ch.tensor([[2.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 1.5]])
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
        unknown_tolerances={"ema": 0.62, "avg": 0.62},
        set_tolerance=0.26,
    )


# Unknown truncation, rho 0.3: over seeds 0,1,2 the estimate's KL measured
# 0.160 to 0.387 (ema), 0.159 to 0.388 (avg).
@pytest.mark.slow
@helpers.KNOWN_AND_UNKNOWN
def test_small_mass(known):
    df = ch.tensor([8.0])
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
        unknown_tolerances={"ema": 0.78, "avg": 0.78},
        hyperparameters={"truncation_set_rho": 0.3},
    )


@pytest.mark.slow
@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the learned window is too coarse to separate the pair",
    )
)
def test_indistinguishable_pair(known):
    # Shifting theta by s (1, -vech(X*)) keeps grad log p at X*, so a small
    # box around X* hides df 8 from df 12.
    first = _inverse_wishart(ch.tensor([8.0]), _SCALE)
    center = _SCALE / 5.0
    shift = 2.0
    second = inverse_wishart.InverseWishart(
        first.theta + shift * ch.cat([ch.ones(1), -helpers_lib.vech(center)]), 3
    )
    point = helpers_lib.vech(center)
    ch.manual_seed(0)
    half_width = 0.3 * first.sample([20000]).std(0)

    def phi(z):
        return ((z - point).abs() <= half_width).all(-1)

    helpers.assert_family_distinguishes(
        model_class=truncated_inverse_wishart.TruncatedInverseWishart,
        targets=(first, second),
        phi=phi,
        dims=2,
        parameter=("df_", "scale_"),
        empirical=helpers.untruncated_mle(
            truncated_inverse_wishart.TruncatedInverseWishart, 2
        ),
        rebuild=_inverse_wishart,
        known=known,
    )


@pytest.mark.slow
@helpers.KNOWN_AND_UNKNOWN
def test_six_by_six_small_mass(known):
    # Twenty-one vech entries, so a sample lives in twenty-one dimensions.
    df = ch.tensor([14.0])
    off_diagonal = ch.full((5,), 0.3)
    scale = 7.0 * (ch.eye(6) + ch.diag(off_diagonal, 1) + ch.diag(off_diagonal, -1))
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
