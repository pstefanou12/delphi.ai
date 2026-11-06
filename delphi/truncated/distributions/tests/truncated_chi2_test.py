# Author: pstefanou12@
"""Tests for the truncated chi-squared distribution."""

import functools
from typing import Any, Optional

import pytest
import torch as ch
import torch.distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_chi2
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dims: int,
    df: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: bool = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_chi2.TruncatedChi2,
        target=torch.distributions.Chi2(df),
        phi=phi,
        dims=dims,
        parameter="df_",
        empirical=helpers.untruncated_mle(truncated_chi2.TruncatedChi2, dims),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        known=known,
        **unknown,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension(known):
    _recovers(dims=1, df=ch.tensor([4.0]), phi=lambda z: z[:, 0] > 5.0, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions(known):
    df = ch.tensor([4.0, 8.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.Chi2(df),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 100.0),
    )
    _recovers(dims=2, df=df, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_half_space(known):
    df = ch.tensor([4.0, 8.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        torch.distributions.Chi2(df),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(dims=2, df=df, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN_CI
def test_one_dimension_identity(known):
    _recovers(
        dims=1,
        df=ch.tensor([4.0]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    df = ch.linspace(2.0, 10.0, 10)
    _recovers(
        dims=10,
        df=df,
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_small_mass(known):
    df = ch.tensor([4.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.Chi2(df),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 100.0),
    )
    _recovers(dims=1, df=df, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_sum_threshold(known):
    df = ch.linspace(2.0, 10.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.Chi2(df),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 500.0),
    )
    _recovers(dims=10, df=df, phi=calibrated.phi, known=known)


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the learned window is too coarse to separate the pair",
    )
)
def test_indistinguishable_pair(known):
    helpers.assert_family_distinguishes(
        model_class=truncated_chi2.TruncatedChi2,
        targets=(
            torch.distributions.Chi2(ch.tensor([2.0])),
            torch.distributions.Chi2(ch.tensor([6.0])),
        ),
        phi=lambda z: (z[:, 0] > 2.0) & (z[:, 0] < 3.0),
        dims=1,
        parameter="df_",
        empirical=helpers.untruncated_mle(truncated_chi2.TruncatedChi2, 1),
        known=known,
    )
