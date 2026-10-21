# Author: pstefanou12@
"""Tests for the truncated chi-squared distribution."""

import functools
from typing import Optional

import torch as ch
from torch import distributions

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
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_chi2.TruncatedChi2,
        target=distributions.Chi2(df),
        phi=phi,
        dims=dims,
        parameter="df_",
        empirical=helpers.untruncated_mle(truncated_chi2.TruncatedChi2, dims),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
    )


def test_one_dimension():
    _recovers(dims=1, df=ch.Tensor([4.0]), phi=lambda z: z[:, 0] > 5.0)


def test_two_dimensions():
    df = ch.Tensor([4.0, 8.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Chi2(df),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 100.0),
    )
    _recovers(dims=2, df=df, phi=calibrated.phi)


def test_two_dimensions_random_half_space():
    df = ch.Tensor([4.0, 8.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        distributions.Chi2(df),
        target_alpha=(0.15, 0.25),
        bounds=(-50.0, 50.0),
    )
    _recovers(dims=2, df=df, phi=calibrated.phi)


def test_one_dimension_identity():
    _recovers(
        dims=1,
        df=ch.Tensor([4.0]),
        phi=truncation_sets.identity,
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    df = ch.linspace(2.0, 10.0, 10)
    _recovers(
        dims=10,
        df=df,
        phi=truncation_sets.identity,
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_one_dimension_small_mass():
    df = ch.Tensor([4.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Chi2(df),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 100.0),
    )
    _recovers(dims=1, df=df, phi=calibrated.phi)


def test_ten_dimensions_sum_threshold():
    df = ch.linspace(2.0, 10.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        distributions.Chi2(df),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 500.0),
    )
    _recovers(dims=10, df=df, phi=calibrated.phi)


def test_indistinguishable_pair():
    helpers.assert_family_distinguishes(
        model_class=truncated_chi2.TruncatedChi2,
        targets=(
            distributions.Chi2(ch.Tensor([2.0])),
            distributions.Chi2(ch.Tensor([6.0])),
        ),
        phi=lambda z: (z[:, 0] > 2.0) & (z[:, 0] < 3.0),
        dims=1,
        parameter="df_",
        empirical=helpers.untruncated_mle(truncated_chi2.TruncatedChi2, 1),
    )
