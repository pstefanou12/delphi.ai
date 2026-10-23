# Author: pstefanou12@
"""Tests for the truncated Wishart distribution.

Samples are vech rows, so a set reads ``X_00, X_10, X_11, ...`` and
``coordinate_threshold(0, c)`` cuts the first diagonal entry.
"""

import functools
from typing import Optional

import pytest
import torch as ch
from torch import distributions

from delphi.exponential_family import wishart
import delphi.oracle
from delphi.truncated.distributions import truncated_wishart
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)

_SCALE = ch.Tensor([[2.0, 0.5], [0.5, 1.0]])


def _wishart(df: ch.Tensor, scale: ch.Tensor) -> wishart.Wishart:
    canonical = ch.cat([ch.atleast_1d(df), wishart.to_vech(scale)])
    return wishart.Wishart(wishart.Wishart.to_natural(canonical), canonical.numel() - 1)


def _kl(
    fitted: distributions.Distribution, target: distributions.Distribution
) -> ch.Tensor:
    """Return the KL, which torch does not register, for df n and scale V.

    ``n1 / 2 (tr(V2^-1 V1) - p) + n2 / 2 log(|V2| / |V1|)
    + log G_p(n2 / 2) - log G_p(n1 / 2) + (n1 - n2) / 2 psi_p(n1 / 2)``,
    checked against 4e5 Monte Carlo draws to 2e-3.
    """
    assert isinstance(fitted, wishart.Wishart)
    assert isinstance(target, wishart.Wishart)
    df, other_df = fitted.df, target.df
    scale, other_scale = fitted.covariance_matrix, target.covariance_matrix
    size = scale.size(-1)
    offsets = ch.arange(size, dtype=scale.dtype) / 2.0
    return (
        0.5 * df * (ch.linalg.solve(other_scale, scale).trace() - size)  # pylint: disable=not-callable
        + 0.5 * other_df * (ch.logdet(other_scale) - ch.logdet(scale))
        + ch.mvlgamma(other_df / 2.0, size)
        - ch.mvlgamma(df / 2.0, size)
        + 0.5 * (df - other_df) * ch.digamma(df / 2.0 - offsets).sum()
    )


def _untruncated_mle(sample: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
    size = wishart.side(sample.size(1))
    # The warm start is the estimate; one phase of one step completes the fit.
    model = truncated_wishart.TruncatedWishart(
        {"iterations": 1, "max_phases": 1}, delphi.oracle.Identity(), size
    )
    model.fit(sample)
    assert model.emp_theta is not None
    df, scale = model._canonical(model.emp_theta)  # pylint: disable=protected-access
    return df.detach(), scale.detach()


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    df: ch.Tensor,
    scale: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_wishart.TruncatedWishart,
        target=_wishart(df, scale),
        phi=phi,
        dims=scale.size(0),
        parameter=("df_", "covariance_matrix_"),
        empirical=_untruncated_mle,
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        rebuild=_wishart,
        divergence=_kl,
    )


def _calibrated(set_family, df: ch.Tensor, scale: ch.Tensor, bounds, **kwargs):
    return truncation_sets.calibrate(
        set_family, _wishart(df, scale), bounds=bounds, **kwargs
    ).phi


def test_first_diagonal_threshold():
    df = ch.Tensor([5.0])
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
    )


def test_determinant_threshold():
    def phi(z):
        return ch.linalg.det(wishart.from_vech(z)) > 60.0  # pylint: disable=not-callable

    _recovers(df=ch.Tensor([5.0]), scale=_SCALE, phi=phi)


def test_random_half_space():
    df = ch.Tensor([5.0])
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
    )


def test_identity():
    _recovers(
        df=ch.Tensor([5.0]),
        scale=_SCALE,
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_three_by_three_identity():
    _recovers(
        df=ch.Tensor([6.0]),
        scale=ch.Tensor([[2.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 1.5]]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_three_by_three_sum_threshold():
    df = ch.Tensor([6.0])
    scale = ch.Tensor([[2.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 1.5]])
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
    )


@pytest.mark.slow
def test_small_mass():
    df = ch.Tensor([5.0])
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
    )
