# Author: pstefanou12@
"""Tests for the truncated negative binomial (known failures) distribution."""

import functools
from typing import Optional

import torch as ch
from torch import distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_negative_binomial
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def negative_binomial_kl_divergence(
    fitted: distributions.Distribution, target: distributions.Distribution
) -> ch.Tensor:
    """Return KL between two negative binomials sharing total_count.

    torch registers none. As an exponential family with natural parameter
    ``log p`` and log-partition ``-r log(1 - p)``, the divergence is
    ``(log p1 - log p2) r p1 / (1 - p1) + r log((1 - p1) / (1 - p2))``.
    Checked against 4e5 Monte Carlo draws to 1e-3.

    Args:
        fitted: First distribution.
        target: Second distribution, with the same total_count.

    Returns:
        Per-coordinate KL divergence.
    """
    count = fitted.total_count  # type: ignore[attr-defined]
    first, second = fitted.probs, target.probs  # type: ignore[attr-defined]
    return (ch.log(first) - ch.log(second)) * count * first / (1 - first) + count * (
        ch.log1p(-first) - ch.log1p(-second)
    )


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dims: int,
    total_count: ch.Tensor,
    probs: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_negative_binomial.TruncatedNegativeBinomial,
        target=distributions.NegativeBinomial(total_count, probs),
        phi=phi,
        dims=dims,
        parameter="probs_",
        empirical=lambda s: s.mean(0) / (total_count + s.mean(0)),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        model_args=(total_count,),
        rebuild=functools.partial(distributions.NegativeBinomial, total_count),
        divergence=negative_binomial_kl_divergence,
    )


def test_one_dimension():
    _recovers(
        dims=1,
        total_count=ch.Tensor([4.0]),
        probs=ch.Tensor([0.5]),
        phi=lambda z: z[:, 0] > 5,
    )


def test_two_dimensions():
    total_count, probs = ch.Tensor([4.0, 6.0]), ch.Tensor([0.5, 0.3])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.NegativeBinomial(total_count, probs),
        target_alpha=(0.10, 0.40),
        bounds=(-1.0, 200.0),
    )
    _recovers(dims=2, total_count=total_count, probs=probs, phi=calibrated.phi)


def test_two_dimensions_random_half_space():
    total_count, probs = ch.Tensor([4.0, 6.0]), ch.Tensor([0.5, 0.3])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        distributions.NegativeBinomial(total_count, probs),
        target_alpha=(0.10, 0.40),
        bounds=(-50.0, 50.0),
    )
    _recovers(dims=2, total_count=total_count, probs=probs, phi=calibrated.phi)


def test_one_dimension_identity():
    _recovers(
        dims=1,
        total_count=ch.Tensor([4.0]),
        probs=ch.Tensor([0.5]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    total_count, probs = 4.0 * ch.ones(10), ch.linspace(0.3, 0.7, 10)
    _recovers(
        dims=10,
        total_count=total_count,
        probs=probs,
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_one_dimension_small_mass():
    total_count, probs = ch.Tensor([4.0]), ch.Tensor([0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.NegativeBinomial(total_count, probs),
        target_alpha=(0.005, 0.03),
        bounds=(-1.0, 200.0),
    )
    _recovers(dims=1, total_count=total_count, probs=probs, phi=calibrated.phi)


def test_ten_dimensions_sum_threshold():
    total_count, probs = 4.0 * ch.ones(10), ch.linspace(0.3, 0.7, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        distributions.NegativeBinomial(total_count, probs),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 600.0),
    )
    _recovers(dims=10, total_count=total_count, probs=probs, phi=calibrated.phi)


def test_indistinguishable_pair():
    total_count = ch.Tensor([4.0])
    helpers.assert_family_distinguishes(
        model_class=truncated_negative_binomial.TruncatedNegativeBinomial,
        targets=(
            distributions.NegativeBinomial(total_count, ch.Tensor([0.4])),
            distributions.NegativeBinomial(total_count, ch.Tensor([0.7])),
        ),
        phi=lambda z: (z[:, 0] > 5.5) & (z[:, 0] < 7.5),
        dims=1,
        parameter="probs_",
        empirical=lambda s: s.mean(0) / (total_count + s.mean(0)),
        model_args=(total_count,),
        rebuild=functools.partial(distributions.NegativeBinomial, total_count),
        divergence=negative_binomial_kl_divergence,
    )
