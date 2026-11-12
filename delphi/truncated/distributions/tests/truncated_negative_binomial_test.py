# Author: pstefanou12@
"""Tests for the truncated negative binomial (known failures) distribution."""

import functools
from typing import Any, Optional

import pytest
import torch as ch
import torch.distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_negative_binomial
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def negative_binomial_kl_divergence(
    fitted: torch.distributions.Distribution, target: torch.distributions.Distribution
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
    total_count: ch.Tensor,
    probs: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: bool = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_negative_binomial.TruncatedNegativeBinomial,
        target=torch.distributions.NegativeBinomial(total_count, probs),
        phi=phi,
        parameter="probs_",
        empirical=lambda s: s.mean(0) / (total_count + s.mean(0)),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        model_args=(total_count,),
        rebuild=functools.partial(torch.distributions.NegativeBinomial, total_count),
        divergence=negative_binomial_kl_divergence,
        known=known,
        **unknown,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension(known):
    _recovers(
        total_count=ch.tensor([4.0]),
        probs=ch.tensor([0.5]),
        phi=lambda z: z[:, 0] > 5,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions(known):
    total_count, probs = ch.tensor([4.0, 6.0]), ch.tensor([0.5, 0.3])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.NegativeBinomial(total_count, probs),
        target_alpha=(0.10, 0.40),
        bounds=(-1.0, 200.0),
    )
    _recovers(total_count=total_count, probs=probs, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_half_space(known):
    total_count, probs = ch.tensor([4.0, 6.0]), ch.tensor([0.5, 0.3])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        torch.distributions.NegativeBinomial(total_count, probs),
        target_alpha=(0.10, 0.40),
        bounds=(-50.0, 50.0),
    )
    _recovers(total_count=total_count, probs=probs, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN_CI
def test_one_dimension_identity(known):
    _recovers(
        total_count=ch.tensor([4.0]),
        probs=ch.tensor([0.5]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    total_count, probs = 4.0 * ch.ones(10), ch.linspace(0.3, 0.7, 10)
    _recovers(
        total_count=total_count,
        probs=probs,
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_small_mass(known):
    total_count, probs = ch.tensor([4.0]), ch.tensor([0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.NegativeBinomial(total_count, probs),
        target_alpha=(0.005, 0.03),
        bounds=(-1.0, 200.0),
    )
    _recovers(total_count=total_count, probs=probs, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_sum_threshold(known):
    total_count, probs = 4.0 * ch.ones(10), ch.linspace(0.3, 0.7, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.NegativeBinomial(total_count, probs),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 600.0),
    )
    _recovers(total_count=total_count, probs=probs, phi=calibrated.phi, known=known)


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the learned window is too coarse to separate the pair",
    )
)
def test_indistinguishable_pair(known):
    total_count = ch.tensor([4.0])
    helpers.assert_family_distinguishes(
        model_class=truncated_negative_binomial.TruncatedNegativeBinomial,
        targets=(
            torch.distributions.NegativeBinomial(total_count, ch.tensor([0.4])),
            torch.distributions.NegativeBinomial(total_count, ch.tensor([0.7])),
        ),
        phi=lambda z: (z[:, 0] > 5.5) & (z[:, 0] < 7.5),
        parameter="probs_",
        empirical=lambda s: s.mean(0) / (total_count + s.mean(0)),
        model_args=(total_count,),
        rebuild=functools.partial(torch.distributions.NegativeBinomial, total_count),
        divergence=negative_binomial_kl_divergence,
        known=known,
    )


# Seeds 0-2: KL at most 0.0019 (ema), 0.0018 (avg); bounds are twice that.
@pytest.mark.slow
def test_twenty_dimensions_sum_threshold():
    total_count, probs = 4.0 * ch.ones(20), ch.linspace(0.3, 0.7, 20)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.NegativeBinomial(total_count, probs),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 1200.0),
    )
    _recovers(
        dims=20,
        total_count=total_count,
        probs=probs,
        phi=calibrated.phi,
        tolerances={"ema": 0.0039, "avg": 0.0036},
        max_coordinate=0.0017,
    )
