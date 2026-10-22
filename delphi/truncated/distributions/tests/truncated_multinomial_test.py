# Author: pstefanou12@
"""Tests for the truncated multinomial distribution with known trials.

``dims`` is the number of categories, and every sample's counts sum to the
trials, so a sum bound cuts nothing. A norm bound barely moves the category
frequencies (the naive estimate scored 0.088, inside the bound), so the
ten-category coupling test cuts a random half-space instead. With one trial the multinomial is a categorical, whose
truncation to a set of outcomes leaves the excluded outcomes' probabilities
unidentified, so the tests use ten.
"""

import functools
from typing import Optional

import pytest
import torch as ch
from torch import distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_multinomial
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)

_TRIALS = 10


def multinomial_kl_divergence(
    fitted: distributions.Distribution, target: distributions.Distribution
) -> ch.Tensor:
    """Return KL between two multinomials with the same trials.

    torch registers none. A multinomial is ``total_count`` independent
    categoricals, so its divergence is that many times theirs. Checked
    against 4e5 Monte Carlo draws to 1e-3.

    Args:
        fitted: First distribution.
        target: Second distribution, with the same total_count.

    Returns:
        The KL divergence.
    """
    count = fitted.total_count  # type: ignore[attr-defined]
    return count * distributions.kl.kl_divergence(
        distributions.Categorical(fitted.probs),  # type: ignore[attr-defined]
        distributions.Categorical(target.probs),  # type: ignore[attr-defined]
    )


def _recovers(
    probs: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_multinomial.TruncatedMultinomial,
        target=distributions.Multinomial(_TRIALS, probs),
        phi=phi,
        dims=probs.size(0),
        parameter="probs_",
        empirical=lambda s: (s.mean(0) / _TRIALS).clamp(min=1e-6),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        model_args=(_TRIALS,),
        rebuild=functools.partial(distributions.Multinomial, _TRIALS),
        divergence=multinomial_kl_divergence,
    )


def test_three_categories():
    _recovers(probs=ch.Tensor([0.2, 0.3, 0.5]), phi=lambda z: z[:, 0] >= 4)


def test_three_categories_random_half_space():
    probs = ch.Tensor([0.2, 0.3, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 3),
        distributions.Multinomial(_TRIALS, probs),
        target_alpha=(0.10, 0.40),
        bounds=(-20.0, 20.0),
    )
    _recovers(probs=probs, phi=calibrated.phi)


def test_three_categories_identity():
    _recovers(
        probs=ch.Tensor([0.2, 0.3, 0.5]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_categories_identity():
    probs = ch.linspace(1.0, 2.0, 10)
    _recovers(
        probs=probs / probs.sum(),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


@pytest.mark.slow
def test_three_categories_small_mass():
    probs = ch.Tensor([0.2, 0.3, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Multinomial(_TRIALS, probs),
        target_alpha=(0.005, 0.03),
        bounds=(-1.0, 11.0),
    )
    _recovers(probs=probs, phi=calibrated.phi)


def test_ten_categories_random_half_space():
    probs = ch.linspace(1.0, 2.0, 10)
    probs = probs / probs.sum()
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 10),
        distributions.Multinomial(_TRIALS, probs),
        target_alpha=(0.10, 0.40),
        bounds=(-20.0, 20.0),
    )
    _recovers(probs=probs, phi=calibrated.phi)


def test_indistinguishable_pair():
    # Given x0, the other counts split in proportion to the other probs, so a
    # pair sharing that ratio is told apart by x0 alone. One value of x0
    # would carry no information on probs_0 at all, so the window holds two.
    helpers.assert_family_distinguishes(
        model_class=truncated_multinomial.TruncatedMultinomial,
        targets=(
            distributions.Multinomial(_TRIALS, ch.Tensor([0.2, 0.3, 0.5])),
            distributions.Multinomial(_TRIALS, ch.Tensor([0.4, 0.225, 0.375])),
        ),
        phi=lambda z: (z[:, 0] > 2.5) & (z[:, 0] < 4.5),
        dims=3,
        parameter="probs_",
        empirical=lambda s: (s.mean(0) / _TRIALS).clamp(min=1e-6),
        model_args=(_TRIALS,),
        rebuild=functools.partial(distributions.Multinomial, _TRIALS),
        divergence=multinomial_kl_divergence,
    )
