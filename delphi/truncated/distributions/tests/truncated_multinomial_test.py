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
from typing import Any, Optional

import pytest
import torch as ch
import torch.distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_multinomial
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)

_TRIALS = 10


def multinomial_kl_divergence(
    fitted: torch.distributions.Distribution, target: torch.distributions.Distribution
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
    return count * torch.distributions.kl.kl_divergence(
        torch.distributions.Categorical(fitted.probs),  # type: ignore[attr-defined]
        torch.distributions.Categorical(target.probs),  # type: ignore[attr-defined]
    )


def _recovers(
    probs: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: bool = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_multinomial.TruncatedMultinomial,
        target=torch.distributions.Multinomial(_TRIALS, probs),
        phi=phi,
        parameter="probs_",
        empirical=lambda s: (s.mean(0) / _TRIALS).clamp(min=1e-6),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        model_args=(_TRIALS,),
        rebuild=functools.partial(torch.distributions.Multinomial, _TRIALS),
        divergence=multinomial_kl_divergence,
        known=known,
        unlearnable=True,
        **unknown,
    )


@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN_CI
def test_three_categories(known):
    _recovers(probs=ch.tensor([0.2, 0.3, 0.5]), phi=lambda z: z[:, 0] >= 4, known=known)


@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN_CI
def test_three_categories_random_half_space(known):
    probs = ch.tensor([0.2, 0.3, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 3),
        torch.distributions.Multinomial(_TRIALS, probs),
        target_alpha=(0.10, 0.40),
        bounds=(-20.0, 20.0),
    )
    _recovers(probs=probs, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN_CI
def test_three_categories_identity(known):
    _recovers(
        probs=ch.tensor([0.2, 0.3, 0.5]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN_CI
def test_ten_categories_identity(known):
    probs = ch.linspace(1.0, 2.0, 10)
    _recovers(
        probs=probs / probs.sum(),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@pytest.mark.slow
@helpers.KNOWN_AND_UNKNOWN_CI
def test_three_categories_small_mass(known):
    probs = ch.tensor([0.2, 0.3, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.Multinomial(_TRIALS, probs),
        target_alpha=(0.005, 0.03),
        bounds=(-1.0, 11.0),
    )
    _recovers(probs=probs, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN_CI
def test_ten_categories_random_half_space(known):
    probs = ch.linspace(1.0, 2.0, 10)
    probs = probs / probs.sum()
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 10),
        torch.distributions.Multinomial(_TRIALS, probs),
        target_alpha=(0.10, 0.40),
        bounds=(-20.0, 20.0),
    )
    _recovers(probs=probs, phi=calibrated.phi, known=known)


@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN_CI
def test_indistinguishable_pair(known):
    # Given x0, the other counts split in proportion to the other probs, so a
    # pair sharing that ratio is told apart by x0 alone. One value of x0
    # would carry no information on probs_0 at all, so the window holds two.
    helpers.assert_family_distinguishes(
        model_class=truncated_multinomial.TruncatedMultinomial,
        targets=(
            torch.distributions.Multinomial(_TRIALS, ch.tensor([0.2, 0.3, 0.5])),
            torch.distributions.Multinomial(_TRIALS, ch.tensor([0.4, 0.225, 0.375])),
        ),
        phi=lambda z: (z[:, 0] > 2.5) & (z[:, 0] < 4.5),
        parameter="probs_",
        empirical=lambda s: (s.mean(0) / _TRIALS).clamp(min=1e-6),
        model_args=(_TRIALS,),
        rebuild=functools.partial(torch.distributions.Multinomial, _TRIALS),
        divergence=multinomial_kl_divergence,
        known=known,
        unlearnable=True,
    )


def test_unknown_truncation_raises():
    # Every sample sums to the trials, so no polynomial basis is independent
    # on them.
    ch.manual_seed(0)
    target = torch.distributions.Multinomial(_TRIALS, ch.tensor([0.2, 0.3, 0.5]))
    sample = target.sample([500])
    model = truncated_multinomial.TruncatedMultinomial(None, _TRIALS)
    with pytest.raises(ValueError, match="truncation_set_degree"):
        model.fit(sample[sample[:, 0] >= 4])


# Seeds 0-2: KL at most 0.0027 (ema), 0.0028 (avg); bounds are twice that.
@pytest.mark.slow
def test_twenty_categories_random_half_space():
    probs = ch.linspace(1.0, 2.0, 20)
    probs = probs / probs.sum()
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 20),
        torch.distributions.Multinomial(_TRIALS, probs),
        target_alpha=(0.10, 0.40),
        bounds=(-40.0, 40.0),
    )
    _recovers(
        probs=probs,
        phi=calibrated.phi,
        tolerances={"ema": 0.0054, "avg": 0.0057},
    )
