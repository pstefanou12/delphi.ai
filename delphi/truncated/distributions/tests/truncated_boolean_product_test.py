# Author: pstefanou12@
"""Tests for the truncated boolean product distribution."""

import functools
from typing import Any, Optional

import pytest
import torch as ch
import torch.distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_boolean_product
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(
    dims: int,
    probability: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: bool = True,
    **unknown: Any,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_boolean_product.TruncatedBooleanProduct,
        target=torch.distributions.Bernoulli(probability),
        phi=phi,
        dims=dims,
        parameter="probs_",
        empirical=lambda s: s.mean(0),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        known=known,
        unlearnable=True,
        **unknown,
    )


def _not_both_set(z):
    """Return membership for the set excluding the all-ones corner.

    Args:
        z: Samples to test.

    Returns:
        Boolean mask over the batch.
    """
    return ~((z[:, 0] == 1) * (z[:, 1] == 1))


@helpers.KNOWN_AND_UNKNOWN_CI
def test_two_dimensions(known):
    _recovers(
        dims=2,
        probability=ch.tensor([0.5, 0.75]),
        phi=_not_both_set,
        hyperparameters={"batch_size": 1, "lr": 1e-1, "project": False},
        known=known,
    )


@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN_CI
def test_twenty_dimensions(known):
    _recovers(
        dims=20,
        probability=ch.tensor([0.5, 0.75, 0.3, 0.4, 0.8]).repeat(4),
        phi=_not_both_set,
        hyperparameters={"batch_size": 1, "lr": 1e-1},
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN_CI
def test_five_dimensions_random_polytope(known):
    # Two dimensions would leave four atoms, too few for a set carrying
    # survival mass near a third to keep every coordinate identifiable.
    probability = ch.tensor([0.5, 0.75, 0.3, 0.4, 0.8])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_polytope, 0, 5, 4),
        torch.distributions.Bernoulli(probability),
        bounds=(-25.0, 25.0),
    )
    _recovers(
        dims=5,
        probability=probability,
        phi=calibrated.phi,
        hyperparameters={"batch_size": 1, "lr": 1e-1, "project": False},
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN_CI
def test_one_dimension_identity(known):
    _recovers(
        dims=1,
        probability=ch.tensor([0.4]),
        phi=delphi.oracle.Identity(),
        hyperparameters={"iterations": 1500},
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN_CI
def test_ten_dimensions_identity(known):
    _recovers(
        dims=10,
        probability=ch.linspace(0.3, 0.7, 10),
        phi=delphi.oracle.Identity(),
        hyperparameters={"iterations": 1500},
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN_CI
def test_ten_dimensions_sum_threshold(known):
    probability = ch.linspace(0.3, 0.7, 10)
    # One scalar condition, so sampling stays cheap, but it biases every
    # coordinate instead of one.
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.Bernoulli(probability),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 200.0),
    )
    _recovers(
        dims=10,
        probability=probability,
        phi=calibrated.phi,
        hyperparameters={"lr": 1e-1},
        known=known,
    )


@pytest.mark.long
@helpers.KNOWN_AND_UNKNOWN_CI
def test_indistinguishable_pair(known):
    # Conditioning on an exact count would make the pair identical rather
    # than alike: scaling every odds ratio leaves p(x | sum(x) = k) unchanged.
    # Two counts keep the ratio between them, which is what identifies p.
    helpers.assert_family_distinguishes(
        model_class=truncated_boolean_product.TruncatedBooleanProduct,
        targets=(
            torch.distributions.Bernoulli(ch.full((5,), 0.3)),
            torch.distributions.Bernoulli(ch.full((5,), 0.6)),
        ),
        phi=lambda z: (z.sum(-1) >= 2) & (z.sum(-1) <= 3),
        dims=5,
        parameter="probs_",
        empirical=lambda s: s.mean(0).clamp(1e-4, 1 - 1e-4),
        known=known,
        unlearnable=True,
    )


def test_unknown_truncation_raises():
    # On {0, 1} every power of a coordinate equals the coordinate.
    ch.manual_seed(0)
    sample = torch.distributions.Bernoulli(ch.tensor([0.3, 0.6])).sample([500])
    model = truncated_boolean_product.TruncatedBooleanProduct(None, 2)
    with pytest.raises(ValueError, match="truncation_set_degree"):
        model.fit(sample[_not_both_set(sample)])
