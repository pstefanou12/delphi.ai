# Author: pstefanou12@
"""Tests for learning a truncation set from samples inside it."""

from collections.abc import Callable

import pytest
import torch as ch
import torch.distributions

from delphi.torchnomials import hermite, laguerre, polynomial
import delphi.truncation_set


def _normal(shift: float = 0.0, scale: float = 1.0) -> torch.distributions.Distribution:
    loc = ch.full((2,), shift, dtype=ch.float64)
    return torch.distributions.Independent(
        torch.distributions.Normal(loc, ch.full((2,), scale, dtype=ch.float64)), 1
    )


def _exponential(rate: float = 1.0) -> torch.distributions.Distribution:
    rates = ch.full((2,), rate, dtype=ch.float64)
    return torch.distributions.Independent(torch.distributions.Exponential(rates), 1)


def _half_plane(x: ch.Tensor) -> ch.Tensor:
    return x[:, 0] + 0.5 * x[:, 1] >= 0.3


def _disc(x: ch.Tensor) -> ch.Tensor:
    return (x - ch.tensor([0.5, 0.0], dtype=x.dtype)).norm(dim=1) <= 1.0


def _box(x: ch.Tensor) -> ch.Tensor:
    return (x[:, 0] > -0.5) & (x[:, 0] < 1.0) & (x[:, 1] > -1.0) & (x[:, 1] < 0.8)


def _positive_half_plane(x: ch.Tensor) -> ch.Tensor:
    return x[:, 0] + x[:, 1] >= 1.5


def _positive_box(x: ch.Tensor) -> ch.Tensor:
    return (x[:, 0] > 0.3) & (x[:, 0] < 1.5) & (x[:, 1] < 1.2)


def _error(
    truth: torch.distributions.Distribution,
    reference: torch.distributions.Distribution,
    phi: Callable[[ch.Tensor], ch.Tensor],
    basis: polynomial.Polynomial,
    samples: int = 20000,
    **kwargs,
) -> float:
    """Learn the set from samples of each; return its error under the truth."""
    ch.manual_seed(0)
    population = truth.sample((20 * samples,))
    positive = population[phi(population)][:samples]
    learned = delphi.truncation_set.learn_truncation_set(
        positive, reference.sample((samples,)), basis, **kwargs
    )
    fresh = truth.sample((200000,))
    return float((learned(fresh) != phi(fresh)).double().mean())


# Errors measured over seeds 0 to 4. With the reference equal to the truth:
# half plane 0.0003 to 0.0024, disc 0.003 to 0.011, square 0.063 to 0.080
# at degree 8 (the approximation's own error) and 0.017 to 0.031 at 24. With the reference shifted by
# 0.2 and widened by 1.15: half plane 0.019 to 0.047, disc 0.036 to 0.051.
# Every case's trivial set, all of space, is wrong on 0.61 to 0.66.
@pytest.mark.parametrize(
    "reference,phi,degree,tolerance",
    [
        (_normal(), _half_plane, 1, 0.005),
        (_normal(), _disc, 2, 0.025),
        (_normal(), _box, 8, 0.16),
        pytest.param(_normal(), _box, 24, 0.06, marks=pytest.mark.slow),
        (_normal(0.2, 1.15), _half_plane, 1, 0.1),
        (_normal(0.2, 1.15), _disc, 2, 0.1),
    ],
    ids=[
        "half_plane",
        "disc",
        "square",
        "square_degree_24",
        "half_plane_shifted",
        "disc_shifted",
    ],
)
def test_learns_a_gaussian_truncation_set(reference, phi, degree, tolerance):
    basis = hermite.Hermite(degree, 2, prob=True, norm=True)
    assert _error(_normal(), reference, phi, basis) <= tolerance


# Errors measured over seeds 0 to 4. With the reference equal to the truth:
# half plane 0.0009 to 0.0055, box 0.073 to 0.141 at degree 8 (the
# approximation's own error) and 0.024 to 0.027 at 24. With the reference's rate at 0.85: half plane
# 0.004 to 0.014. The trivial set is wrong on 0.44 and 0.64.
@pytest.mark.parametrize(
    "reference,phi,degree,tolerance",
    [
        (_exponential(), _positive_half_plane, 1, 0.012),
        (_exponential(), _positive_box, 8, 0.28),
        pytest.param(_exponential(), _positive_box, 24, 0.06, marks=pytest.mark.slow),
        (_exponential(0.85), _positive_half_plane, 1, 0.03),
    ],
    ids=["half_plane", "box", "box_degree_24", "half_plane_shifted"],
)
def test_learns_an_exponential_truncation_set(reference, phi, degree, tolerance):
    basis = laguerre.Laguerre(degree, 2, norm=True)
    assert _error(_exponential(), reference, phi, basis) <= tolerance


# Errors over seeds 0 to 2, from 5000 samples at degree 4: 0.0009 to 0.0014
# with the reference equal to the truth, 0.0022 to 0.0029 shifted.
@pytest.mark.parametrize(
    "reference,tolerance",
    [(_normal(), 0.003), (_normal(0.2, 1.15), 0.006)],
    ids=["disc", "disc_shifted"],
)
def test_constrained_learner_learns_a_gaussian_truncation_set(reference, tolerance):
    basis = hermite.Hermite(4, 2, prob=True, norm=True)
    error = _error(_normal(), reference, _disc, basis, 5000, learner="constrained")
    assert error <= tolerance


def test_an_unknown_learner_raises():
    with pytest.raises(ValueError, match="learner"):
        delphi.truncation_set.learn_truncation_set(
            ch.ones(3, 2), ch.zeros(3, 2), hermite.Hermite(1, 2), learner="svm"
        )


def test_mixture_weights_split_rho_between_the_samples():
    positive, unlabeled = ch.ones(30, 2), ch.zeros(70, 2)
    X, y, weights = delphi.truncation_set.positive_unlabeled_sample(
        positive, unlabeled, 0.25
    )  # pylint: disable=invalid-name
    assert X.shape == (100, 2)
    ch.testing.assert_close(y, ch.cat([ch.ones(30), ch.zeros(70)]))
    assert abs(float(weights[:30].sum()) - 0.75) < 1e-6
    assert abs(float(weights[30:].sum()) - 0.25) < 1e-6
    assert bool((weights[:30] == weights[0]).all())
    assert bool((weights[30:] == weights[30]).all())


@pytest.mark.parametrize("rho", [0.0, 1.0, -0.1])
def test_rho_outside_the_unit_interval_raises(rho):
    with pytest.raises(ValueError, match="rho"):
        delphi.truncation_set.positive_unlabeled_sample(
            ch.ones(3, 2), ch.zeros(3, 2), rho
        )


@pytest.mark.parametrize(
    "positive,unlabeled,match",
    [
        (ch.ones(3, 2), ch.zeros(3, 3), "dimensions"),
        (ch.ones(0, 2), ch.zeros(3, 2), "non-empty"),
        (ch.ones(3), ch.zeros(3, 2), "shaped"),
    ],
    ids=["dimensions", "empty", "shape"],
)
def test_mismatched_samples_raise(positive, unlabeled, match):
    with pytest.raises(ValueError, match=match):
        delphi.truncation_set.positive_unlabeled_sample(positive, unlabeled)
