# Author: pstefanou12@
"""Tests for learning a set from positive samples."""

from collections.abc import Callable
import math

import pytest
import torch as ch
import torch.distributions

from delphi.learning import positive_only
from delphi.torchnomials import hermite


def _normal(shift=(0.0, 0.0), scale: float = 1.0) -> torch.distributions.Distribution:
    loc = ch.tensor(shift, dtype=ch.float64)
    return torch.distributions.Independent(
        torch.distributions.Normal(loc, ch.full((2,), scale, dtype=ch.float64)), 1
    )


def _uniform(half_width: float) -> positive_only.Sampler:
    def sample(count: int) -> ch.Tensor:
        return half_width * (2.0 * ch.rand(count, 2, dtype=ch.float64) - 1.0)

    return sample


def _sampler(distribution: torch.distributions.Distribution) -> positive_only.Sampler:
    return lambda count: distribution.sample((count,))


_STANDARD = _sampler(_normal())


def _half_plane(x: ch.Tensor) -> ch.Tensor:
    return x[:, 0] + 0.5 * x[:, 1] >= 0.3


def _disc(x: ch.Tensor) -> ch.Tensor:
    return (x - ch.tensor([0.5, 0.0], dtype=x.dtype)).norm(dim=1) <= 1.0


def _in_box(count: int) -> ch.Tensor:
    """Draw N(0, I) restricted to the box [-1, 1]^2."""
    draws = _normal().sample((4 * count,))
    return draws[(draws.abs() <= 1.0).all(1)][:count]


def _learn(
    truth: Callable[[int], ch.Tensor],
    phi: Callable[[ch.Tensor], ch.Tensor],
    reference,
    *,
    degree: int = 4,
    seed: int = 0,
    hyperparameters=None,
) -> tuple[positive_only.PositiveOnlyLearner, float]:
    """Learn the set from 5000 positives; return it and its error under truth."""
    ch.manual_seed(seed)
    population = truth(100000)
    positive = population[phi(population)][:5000]
    learner = positive_only.PositiveOnlyLearner(
        hermite.Hermite(degree, 2, prob=True, norm=True),
        hyperparameters=hyperparameters,
    ).fit(positive, reference)
    fresh = truth(200000)
    return learner, float((learner(fresh) != phi(fresh)).double().mean())


def _design(seed: int) -> tuple[ch.Tensor, ch.Tensor]:
    """Return a degree-4 design at 2000 disc positives and 2000 reference draws."""
    ch.manual_seed(seed)
    basis = hermite.Hermite(4, 2, prob=True, norm=True)
    population = _normal().sample((20000,))
    positive = population[_disc(population)][:2000]
    return basis.basis(positive), basis.basis(_normal().sample((2000,)))


@pytest.mark.parametrize("rho", [1e-2, 1e-4])
def test_constrained_regression_is_feasible(rho):
    positive, reference = _design(seed=0)
    values = positive @ positive_only.constrained_regression(positive, reference, rho)
    assert float((1.0 - values).clamp_min(0.0).mean()) <= rho * (1.0 + 1e-6)
    assert float((values >= 1.0 - math.sqrt(rho)).double().mean()) >= 1.0 - math.sqrt(
        rho
    )


def test_no_feasible_perturbation_lowers_the_objective(monkeypatch):
    monkeypatch.setattr(positive_only, "MULTIPLIER_TOLERANCE", 1e-9)
    rho = 1e-2
    positive, reference = _design(seed=1)
    solution = positive_only.constrained_regression(positive, reference, rho)
    best = float((reference @ solution).abs().mean())
    generator = ch.Generator().manual_seed(2)
    for scale in (1e-3, 1e-2, 1e-1):
        for _ in range(50):
            candidate = solution + scale * ch.randn(
                solution.shape, dtype=ch.float64, generator=generator
            )
            shortfall = float((1.0 - positive @ candidate).clamp_min(0.0).mean())
            if shortfall <= rho:
                assert float((reference @ candidate).abs().mean()) >= best - 1e-9


def test_learned_set_keeps_the_positives():
    learner, _ = _learn(_STANDARD, _disc, _sampler(_normal()))
    rho, rounds = learner.hyperparameters.rho, learner.rounds_
    assert learner.coverage_ >= 1.0 - rounds * math.sqrt(rho)


# Worst errors over seeds 0 to 2: disc 0.0017, half plane 0.0016, and disc
# against a reference shifted by 0.2 and widened by 1.15, 0.0024.
@pytest.mark.parametrize(
    "phi,reference,tolerance",
    [
        (_disc, _normal(), 0.004),
        (_half_plane, _normal(), 0.004),
        (_disc, _normal((0.2, 0.2), 1.15), 0.005),
    ],
    ids=["disc", "half_plane", "disc_misspecified"],
)
def test_learns_a_gaussian_set(phi, reference, tolerance):
    _, error = _learn(_STANDARD, phi, _sampler(reference))
    assert error <= tolerance


def test_rounds_carve_a_reference_with_larger_support():
    # Their Figure 2: the truth lives on [-1, 1]^2, the reference on [-2, 2]^2.
    # Over seeds 0 to 2 the first round errs on 0.106 to 0.120 and the
    # intersection on 0.009 to 0.042. At rho = 1e-4 both sit near 0.045.
    learner, error = _learn(
        _in_box, _half_plane, _uniform(2.0), hyperparameters={"rho": 1e-3}
    )
    fresh = _in_box(200000)
    first_error = float(
        (learner.hypotheses_[0](fresh) != _half_plane(fresh)).double().mean()
    )
    assert learner.rounds_ > 1
    assert error <= 0.08
    assert first_error >= error + 0.05


def test_a_list_with_a_wrong_reference_learns_the_set():
    wrong = _sampler(_normal((3.0, 0.0), 0.5))
    learner, error = _learn(_STANDARD, _disc, [wrong, _sampler(_normal())])
    # Worst over seeds 0 to 2: 0.0018.
    assert error <= 0.004
    assert len(learner.reference_mass_) == 2


def test_reference_mass_estimates_the_set_mass():
    learner, _ = _learn(_STANDARD, _disc, _sampler(_normal()))
    truth = float(_disc(_normal().sample((200000,))).double().mean())
    # Worst over seeds 0 to 2: 0.0034.
    assert abs(learner.reference_mass_ - truth) < 0.007


def test_membership_before_fit_raises():
    learner = positive_only.PositiveOnlyLearner(hermite.Hermite(2, 2))
    with pytest.raises(RuntimeError, match="fit"):
        learner(ch.zeros(3, 2))


@pytest.mark.parametrize(
    "positive,reference,match",
    [
        (ch.zeros(5), _uniform(1.0), "shaped"),
        (ch.zeros(5, 2), [], "empty"),
    ],
    ids=["shape", "empty_list"],
)
def test_bad_inputs_raise(positive, reference, match):
    learner = positive_only.PositiveOnlyLearner(hermite.Hermite(2, 2))
    with pytest.raises(ValueError, match=match):
        learner.fit(positive, reference)
