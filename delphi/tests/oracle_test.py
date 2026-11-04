# Author: pstefanou12@
"""Tests for the stateful truncation sets."""

import pytest
import torch as ch
import torch.distributions

from delphi.learning import l1_regression
import delphi.oracle
from delphi.torchnomials import hermite
from delphi.transformations import standardization


def _half_plane_sample() -> ch.Tensor:
    ch.manual_seed(0)
    population = ch.randn(40000, 2, dtype=ch.float64)
    return population[population[:, 0] > 0.0][:10000]


def _recovered() -> delphi.oracle.RecoveredSet:
    sample = _half_plane_sample()
    transform = standardization.Standardization(ch.eye(2, dtype=ch.float64))
    transform.fit(sample)
    return delphi.oracle.RecoveredSet(20, transform).fit(sample)


def _truth_in_its_coordinates(recovered: delphi.oracle.RecoveredSet):
    # The standard normal, seen through the fitted standardization.
    transform = recovered.transformation
    assert isinstance(transform, standardization.Standardization)
    assert transform.mean is not None
    return torch.distributions.MultivariateNormal(
        -transform.mean, ch.eye(2, dtype=ch.float64)
    )


def test_identity_admits_every_sample():
    assert bool(delphi.oracle.Identity()(ch.randn(5, 3)).all())


def test_membership_needs_a_fitted_distribution():
    with pytest.raises(RuntimeError, match="given"):
        _recovered()(ch.zeros(3, 2))


def test_given_needs_a_fit():
    with pytest.raises(RuntimeError, match="fit"):
        delphi.oracle.RecoveredSet(4).given(
            torch.distributions.MultivariateNormal(ch.zeros(2), ch.eye(2))
        )


def test_threshold_is_half_the_median_ratio_on_the_samples():
    recovered = _recovered()
    bound = recovered.given(_truth_in_its_coordinates(recovered))
    z = recovered.transform(_half_plane_sample())
    assert bound.threshold == pytest.approx(0.5 * float(bound.ratio(z).median()))
    assert recovered.distribution is None, "given must not change the original"


def test_recovers_a_half_plane_from_its_samples():
    recovered = _recovered()
    bound = recovered.given(_truth_in_its_coordinates(recovered))
    ch.manual_seed(1)
    population = ch.randn(20000, 2, dtype=ch.float64)
    error = float((bound(population) != (population[:, 0] > 0.0)).float().mean())
    assert error <= 0.05


def _threshold_error(phi, degree: int, seed: int = 0) -> float:
    """Fit a threshold to labeled Gaussian points; return its fresh error."""
    generator = ch.Generator().manual_seed(seed)
    points = ch.randn(20000, 2, dtype=ch.float64, generator=generator)
    basis = hermite.Hermite(degree, 2, prob=True, norm=True)
    learned = delphi.oracle.PolynomialThreshold(basis).fit(points, phi(points).double())
    fresh = ch.randn(50000, 2, dtype=ch.float64, generator=generator)
    return float((learned(fresh) != phi(fresh)).double().mean())


# Errors measured over seeds 0 to 4: half plane at degree 1, 0.0005 to
# 0.0043; disc at degree 2, 0.0044 to 0.0096; square at degree 6, 0.054 to
# 0.057, the degree's own approximation error.
@pytest.mark.parametrize(
    "phi,degree,tolerance",
    [
        (lambda x: x[:, 0] + x[:, 1] >= 0.5, 1, 0.01),
        (lambda x: x.norm(dim=1) <= 1.2, 2, 0.02),
        (lambda x: (x.abs() <= 1.0).all(1), 6, 0.1),
    ],
    ids=["half_plane", "disc", "square"],
)
def test_threshold_learns_a_set_from_labels(phi, degree, tolerance):
    assert _threshold_error(phi, degree) <= tolerance


def test_threshold_minimizes_the_weighted_error():
    generator = ch.Generator().manual_seed(1)
    points = ch.randn(400, 2, dtype=ch.float64, generator=generator)
    labels = (points[:, 0] > 0.2).double()
    flipped = ch.rand(400, generator=generator, dtype=ch.float64) < 0.2
    labels = ch.where(flipped, 1.0 - labels, labels)
    weights = ch.rand(400, generator=generator, dtype=ch.float64)
    learned = delphi.oracle.PolynomialThreshold(
        hermite.Hermite(2, 2, prob=True, norm=True)
    ).fit(points, labels, weights)
    scores = learned.polynomial(points)

    def error(tau: float) -> float:
        predicted = (scores >= tau).double()
        return float((weights * (predicted != labels).double()).sum())

    candidates = [float(s) for s in scores] + [float(scores.max()) + 1.0]
    best = min(error(tau) for tau in candidates)
    assert abs(error(learned.threshold) - best) < 1e-12


def test_threshold_needs_a_fit():
    learned = delphi.oracle.PolynomialThreshold(hermite.Hermite(2, 2))
    with pytest.raises(RuntimeError, match="fit"):
        learned(ch.zeros(3, 2))


def test_preconditioning_leaves_the_regression_optimum():
    generator = ch.Generator().manual_seed(2)
    points = ch.randn(2000, 2, dtype=ch.float64, generator=generator)
    labels = (points.norm(dim=1) <= 1.2).double()
    weights = ch.rand(2000, dtype=ch.float64, generator=generator)
    basis = hermite.Hermite(3, 2, prob=True, norm=True)
    learned = delphi.oracle.PolynomialThreshold(basis).fit(points, labels, weights)
    direct = l1_regression.L1Regression().fit(basis.basis(points), labels, weights)
    ch.testing.assert_close(learned.polynomial.coefficients, direct.coef_)


def test_a_stalled_regression_warns():
    points = ch.randn(500, 2, dtype=ch.float64, generator=ch.Generator().manual_seed(3))
    learned = delphi.oracle.PolynomialThreshold(
        hermite.Hermite(4, 2, prob=True, norm=True), hyperparameters={"iterations": 1}
    )
    with pytest.warns(UserWarning, match="without converging"):
        learned.fit(points, (points[:, 0] > 0).double())
