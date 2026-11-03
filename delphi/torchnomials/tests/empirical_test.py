# Author: pstefanou12@
"""Tests for the empirically orthonormal polynomial basis."""

import pytest
import torch as ch
import torch.distributions

from delphi.torchnomials import empirical, hermite, laguerre, polynomial


def _standard_normal(dims: int) -> torch.distributions.Distribution:
    return torch.distributions.MultivariateNormal(
        ch.zeros(dims, dtype=ch.float64), ch.eye(dims, dtype=ch.float64)
    )


def _gram(basis: ch.Tensor) -> ch.Tensor:
    return basis.T @ basis / basis.size(0)


def _fresh_draw(
    reference: torch.distributions.Distribution, size: int, seed: int
) -> ch.Tensor:
    with ch.random.fork_rng():
        ch.manual_seed(seed)
        return reference.sample((size,)).to(ch.float64)


def test_orthonormal_on_the_reference_draw():
    reference = _standard_normal(2)
    family = empirical.EmpiricalOrthonormal(
        hermite.Hermite(4, 2), reference, num_samples=5000, seed=3
    )
    draws = _fresh_draw(reference, 5000, seed=3)
    ch.testing.assert_close(
        _gram(family.basis(draws)), ch.eye(family.num_terms, dtype=ch.float64)
    )


def _fresh_gram_error(family: empirical.EmpiricalOrthonormal, seed: int) -> float:
    """Return the worst entry of the Gram error on a fresh 200000-point draw."""
    gram = _gram(family.basis(_fresh_draw(family.reference, 200000, seed)))
    return float((gram - ch.eye(family.num_terms, dtype=ch.float64)).abs().max())


# Both Gram estimates carry sampling noise; at degree 3 and 200000 draws the
# worst entry measured 0.026 to 0.046 over five seeds.
_FRESH_TOLERANCE = 0.1


def test_orthonormal_under_the_reference_on_a_fresh_draw():
    family = empirical.EmpiricalOrthonormal(
        hermite.Hermite(3, 2), _standard_normal(2), num_samples=200000
    )
    assert _fresh_gram_error(family, seed=7) < _FRESH_TOLERANCE


def test_reproduces_the_normalized_hermite_family():
    family = empirical.EmpiricalOrthonormal(
        hermite.Hermite(3), _standard_normal(1), num_samples=4000000
    )
    points = ch.linspace(-2.0, 2.0, 9, dtype=ch.float64)[:, None]
    expected = hermite.Hermite(3, prob=True, norm=True).basis(points)
    # The worst entry measured 0.003 to 0.016 over six seeds.
    ch.testing.assert_close(family.basis(points), expected, atol=0.04, rtol=0.0)


def test_reproduces_the_normalized_laguerre_family():
    reference = torch.distributions.Independent(
        torch.distributions.Exponential(ch.ones(1, dtype=ch.float64)), 1
    )
    family = empirical.EmpiricalOrthonormal(
        laguerre.Laguerre(3), reference, num_samples=4000000
    )
    points = ch.linspace(0.0, 4.0, 9, dtype=ch.float64)[:, None]
    # The Cholesky factor's positive diagonal keeps each L_n's leading sign.
    expected = laguerre.Laguerre(3, norm=True).basis(points)
    # Exp(1)'s heavier tail makes the degree-3 moments noisier: the worst
    # entry measured 0.005 to 0.071 over six seeds.
    ch.testing.assert_close(family.basis(points), expected, atol=0.15, rtol=0.0)


def test_orthonormal_under_a_non_product_reference():
    reference = torch.distributions.MultivariateNormal(
        ch.tensor([0.5, -0.5], dtype=ch.float64),
        ch.tensor([[1.0, 0.6], [0.6, 1.5]], dtype=ch.float64),
    )
    family = empirical.EmpiricalOrthonormal(
        hermite.Hermite(3, 2), reference, num_samples=200000
    )
    assert _fresh_gram_error(family, seed=11) < _FRESH_TOLERANCE


def test_a_lower_degree_basis_is_a_prefix():
    reference = _standard_normal(2)
    high = empirical.EmpiricalOrthonormal(hermite.Hermite(5, 2), reference, 4000)
    low = empirical.EmpiricalOrthonormal(hermite.Hermite(3, 2), reference, 4000)
    points = _fresh_draw(reference, 50, seed=5)
    prefix = high.basis(points)[:, : polynomial.num_terms(3, 2)]
    ch.testing.assert_close(low.basis(points), prefix)


def test_fit_and_weight():
    reference = _standard_normal(1)
    family = empirical.EmpiricalOrthonormal(hermite.Hermite(4), reference)
    assert family.weight() is reference
    samples = _fresh_draw(reference, 1000, seed=2)
    family.fit(samples)
    ch.testing.assert_close(
        family(samples), family.basis(samples) @ family.coefficients
    )
    # Under its own reference the density ratio is one: c_0 = 1, the rest 0.
    assert abs(float(family.coefficients[0]) - 1.0) < 1e-6


def test_too_few_draws_raises():
    with pytest.raises(ValueError, match="draws"):
        empirical.EmpiricalOrthonormal(hermite.Hermite(4, 2), _standard_normal(2), 10)


def test_singular_gram_raises():
    # A binomial with 3 trials has 4 support points, too few to separate the
    # 6 polynomials of degree 5.
    reference = torch.distributions.Independent(
        torch.distributions.Binomial(3, probs=ch.full((1,), 0.5, dtype=ch.float64)), 1
    )
    with pytest.raises(ValueError, match="singular"):
        empirical.EmpiricalOrthonormal(hermite.Hermite(5), reference)


def test_a_reference_on_a_lower_dimensional_set_raises():
    # Dirichlet points satisfy x_1 + x_2 + x_3 = 1, so the polynomials are
    # linearly dependent on its support.
    reference = torch.distributions.Dirichlet(
        ch.tensor([2.0, 3.0, 4.0], dtype=ch.float64)
    )
    with pytest.raises(ValueError, match="singular"):
        empirical.EmpiricalOrthonormal(laguerre.Laguerre(2, 3), reference)
