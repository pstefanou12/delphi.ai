# Author: pstefanou12@
"""Tests for the Hermite polynomials and the polynomial family they extend."""

import math

import numpy as np
import pytest
import torch as ch

from delphi.torchnomials import hermite, polynomial

_VARIANTS = [(prob, norm) for prob in (False, True) for norm in (False, True)]


def _numpy_values(prob: bool, norm: bool, x: np.ndarray, degree: int) -> np.ndarray:
    """Return P_0..P_degree at x from numpy's raw Hermite series."""
    evaluate = (
        np.polynomial.hermite_e.hermeval if prob else np.polynomial.hermite.hermval
    )
    values = np.stack(
        [evaluate(x, np.eye(degree + 1)[j]) for j in range(degree + 1)], axis=-1
    )
    if norm:
        j = np.arange(degree + 1)
        squared = np.array([math.factorial(int(n)) for n in j], dtype=float)
        if not prob:
            squared = squared * 2.0**j
        values = values / np.sqrt(squared)
    return values


def _quadrature(prob: bool, nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss nodes and weights for the family's weight, as a law."""
    if prob:
        points, weights = np.polynomial.hermite_e.hermegauss(nodes)
        return points, weights / math.sqrt(2.0 * math.pi)
    points, weights = np.polynomial.hermite.hermgauss(nodes)
    return points, weights / math.sqrt(math.pi)


@pytest.mark.parametrize("degree,dims", [(0, 1), (3, 1), (4, 2), (3, 3)])
def test_multi_indices_enumerate_the_degree_simplex(degree, dims):
    indices = polynomial.multi_indices(degree, dims)
    assert indices.shape == (polynomial.num_terms(degree, dims), dims)
    assert len({tuple(row) for row in indices.tolist()}) == indices.size(0)
    totals = indices.sum(1)
    assert int(totals.max()) == degree
    assert bool((totals[1:] >= totals[:-1]).all()), "not ordered by degree"


@pytest.mark.parametrize("prob,norm", _VARIANTS)
def test_univariate_matches_numpy(prob, norm):
    x = np.linspace(-4.0, 4.0, 41)
    expected = _numpy_values(prob, norm, x, 15)
    got = hermite.Hermite(15, prob=prob, norm=norm).univariate(ch.tensor(x)[:, None])[
        :, 0, :
    ]
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("prob,norm", _VARIANTS)
@pytest.mark.parametrize("degree,dims", [(60, 1), (10, 2)])
def test_gram_matrix_under_the_weight(prob, norm, degree, dims):
    nodes, weights = _quadrature(prob, 90)
    grids = np.meshgrid(*([nodes] * dims), indexing="ij")
    points = ch.tensor(np.stack([g.ravel() for g in grids], axis=1))
    mass = np.ones(1)
    for _ in range(dims):
        mass = np.outer(mass, weights).ravel()
    family = hermite.Hermite(degree, dims, prob=prob, norm=norm)
    values = family.basis(points).numpy()
    gram = values.T @ (mass[:, None] * values)
    # The raw norms reach 1e100, so compare the Gram matrix scaled by them:
    # orthogonal polynomials with those norms make it the identity.
    scale = np.sqrt(family.squared_norms().numpy())
    np.testing.assert_allclose(
        gram / np.outer(scale, scale), np.eye(family.num_terms), atol=1e-9
    )


@pytest.mark.parametrize("prob", [False, True])
def test_normalized_basis_stays_orthonormal_at_high_degree(prob):
    nodes, weights = _quadrature(prob, 300)
    values = hermite.Hermite(200, prob=prob, norm=True).basis(ch.tensor(nodes)[:, None])
    gram = values.numpy().T @ (weights[:, None] * values.numpy())
    np.testing.assert_allclose(gram, np.eye(201), atol=1e-10)


@pytest.mark.parametrize("prob", [False, True])
def test_fit_recovers_a_shifted_gaussian_ratio(prob):
    # For x ~ N(mu, s) with s the weight's variance, the normalized
    # coefficients are mu^n / sqrt(n!) (probabilist, s = 1) and
    # (2 mu)^n / sqrt(2^n n!) (physicist, s = 1/2).
    ch.manual_seed(0)
    loc, variance = 0.3, 1.0 if prob else 0.5
    samples = loc + math.sqrt(variance) * ch.randn(400000, 1, dtype=ch.float64)
    family = hermite.Hermite(5, prob=prob, norm=True).fit(samples)
    n = np.arange(6)
    base = loc if prob else 2.0 * loc
    expected = base**n / np.sqrt(
        np.array([math.factorial(int(j)) for j in n]) * (1.0 if prob else 2.0**n)
    )
    np.testing.assert_allclose(family.coefficients.numpy(), expected, atol=5e-3)


@pytest.mark.parametrize("prob", [False, True])
def test_normalizing_leaves_the_fitted_function(prob):
    ch.manual_seed(1)
    samples = ch.randn(2000, 2, dtype=ch.float64) + 0.5
    points = ch.randn(50, 2, dtype=ch.float64)
    raw = hermite.Hermite(6, 2, prob=prob, norm=False).fit(samples)
    normalized = hermite.Hermite(6, 2, prob=prob, norm=True).fit(samples)
    ch.testing.assert_close(raw(points), normalized(points))


def test_weight_is_the_orthogonality_measure():
    physicist = hermite.Hermite(3, 2).weight()
    probabilist = hermite.Hermite(3, 2, prob=True).weight()
    ch.testing.assert_close(physicist.covariance_matrix, 0.5 * ch.eye(2).double())
    ch.testing.assert_close(probabilist.covariance_matrix, ch.eye(2).double())


def test_chunked_evaluation_matches_one_pass():
    ch.manual_seed(2)
    family = hermite.Hermite(8, 2, prob=True, norm=True)
    family.fit(ch.randn(500, 2, dtype=ch.float64))
    points = ch.randn(family.chunk_size() * 3 + 7, 2, dtype=ch.float64)
    ch.testing.assert_close(family(points), family.basis(points) @ family.coefficients)


def test_evaluating_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        hermite.Hermite(3)(ch.zeros(4, 1))


def test_rejects_a_batch_of_the_wrong_width():
    with pytest.raises(ValueError, match="shape"):
        hermite.Hermite(3, 2).basis(ch.zeros(4, 3))


def test_largest_degree_fills_the_term_budget():
    for dims in (1, 2, 3, 10):
        degree = polynomial.largest_degree(dims, 8000, 200)
        assert polynomial.num_terms(degree, dims) <= 8000
        assert degree == 200 or polynomial.num_terms(degree + 1, dims) > 8000


def test_largest_degree_respects_its_cap_and_floor():
    assert polynomial.largest_degree(1, 8000, 20) == 20
    assert polynomial.largest_degree(50, 10, 200) == 1
