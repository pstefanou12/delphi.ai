# Author: pstefanou12@
"""Tests for the generalized Laguerre polynomials."""

import math

import numpy as np
import pytest
import torch as ch

from delphi.torchnomials import laguerre

_ALPHAS = (0.0, 0.5, 2.0)
_VARIANTS = [(alpha, norm) for alpha in _ALPHAS for norm in (False, True)]


def _squared_norms(alpha: float, degree: int) -> np.ndarray:
    """Return ``Gamma(n + a + 1) / (n! Gamma(a + 1))`` for n up to degree."""
    n = np.arange(degree + 1)
    return np.exp(
        np.array([math.lgamma(j + alpha + 1) - math.lgamma(j + 1) for j in n])
        - math.lgamma(alpha + 1)
    )


def _explicit_values(alpha: float, x: np.ndarray, degree: int) -> np.ndarray:
    """Return ``L_0^(a)..L_degree^(a)`` at x from their explicit sums."""
    values = []
    for n in range(degree + 1):
        terms = [
            (-1) ** i
            * math.exp(
                math.lgamma(n + alpha + 1)
                - math.lgamma(n - i + 1)
                - math.lgamma(alpha + i + 1)
            )
            * x**i
            / math.factorial(i)
            for i in range(n + 1)
        ]
        values.append(np.sum(terms, axis=0))
    return np.stack(values, axis=-1)


def _quadrature(alpha: float, nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss nodes and weights for ``Gamma(alpha + 1, 1)``, as a law.

    The nodes are the eigenvalues of the Jacobi matrix of the monic
    recurrence. The weights come from the Christoffel formula, one over the
    sum of the squared orthonormal polynomials at each node, rather than from
    the eigenvectors, whose tiny far-tail entries carry only absolute
    precision.
    """
    n = np.arange(nodes)
    off = np.sqrt(n[1:] * (n[1:] + alpha))
    jacobi = np.diag(2 * n + alpha + 1) + np.diag(off, 1) + np.diag(off, -1)
    points = np.linalg.eigvalsh(jacobi)
    previous, current = np.zeros_like(points), np.ones_like(points)
    total = current**2
    for j in range(nodes - 1):
        # The orthonormal recurrence, written apart from the library's.
        following = (
            (2 * j + 1 + alpha - points) * current
            - math.sqrt(j * (j + alpha)) * previous
        ) / math.sqrt((j + 1) * (j + 1 + alpha))
        previous, current = current, following
        total = total + current**2
    return points, 1.0 / total


@pytest.mark.parametrize("norm", [False, True])
def test_univariate_matches_numpy_at_alpha_zero(norm):
    x = np.linspace(0.0, 30.0, 31)
    expected = np.stack(
        [np.polynomial.laguerre.lagval(x, np.eye(16)[j]) for j in range(16)], axis=-1
    )
    got = laguerre.Laguerre(15, norm=norm).univariate(ch.tensor(x)[:, None])[:, 0, :]
    # Plain Laguerre polynomials are orthonormal, so norm changes nothing.
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("alpha,norm", _VARIANTS)
def test_univariate_matches_the_explicit_sum(alpha, norm):
    x = np.linspace(0.0, 10.0, 21)
    expected = _explicit_values(alpha, x, 12)
    if norm:
        expected = expected / np.sqrt(_squared_norms(alpha, 12))
    family = laguerre.Laguerre(12, alpha=alpha, norm=norm)
    got = family.univariate(ch.tensor(x)[:, None])[:, 0, :]
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("alpha,norm", _VARIANTS)
@pytest.mark.parametrize("degree,dims", [(60, 1), (8, 2)])
def test_gram_matrix_under_the_weight(alpha, norm, degree, dims):
    nodes, weights = _quadrature(alpha, 90)
    grids = np.meshgrid(*([nodes] * dims), indexing="ij")
    points = ch.tensor(np.stack([g.ravel() for g in grids], axis=1))
    mass = np.ones(1)
    for _ in range(dims):
        mass = np.outer(mass, weights).ravel()
    family = laguerre.Laguerre(degree, dims, alpha=alpha, norm=norm)
    values = family.basis(points).numpy()
    gram = values.T @ (mass[:, None] * values)
    # Scaled by the norms, orthogonal polynomials make the Gram the identity.
    scale = np.sqrt(family.squared_norms().numpy())
    np.testing.assert_allclose(
        gram / np.outer(scale, scale), np.eye(family.num_terms), atol=1e-9
    )


@pytest.mark.parametrize("alpha", _ALPHAS)
def test_normalized_basis_stays_orthonormal_at_high_degree(alpha):
    nodes, weights = _quadrature(alpha, 150)
    family = laguerre.Laguerre(100, alpha=alpha, norm=True)
    values = family.basis(ch.tensor(nodes)[:, None]).numpy()
    gram = values.T @ (weights[:, None] * values)
    np.testing.assert_allclose(gram, np.eye(101), atol=1e-9)


@pytest.mark.parametrize("alpha,norm", _VARIANTS)
def test_fit_recovers_a_gamma_ratio(alpha, norm):
    # For x ~ Gamma(a + 1, rate), the Laplace transform of x^a L_n^(a) gives
    # E[L_n^(a)] = ||L_n^(a)||^2 r^n with r = 1 - 1 / rate, so the raw
    # coefficients are r^n and the normalized ones r^n ||L_n^(a)||.
    ch.manual_seed(0)
    rate = 1.5
    gamma = ch.distributions.Gamma(
        ch.tensor([alpha + 1.0], dtype=ch.float64), ch.tensor([rate], dtype=ch.float64)
    )
    samples = gamma.sample((400000,))
    family = laguerre.Laguerre(5, alpha=alpha, norm=norm).fit(samples)
    expected = (1.0 - 1.0 / rate) ** np.arange(6)
    if norm:
        expected = expected * np.sqrt(_squared_norms(alpha, 5))
    np.testing.assert_allclose(family.coefficients.numpy(), expected, atol=1e-2)


@pytest.mark.parametrize("alpha", _ALPHAS)
def test_normalizing_leaves_the_fitted_function(alpha):
    ch.manual_seed(1)
    samples = ch.rand(2000, 2, dtype=ch.float64) * 3.0
    points = ch.rand(50, 2, dtype=ch.float64) * 3.0
    raw = laguerre.Laguerre(6, 2, alpha=alpha, norm=False).fit(samples)
    normalized = laguerre.Laguerre(6, 2, alpha=alpha, norm=True).fit(samples)
    ch.testing.assert_close(raw(points), normalized(points))


@pytest.mark.parametrize("alpha", _ALPHAS)
def test_weight_is_the_orthogonality_measure(alpha):
    weight = laguerre.Laguerre(3, 2, alpha=alpha).weight()
    expected = ch.full((2,), alpha + 1.0, dtype=ch.float64)
    ch.testing.assert_close(weight.mean, expected)
    ch.testing.assert_close(weight.variance, expected)


@pytest.mark.parametrize("alpha", [-1.0, -2.5])
def test_alpha_at_or_below_minus_one_raises(alpha):
    with pytest.raises(ValueError, match="alpha"):
        laguerre.Laguerre(3, alpha=alpha)
