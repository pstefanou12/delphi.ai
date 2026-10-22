# Author: pstefanou12@
"""Abstract polynomial family over total-degree multi-indices.

A family supplies its univariate polynomials ``P_0, ..., P_k``, their squared
norms under its weight measure ``w``, and ``w`` itself. In d dimensions a
multi-index ``V`` names the product ``P_V(x) = prod_i P_{V_i}(x_i)``, and the
basis holds every ``V`` of total degree at most k: ``C(k + d, d)`` terms.

Fitting to samples from a law ``D`` sets ``c_V = E_D[P_V] / ||P_V||^2``,
the coefficients of the density ratio ``dD / dw`` when the family is
orthogonal under ``w``.
"""

import abc
from collections.abc import Iterator
import math
from typing import Optional

import torch as ch
from torch import distributions

# Rows times terms evaluated at once, which bounds the working memory of an
# evaluation to tens of megabytes whatever the sample size.
CHUNK_ELEMENTS = 2**22


def num_terms(degree: int, dims: int) -> int:
    """Return how many multi-indices have total degree at most ``degree``.

    Args:
        degree: Maximum total degree.
        dims: Number of variables.

    Returns:
        ``C(degree + dims, dims)``.
    """
    return math.comb(degree + dims, dims)


def largest_degree(dims: int, max_terms: int, max_degree: int) -> int:
    """Return the largest degree whose basis fits a budget of terms.

    Args:
        dims: Number of variables.
        max_terms: Most terms the basis may have.
        max_degree: Cap on the degree.

    Returns:
        The largest degree, at least 1, with at most ``max_terms`` terms.
    """
    degree = 1
    while degree < max_degree and num_terms(degree + 1, dims) <= max_terms:
        degree += 1
    return degree


def _compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Yield every way of writing total as an ordered sum of parts terms."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total, -1, -1):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def multi_indices(degree: int, dims: int) -> ch.Tensor:
    """Return every multi-index of total degree at most ``degree``.

    Ordered by total degree, so the first ``num_terms(j, dims)`` rows are the
    basis of degree j and a truncation is a prefix.

    Args:
        degree: Maximum total degree.
        dims: Number of variables.

    Returns:
        Long tensor of shape ``(num_terms(degree, dims), dims)``.

    Raises:
        ValueError: If degree is negative or dims is not positive.
    """
    if degree < 0 or dims < 1:
        raise ValueError(
            f"degree must be non-negative and dims positive; got {degree}, {dims}."
        )
    indices = [
        index for total in range(degree + 1) for index in _compositions(total, dims)
    ]
    return ch.tensor(indices, dtype=ch.long)


class Polynomial(abc.ABC):
    """A polynomial over a total-degree basis, fit to data or given coefficients.

    Attributes:
        degree: Maximum total degree.
        dims: Number of variables.
        multi_indices: The basis's multi-indices, shaped ``(num_terms, dims)``.
        coefficients: One coefficient per basis polynomial; None until fit
            or assigned.
    """

    def __init__(self, degree: int, dims: int = 1):
        """Initialize an unfit polynomial.

        Args:
            degree: Maximum total degree.
            dims: Number of variables.

        Raises:
            ValueError: If degree is negative or dims is not positive.
        """
        self.degree = degree
        self.dims = dims
        self.multi_indices = multi_indices(degree, dims)
        self.coefficients: Optional[ch.Tensor] = None

    @abc.abstractmethod
    def univariate(self, x: ch.Tensor) -> ch.Tensor:
        """Return ``P_0, ..., P_degree`` at every coordinate of every row.

        Args:
            x: Batch of shape ``(n, dims)``.

        Returns:
            Tensor of shape ``(n, dims, degree + 1)``.
        """

    @abc.abstractmethod
    def univariate_squared_norms(self) -> ch.Tensor:
        """Return ``||P_j||^2`` under the weight measure, for j up to degree."""

    @abc.abstractmethod
    def weight(self) -> distributions.Distribution:
        """Return the weight measure in ``dims`` dimensions."""

    @property
    def num_terms(self) -> int:
        """Number of polynomials in the basis."""
        return self.multi_indices.size(0)

    def squared_norms(self) -> ch.Tensor:
        """Return ``||P_V||^2`` for every multi-index, shaped ``(num_terms,)``."""
        univariate = self.univariate_squared_norms()
        return univariate[self.multi_indices].prod(dim=1)

    def basis(self, x: ch.Tensor) -> ch.Tensor:
        """Evaluate every ``P_V`` in the basis.

        Args:
            x: Batch of shape ``(n, dims)``, or ``(n,)`` when dims is one.

        Returns:
            Tensor of shape ``(n, num_terms)``.
        """
        univariate = self.univariate(self._as_batch(x))
        values = univariate[:, 0, self.multi_indices[:, 0]]
        for i in range(1, self.dims):
            values = values * univariate[:, i, self.multi_indices[:, i]]
        return values

    def fit(self, x: ch.Tensor) -> "Polynomial":
        """Set the coefficients from samples, ``c_V = mean(P_V) / ||P_V||^2``.

        For samples from a law ``D``, these are unbiased estimates of the
        coefficients of ``dD / dw`` when the family is orthogonal under w.

        Args:
            x: Samples of shape ``(n, dims)``, or ``(n,)`` when dims is one.

        Returns:
            self.
        """
        x = self._as_batch(x)
        total = ch.zeros(self.num_terms, dtype=x.dtype, device=x.device)
        for chunk in x.split(self.chunk_size()):
            total += self.basis(chunk).sum(0)
        norms = self.squared_norms().to(dtype=x.dtype, device=x.device)
        self.coefficients = total / x.size(0) / norms
        return self

    def __call__(self, x: ch.Tensor) -> ch.Tensor:
        """Evaluate ``sum_V c_V P_V``.

        Args:
            x: Batch of shape ``(n, dims)``, or ``(n,)`` when dims is one.

        Returns:
            Tensor of shape ``(n,)``.

        Raises:
            RuntimeError: If the polynomial has no coefficients yet.
            ValueError: If there is not one coefficient per basis polynomial.
        """
        if self.coefficients is None:
            raise RuntimeError("the polynomial has no coefficients; call fit first.")
        if self.coefficients.shape != (self.num_terms,):
            raise ValueError(
                f"expected {self.num_terms} coefficients; got "
                f"{tuple(self.coefficients.shape)}."
            )
        x = self._as_batch(x)
        coefficients = self.coefficients.to(x.dtype)
        return ch.cat(
            [self.basis(chunk) @ coefficients for chunk in x.split(self.chunk_size())]
        )

    def chunk_size(self) -> int:
        """Return how many rows to evaluate at once within ``CHUNK_ELEMENTS``."""
        return max(1, CHUNK_ELEMENTS // self.num_terms)

    def _as_batch(self, x: ch.Tensor) -> ch.Tensor:
        """Return x shaped ``(n, dims)``, accepting ``(n,)`` when dims is one.

        Raises:
            ValueError: If x is neither shape.
        """
        if x.dim() == 1 and self.dims == 1:
            x = x[:, None]
        if x.dim() != 2 or x.size(1) != self.dims:
            raise ValueError(
                f"expected a batch of shape (n, {self.dims}); got {tuple(x.shape)}."
            )
        return x
