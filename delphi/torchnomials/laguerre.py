# Author: pstefanou12@
"""Generalized Laguerre polynomials, normalized or not.

``L_n^(a)`` are orthogonal under the law ``Gamma(a + 1, 1)``, with density
``x^a exp(-x) / Gamma(a + 1)`` on ``[0, inf)``, and
``||L_n^(a)||^2 = Gamma(n + a + 1) / (n! Gamma(a + 1))``. ``a = 0`` gives the
Laguerre polynomials, orthonormal under ``Exp(1)``. Normalizing divides each
by its norm, which makes the basis orthonormal under its weight.
"""

import math

import torch as ch
import torch.distributions

from delphi.torchnomials import polynomial


class Laguerre(polynomial.ProductPolynomial):
    """Generalized Laguerre polynomials over a total-degree basis.

    Usage::

        laguerre = Laguerre(degree=10, dims=2, norm=True)
        laguerre.fit(samples)
        values = laguerre(points)

    Attributes:
        alpha: The parameter a of ``L_n^(a)``, above -1.
        norm: Each polynomial divided by its norm under the weight.
    """

    def __init__(
        self, degree: int, dims: int = 1, alpha: float = 0.0, norm: bool = False
    ):
        """Initialize an unfit Laguerre polynomial.

        Args:
            degree: Maximum total degree.
            dims: Number of variables.
            alpha: The parameter a; the weight is ``Gamma(a + 1, 1)``.
            norm: Normalize each polynomial under its weight measure.

        Raises:
            ValueError: If alpha is not above -1, degree is negative, or
                dims is not positive.
        """
        if alpha <= -1.0:
            raise ValueError(f"alpha must exceed -1; got {alpha}.")
        super().__init__(degree, dims)
        self.alpha = alpha
        self.norm = norm

    def univariate(self, x: ch.Tensor) -> ch.Tensor:
        """Return ``P_0, ..., P_degree`` at every coordinate of every row.

        Runs the normalized recurrence, ``sqrt((j + 1)(j + a + 1)) l_{j+1} =
        (2j + 1 + a - x) l_j - sqrt(j (j + a)) l_{j-1}``, which keeps values
        of order one where the raw one grows like ``C(j + a, j)``, then
        rescales when ``norm`` is off.

        Args:
            x: Batch of shape ``(n, dims)``, or ``(n,)`` when dims is one.

        Returns:
            Tensor of shape ``(n, dims, degree + 1)``.
        """
        x = self._as_batch(x)
        a = self.alpha
        values = [ch.ones_like(x)]
        if self.degree >= 1:
            values.append((1.0 + a - x) / math.sqrt(1.0 + a))
        for j in range(1, self.degree):
            values.append(
                (
                    (2 * j + 1 + a - x) * values[j]
                    - math.sqrt(j * (j + a)) * values[j - 1]
                )
                / math.sqrt((j + 1) * (j + 1 + a))
            )
        normalized = ch.stack(values, dim=-1)
        if self.norm:
            return normalized
        return normalized * self.univariate_squared_norms().sqrt().to(x.dtype)

    def univariate_squared_norms(self) -> ch.Tensor:
        """Return ``||P_j||^2``: one when normalized, else ``h_j`` above."""
        if self.norm:
            return ch.ones(self.degree + 1, dtype=ch.float64)
        j = ch.arange(self.degree + 1, dtype=ch.float64)
        log_norms = (
            ch.lgamma(j + self.alpha + 1)
            - ch.lgamma(j + 1)
            - math.lgamma(self.alpha + 1)
        )
        return log_norms.exp()

    def weight(self) -> torch.distributions.Distribution:
        """Return ``Gamma(a + 1, 1)`` in every coordinate, independently."""
        concentration = ch.full((self.dims,), self.alpha + 1.0, dtype=ch.float64)
        return torch.distributions.Independent(
            torch.distributions.Gamma(concentration, ch.ones_like(concentration)), 1
        )
