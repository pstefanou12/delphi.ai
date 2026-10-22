# Author: pstefanou12@
"""Hermite polynomials, physicist or probabilist, normalized or not.

The physicist polynomials ``H_n`` are orthogonal under the weight
``exp(-x^2) / sqrt(pi)``, the law ``N(0, 1/2)``, with ``||H_n||^2 = 2^n n!``.
The probabilist ones ``He_n`` are orthogonal under ``N(0, 1)``, with
``||He_n||^2 = n!``. Normalizing divides each by its norm, which makes the
basis orthonormal under its weight.
"""

import math

import torch as ch
from torch import distributions

from delphi.torchnomials import polynomial


class Hermite(polynomial.Polynomial):
    """Hermite polynomials over a total-degree basis.

    Usage::

        hermite = Hermite(degree=10, dims=2, prob=True, norm=True)
        hermite.fit(samples)
        values = hermite(points)

    Attributes:
        prob: Probabilist ``He_n`` rather than physicist ``H_n``.
        norm: Each polynomial divided by its norm under the weight.
    """

    def __init__(
        self, degree: int, dims: int = 1, prob: bool = False, norm: bool = False
    ):
        """Initialize an unfit Hermite polynomial.

        Args:
            degree: Maximum total degree.
            dims: Number of variables.
            prob: Use the probabilist polynomials; physicist by default.
            norm: Normalize each polynomial under its weight measure.

        Raises:
            ValueError: If degree is negative or dims is not positive.
        """
        super().__init__(degree, dims)
        self.prob = prob
        self.norm = norm

    def univariate(self, x: ch.Tensor) -> ch.Tensor:
        """Return ``P_0, ..., P_degree`` at every coordinate of every row.

        Runs the normalized recurrence, ``sqrt(j) h_j = a x h_{j-1} -
        sqrt(j - 1) h_{j-2}`` with ``a = 1`` (probabilist) or ``sqrt(2)``
        (physicist), which keeps values of order one where the raw
        recurrence overflows, then rescales when ``norm`` is off.

        Args:
            x: Batch of shape ``(n, dims)``, or ``(n,)`` when dims is one.

        Returns:
            Tensor of shape ``(n, dims, degree + 1)``.
        """
        x = self._as_batch(x)
        scale = 1.0 if self.prob else math.sqrt(2.0)
        values = [ch.ones_like(x)]
        if self.degree >= 1:
            values.append(scale * x)
        for j in range(2, self.degree + 1):
            values.append(
                (scale * x * values[j - 1] - math.sqrt(j - 1) * values[j - 2])
                / math.sqrt(j)
            )
        normalized = ch.stack(values, dim=-1)
        if self.norm:
            return normalized
        return normalized * self.univariate_squared_norms().sqrt().to(x.dtype)

    def univariate_squared_norms(self) -> ch.Tensor:
        """Return ``||P_j||^2``: one when normalized, else ``j!`` or ``2^j j!``."""
        if self.norm:
            return ch.ones(self.degree + 1, dtype=ch.float64)
        j = ch.arange(self.degree + 1, dtype=ch.float64)
        log_norms = ch.lgamma(j + 1)
        if not self.prob:
            log_norms = log_norms + j * math.log(2.0)
        return log_norms.exp()

    def weight(self) -> distributions.Distribution:
        """Return ``N(0, I)`` (probabilist) or ``N(0, I / 2)`` (physicist)."""
        variance = 1.0 if self.prob else 0.5
        return distributions.MultivariateNormal(
            ch.zeros(self.dims, dtype=ch.float64),
            variance * ch.eye(self.dims, dtype=ch.float64),
        )
