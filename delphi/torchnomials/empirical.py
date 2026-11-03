# Author: pstefanou12@
"""A polynomial basis made orthonormal under samples from any distribution.

Hermite polynomials are orthonormal only under a Gaussian, and Laguerre
polynomials only under a Gamma. A regression over polynomials is best
conditioned in a basis orthonormal under the data's own distribution, which
for most laws has no closed form. This module builds one from samples: same
polynomials, new basis, with the interface of every other family here.

For a law ``D`` that can be sampled, ``EmpiricalOrthonormal`` evaluates a base
family on draws from ``D``, factors their Gram matrix ``G = L L^T``, and uses
``P(x) L^{-T}``: orthonormal under the draws, and so under ``D`` up to
sampling error. The factor is triangular and the base orders its terms by
degree, so each degree-prefix of the basis spans the base's polynomials of
that degree, and truncating to a lower degree stays a prefix.
"""

from typing import Optional

import torch as ch
import torch.distributions

from delphi.torchnomials import polynomial

# Draws per basis term when the caller gives no sample size.
SAMPLES_PER_TERM = 20


class EmpiricalOrthonormal(polynomial.Polynomial):
    """A base family's span, orthonormalized under a reference distribution.

    Usage::

        reference = torch.distributions.Dirichlet(ch.ones(3))
        basis = EmpiricalOrthonormal(Laguerre(degree=6, dims=3), reference)
        values = basis.basis(points)

    Attributes:
        base: The family whose span this basis orthonormalizes.
        reference: The distribution the basis is orthonormal under.
        transform: ``L^{-T}``, mapping the base basis to this one.
    """

    def __init__(
        self,
        base: polynomial.Polynomial,
        reference: torch.distributions.Distribution,
        num_samples: Optional[int] = None,
        seed: int = 0,
    ):
        """Draw from the reference and orthonormalize the base under the draws.

        Args:
            base: The family to orthonormalize; its degree and dims carry over.
            reference: A distribution over ``(base.dims,)`` points.
            num_samples: Draws estimating the Gram matrix;
                ``SAMPLES_PER_TERM`` per basis term by default.
            seed: Seed of the draws, which leaves the caller's RNG unchanged.

        Raises:
            ValueError: If there are fewer draws than basis terms, or the
                Gram matrix is singular, as when a discrete reference has too
                few support points for the degree.
        """
        super().__init__(base.degree, base.dims)
        self.base = base
        self.reference = reference
        num_samples = num_samples or SAMPLES_PER_TERM * base.num_terms
        if num_samples < base.num_terms:
            raise ValueError(
                f"need at least {base.num_terms} draws for {base.num_terms} "
                f"terms; got {num_samples}."
            )
        with ch.random.fork_rng():
            ch.manual_seed(seed)
            draws = reference.sample((num_samples,)).to(ch.float64)
        gram = ch.zeros(base.num_terms, base.num_terms, dtype=ch.float64)
        for chunk in self._as_batch(draws).split(base.chunk_size()):
            values = base.basis(chunk).to(ch.float64)
            gram += values.T @ values
        factor, info = ch.linalg.cholesky_ex(gram / num_samples)  # pylint: disable=not-callable
        if info.item() != 0:
            raise ValueError(
                "the Gram matrix under the reference is singular: its support "
                "cannot separate every polynomial of this degree."
            )
        identity = ch.eye(base.num_terms, dtype=ch.float64)
        self.transform = ch.linalg.solve_triangular(factor, identity, upper=False).T  # pylint: disable=not-callable

    def basis(self, x: ch.Tensor) -> ch.Tensor:
        x = self._as_batch(x)
        return self.base.basis(x) @ self.transform.to(x.dtype)

    def squared_norms(self) -> ch.Tensor:
        """Return ones: the basis is orthonormal under the reference draws."""
        return ch.ones(self.num_terms, dtype=ch.float64)

    def weight(self) -> torch.distributions.Distribution:
        """Return the reference distribution."""
        return self.reference
