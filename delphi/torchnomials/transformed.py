# Author: pstefanou12@
"""A polynomial family in transformed coordinates.

``Transformed(base, t)`` evaluates ``base`` at ``t(x)``: polynomials in
``log x`` rather than ``x``, say. Its terms are not polynomials in ``x``, but
they keep the interface, so anything that regresses on a basis takes them.
On a heavy tail the powers of ``x`` are dominated by a handful of extreme
draws, while the powers of ``log x`` are not.
"""

import torch as ch
import torch.distributions
import torch.distributions.transforms

from delphi.torchnomials import polynomial


class Transformed(polynomial.Polynomial):
    """A base family evaluated in coordinates ``t(x)``.

    Usage::

        log = torch.distributions.transforms.ExpTransform().inv
        basis = Transformed(Hermite(4, 1, prob=True, norm=True), log)
        values = basis.basis(points)

    Attributes:
        base: The family evaluated in the new coordinates.
        transform: The map from the samples' coordinates to the base's.
    """

    def __init__(
        self,
        base: polynomial.Polynomial,
        transform: torch.distributions.transforms.Transform,
    ):
        """Initialize from a base family and a change of coordinates.

        Args:
            base: The family to evaluate; its degree and dims carry over.
            transform: A coordinatewise bijection onto the base's domain.
        """
        super().__init__(base.degree, base.dims)
        self.base = base
        self.transform = transform

    def basis(self, x: ch.Tensor) -> ch.Tensor:
        return self.base.basis(self.transform(self._as_batch(x)))

    def squared_norms(self) -> ch.Tensor:
        """Return the base's norms, which the change of variables keeps."""
        return self.base.squared_norms()

    def weight(self) -> torch.distributions.Distribution:
        """Return the base's weight, pulled back to the samples' coordinates."""
        return torch.distributions.TransformedDistribution(
            self.base.weight(), [self.transform.inv]
        )
