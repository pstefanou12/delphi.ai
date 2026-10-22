# Author: pstefanou12@
"""Stateful truncation sets.

A known truncation set is a predicate the caller supplies as a plain
callable. The sets here carry state: ``Identity`` admits everything, and
``RecoveredSet`` is not given but estimated from samples drawn inside it.
"""

from abc import ABC
import copy
from typing import Optional

import torch as ch
from torch import distributions as torch_distributions

from delphi.torchnomials import hermite
import delphi.transformations.transformation


class oracle(ABC):  # pylint: disable=invalid-name,too-few-public-methods
    """Base class for stateful truncation sets."""

    def __call__(self, x):
        """Return membership of each sample.

        Args:
            x: Samples to check membership of.
        """


class Identity(oracle):  # pylint: disable=too-few-public-methods
    """Oracle admitting every sample: nothing is truncated.

    Survival mass is 1, so a procedure given this set has nothing to correct
    for and reduces to the untruncated MLE.
    """

    def __call__(self, x: ch.Tensor) -> ch.Tensor:
        """Return membership for every sample.

        Args:
            x: Samples of shape (n, dims).

        Returns:
            Mask over the batch, all true.
        """
        return ch.ones(x.size(0), dtype=ch.bool)


class RecoveredSet(oracle):  # pylint: disable=too-few-public-methods
    """A truncation set recovered from samples drawn inside it.

    Kontonis et al. (2019), Theorem 6. Fit to samples from a law ``D``
    supported on the set S, the probabilist Hermite expansion ``psi_k``
    estimates ``dD / dN(0, I)``, which is ``1_S p / (alpha N(0, I))`` for the
    untruncated density p. Given a fitted ``p``, the ratio
    ``psi_k N(0, I) / p`` approximates ``1_S / alpha``: ``1 / alpha`` on the
    set and zero off it. Any fitted distribution with a ``log_prob`` serves.

    The paper thresholds the ratio at one half. When alpha is small, that
    sits next to the outside level, where the expansion's ringing crosses
    it: a Sierpinski set at 2% mass came back wrong on 24% of the
    population. Here the threshold is half the ratio's median over the
    fitting samples, all inside S, which estimates the midpoint between the
    two levels without knowing alpha; the same set came back wrong on 6%.

    Usage::

        recovered = RecoveredSet(degree, standardization).fit(samples)
        inside = recovered.given(fitted_distribution)(points)

    Attributes:
        degree: Degree of the Hermite expansion.
        transformation: Coordinates the expansion is fit in; the identity
            when None.
        polynomial: The fitted expansion, set by fit.
        distribution: The fitted distribution in those coordinates, set by
            ``given``.
        threshold: The membership threshold on the ratio, set by ``given``.
    """

    def __init__(
        self,
        degree: int,
        transformation: Optional[
            delphi.transformations.transformation.Transformation
        ] = None,
    ):
        """Initialize an unfit set estimate.

        Args:
            degree: Degree of the Hermite expansion.
            transformation: Fitted change of coordinates to expand in.
        """
        self.degree = degree
        self.transformation = transformation
        self.polynomial: Optional[hermite.Hermite] = None
        self.distribution: Optional[torch_distributions.Distribution] = None
        self.threshold: Optional[float] = None
        # The fitting samples, in the expansion's coordinates, which set the
        # threshold once a distribution is given.
        self._samples: Optional[ch.Tensor] = None

    def fit(self, samples: ch.Tensor) -> "RecoveredSet":
        """Fit the expansion to samples from inside the set.

        Args:
            samples: Samples of shape (n, dims), in original coordinates.

        Returns:
            self.
        """
        z = self.transform(samples)
        self.polynomial = hermite.Hermite(self.degree, z.size(1), prob=True, norm=True)
        self.polynomial.fit(z)
        self._samples = z
        return self

    def given(self, distribution: torch_distributions.Distribution) -> "RecoveredSet":
        """Return a copy that decides membership against a fitted distribution.

        Args:
            distribution: The fitted untruncated distribution, in the
                transformation's coordinates.

        Returns:
            The bound copy.

        Raises:
            RuntimeError: If called before fit.
        """
        if self._samples is None:
            raise RuntimeError("the recovered set has not been fit.")
        bound = copy.copy(self)
        bound.distribution = distribution
        bound.threshold = 0.5 * float(bound.ratio(self._samples).median())
        return bound

    def transform(self, x: ch.Tensor) -> ch.Tensor:
        """Return x in the expansion's coordinates, in float64."""
        if self.transformation is None:
            return x.to(ch.float64)
        return self.transformation.transform(x)

    def weighted_indicator(self, z: ch.Tensor) -> ch.Tensor:
        """Return ``psi_k``, clipped at zero as Theorem 4 defines it.

        Args:
            z: Points of shape (n, dims), in the expansion's coordinates.

        Returns:
            Shaped (n,).

        Raises:
            RuntimeError: If called before fit.
        """
        if self.polynomial is None:
            raise RuntimeError("the recovered set has not been fit.")
        return self.polynomial(z).clamp(min=0.0)

    def ratio(self, z: ch.Tensor) -> ch.Tensor:
        """Return ``psi_k N(0, I) / p``, which approximates ``1_S / alpha``.

        Args:
            z: Points of shape (n, dims), in the expansion's coordinates.

        Returns:
            Shaped (n,).

        Raises:
            RuntimeError: If no fitted distribution has been given.
        """
        if self.distribution is None:
            raise RuntimeError("membership needs a fitted distribution; call given.")
        reference = torch_distributions.MultivariateNormal(
            ch.zeros(z.size(1), dtype=z.dtype), ch.eye(z.size(1), dtype=z.dtype)
        )
        log_ratio = reference.log_prob(z) - self.distribution.log_prob(z)
        return self.weighted_indicator(z) * log_ratio.exp()

    def __call__(self, x: ch.Tensor) -> ch.Tensor:
        """Return membership in the recovered set.

        Args:
            x: Samples of shape (n, dims), in original coordinates.

        Returns:
            Boolean mask over the batch.

        Raises:
            RuntimeError: If no fitted distribution has been given.
        """
        if self.threshold is None:
            raise RuntimeError("membership needs a fitted distribution; call given.")
        return self.ratio(self.transform(x)) > self.threshold
