# Author: pstefanou12@
"""Stateful truncation sets.

A known truncation set is a predicate the caller supplies as a plain
callable. The sets here carry state: ``Identity`` admits everything,
``RecoveredSet`` is estimated from samples drawn inside it, and
``PolynomialThreshold`` is learned from labeled samples.
"""

import abc
import copy
from typing import Optional, cast
import warnings

import torch as ch
import torch.distributions

from delphi.learning import l1_regression
from delphi.torchnomials import hermite
import delphi.torchnomials.polynomial
import delphi.transformations.transformation


class Oracle(abc.ABC):  # pylint: disable=too-few-public-methods
    """Base class for stateful truncation sets."""

    @abc.abstractmethod
    def __call__(self, x):
        """Return membership of each sample.

        Args:
            x: Samples to check membership of.
        """


class Identity(Oracle):  # pylint: disable=too-few-public-methods
    """Oracle admitting every sample: nothing is truncated.

    Survival mass is 1, so a procedure given this set has nothing to correct
    for and reduces to the untruncated MLE.
    """

    def __call__(self, x: ch.Tensor) -> ch.Tensor:
        """Return membership for every sample.

        Args:
            x: Samples of shape (..., dims).

        Returns:
            Mask over the batch dims, all true.
        """
        return ch.ones(x.shape[:-1], dtype=ch.bool)


class RecoveredSet(Oracle):  # pylint: disable=too-few-public-methods
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
        self.distribution: Optional[torch.distributions.Distribution] = None
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
        self.polynomial = hermite.Hermite(
            self.degree, z.size(1), prob=True, norm=True
        ).fit(z)
        self._samples = z
        return self

    def given(self, distribution: torch.distributions.Distribution) -> "RecoveredSet":
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
        reference = torch.distributions.MultivariateNormal(
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


class PolynomialThreshold(Oracle):  # pylint: disable=too-few-public-methods
    """The set ``{x : p(x) >= tau}`` of a polynomial fit to 0/1 labels.

    Kalai, Klivans, Mansour and Servedio (2008): fit p over a polynomial
    basis by L1 regression on the labels, then take the threshold tau with
    the least weighted error on the same sample. The basis decides the
    hypothesis class and how well conditioned the regression is.

    Learning a truncation set, the labels come from the positive-unlabeled
    reduction (Lee et al., 2024, Definition 4): 1 for the observed samples,
    all inside the set, and 0 for draws from a reference distribution,
    inside or outside it.

    Usage::

        learned = PolynomialThreshold(hermite.Hermite(4, 2, prob=True, norm=True))
        learned.fit(X, y)
        inside = learned(X_new)

    Attributes:
        polynomial: The basis, whose coefficients fit sets.
        threshold: tau, set by fit.
        regression: The fitted L1 regression, set by fit.
    """

    def __init__(
        self,
        polynomial: delphi.torchnomials.polynomial.Polynomial,
        *,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize an unfit threshold over a polynomial basis.

        Args:
            polynomial: The basis to regress over; fit overwrites its
                coefficients.
            hyperparameters: Overrides of the L1 regression's config.
        """
        self.polynomial = polynomial
        self.hyperparameters = hyperparameters
        self.threshold: Optional[float] = None
        self.regression: Optional[l1_regression.L1Regression] = None

    def fit(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        y: ch.Tensor,
        weights: Optional[ch.Tensor] = None,
    ) -> "PolynomialThreshold":
        """Fit p by L1 regression on the labels, then choose tau.

        Args:
            X: Samples of shape (n, dims).
            y: 0/1 labels, shaped (n,).
            weights: Nonnegative sample weights, shaped (n,); ones if None.

        Returns:
            self.
        """
        design = self.polynomial.basis(X).to(ch.float64)
        target = y.to(design)
        weights = ch.ones_like(target) if weights is None else weights.to(design)
        # The basis is orthonormal under its own weight, not under this
        # sample: at degree 24 its columns reached condition number 5e9 and
        # the solver stalled. L1 regression's optimum does not depend on the
        # basis, so solve in one orthonormal under the weighted sample.
        _, factor = ch.linalg.qr(design * weights.sqrt()[:, None])  # pylint: disable=not-callable
        orthonormal = ch.linalg.solve_triangular(factor, design, upper=True, left=False)  # pylint: disable=not-callable
        self.regression = l1_regression.L1Regression(
            hyperparameters=self.hyperparameters
        ).fit(orthonormal, target, weights)
        if not self.regression.converged_:
            warnings.warn(
                "the L1 regression stopped without converging (duality gap "
                f"{self.regression.gap_:.2e}); the threshold may be poor.",
                stacklevel=2,
            )
        solution = cast(ch.Tensor, self.regression.coef_)[:, None]
        coefficients = ch.linalg.solve_triangular(factor, solution, upper=True)[:, 0]  # pylint: disable=not-callable
        self.polynomial.coefficients = coefficients
        self.threshold = _least_error_threshold(design @ coefficients, target, weights)
        return self

    def __call__(self, X: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        """Return membership, ``p(x) >= tau``.

        Args:
            X: Samples of shape (n, dims).

        Returns:
            Boolean mask over the batch.

        Raises:
            RuntimeError: If called before fit.
        """
        if self.threshold is None:
            raise RuntimeError("the threshold has not been fit.")
        return self.polynomial(X) >= self.threshold


def _least_error_threshold(
    scores: ch.Tensor, labels: ch.Tensor, weights: ch.Tensor
) -> float:
    """Return the tau minimizing the weighted error of ``scores >= tau``.

    Candidates are the midpoints between consecutive distinct scores, plus
    one below and one above them all; sorting makes every candidate's error
    a cumulative sum.
    """
    order = scores.argsort()
    sorted_scores = scores[order]
    positive = weights[order] * labels[order]
    negative = weights[order] * (1.0 - labels[order])
    # Cutting after k predicts the first k + 1 negative and the rest positive.
    missed = ch.cat([positive.new_zeros(1), positive.cumsum(0)])
    false_alarms = negative.sum() - ch.cat([negative.new_zeros(1), negative.cumsum(0)])
    errors = missed + false_alarms
    # Only cut between distinct scores, or tied samples would split.
    distinct = ch.cat(
        [
            ch.ones(1, dtype=ch.bool, device=scores.device),
            sorted_scores[1:] > sorted_scores[:-1],
            ch.ones(1, dtype=ch.bool, device=scores.device),
        ]
    )
    errors = ch.where(distinct, errors, ch.full_like(errors, float("inf")))
    cut = int(errors.argmin())
    if cut == 0:
        return float(sorted_scores[0]) - 1.0
    if cut == scores.numel():
        return float(sorted_scores[-1]) + 1.0
    return float(0.5 * (sorted_scores[cut - 1] + sorted_scores[cut]))
