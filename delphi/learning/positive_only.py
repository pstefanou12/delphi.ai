# Author: pstefanou12@
"""Learning a set from samples inside it and a reference distribution.

Lee, Mehrotra and Zampetakis (2026), "Smoothed Analysis of Learning from
Positive Samples". The positives P are drawn from ``D*`` restricted to the
set; the reference D, which can only be sampled, need only be smooth with
respect to ``D*``, and its support may be larger. Each fit solves their
Program 18 over polynomials of degree at most k,

    min_p  mean_U |p(u)|   s.t.  mean_P min(p(x), 1) >= 1 - rho,

for draws U from the reference, and returns ``H = {p >= 1 - sqrt(rho)}``,
which holds at least ``1 - sqrt(rho)`` of P (Lemma 4.10). That threshold is
the one in ``[0, 1 - sqrt(rho)]`` holding the fewest draws of U, since the
count only falls as the threshold rises.

The program's Lagrangian at a multiplier ``lambda >= 0`` is
``mean_U |p(u)| + lambda mean_P max(1 - p(x), 0)``: weighted quantile
regression, at level 1/2 on U with target 0 and level 1 on P with target 1,
which ``L1Regression`` solves exactly. The shortfall falls as lambda rises:
lambda = 0 gives p = 0, and lambda = 1/rho already meets the constraint, as
the constant 1 does with Lagrangian value 1. The dual function is concave
and piecewise linear in lambda, and each solve gives a line touching it, so
a cutting-plane search on the bracket reaches the optimal multiplier in a
few solves. There an infeasible and a feasible fit both minimize the same
Lagrangian, on whose minimizers the objective and shortfall are affine, so
the convex combination of the two meeting the constraint is optimal. The
shortfall's convexity keeps that combination feasible regardless.

Algorithm 3 repeats a fit on fresh reference draws and keeps the set holding
the fewest. Algorithm 2 runs rounds, each restricting both P and the
reference to the intersection of the sets so far, and returns the
intersection: a reference whose support exceeds the truth's leaves a low
degree polynomial room it cannot carve away in one round (their Figure 2).
Given a list of references, one of them close to ``D*``, Algorithm 4 learns
against each, keeps those whose set holds a held-out share of P, and
intersects them.
"""

from collections.abc import Callable, Sequence
import copy
import math
from typing import NamedTuple, Optional, Union, cast
import warnings

import torch as ch

from delphi.configs.learning import positive_only as positive_only_configs
from delphi.learning import l1_regression
import delphi.oracle
import delphi.torchnomials.polynomial

# Draws n points, shaped (n, d), from a reference distribution.
Sampler = Callable[[int], ch.Tensor]

# Most Lagrangian solves the search for the multiplier makes.
MAX_MULTIPLIER_STEPS = 50

# The meeting value less a solve's Lagrangian value there bounds how far the
# returned fit's objective exceeds the optimum. Stopping at this, relative,
# took about 8 solves against 14 to the exact multiplier.
MULTIPLIER_TOLERANCE = 1e-3

# Singular values of the design below this, relative to the largest, mark
# directions the sample cannot tell from zero.
RANK_TOLERANCE = 1e-10

# Rejection sampling gives up after this many batches.
MAX_REJECTION_BATCHES = 20


class _Fit(NamedTuple):
    """A Lagrangian minimizer, its L1 mass f on the reference, its shortfall v."""

    coefficients: ch.Tensor
    mass: float
    shortfall: float


def constrained_regression(
    positive: ch.Tensor,
    reference: ch.Tensor,
    rho: float,
    *,
    hyperparameters: Optional[dict] = None,
) -> ch.Tensor:
    """Solve Program 18 over the span of a design's columns.

    Args:
        positive: The design at the positives, shaped (n, k).
        reference: The design at the reference draws, shaped (m, k). Its
            span must hold the constant 1.
        rho: The tolerance, in (0, 1).
        hyperparameters: Overrides of the L1 regression's config.

    Returns:
        The coefficients c, shaped (k,), of ``p = design @ c``.
    """
    positive, reference = positive.to(ch.float64), reference.to(ch.float64)
    n, m = positive.size(0), reference.size(0)
    # Program 18's optimum does not depend on the basis, and one orthonormal
    # under the sample keeps the Newton systems well conditioned. On a small
    # discrete support the columns can be dependent there, so the null
    # directions, which change nothing on the sample, are dropped.
    _, scales, directions = ch.linalg.svd(  # pylint: disable=not-callable
        ch.cat([reference / math.sqrt(m), positive / math.sqrt(n)]),
        full_matrices=False,
    )
    kept = scales > RANK_TOLERANCE * scales[0]
    to_orthonormal = directions[kept].T / scales[kept]
    design = ch.cat([reference, positive]) @ to_orthonormal
    targets = ch.cat([design.new_zeros(m), design.new_ones(n)])
    quantiles = ch.cat([design.new_full((m,), 0.5), design.new_ones(n)])

    def solve(multiplier: float) -> _Fit:
        # The solver's objective is 2 w rho_tau(r): |p| / m on the reference
        # rows, and multiplier max(1 - p, 0) / n on the positives.
        weights = ch.cat(
            [
                design.new_full((m,), 1.0 / m),
                design.new_full((n,), multiplier / (2 * n)),
            ]
        )
        coefficients = cast(
            ch.Tensor,
            l1_regression.L1Regression(hyperparameters=hyperparameters)
            .fit(design, targets, weights, quantiles)
            .coef_,
        )
        values = design @ coefficients
        return _Fit(
            coefficients, float(values[:m].abs().mean()), _shortfall(values[m:])
        )

    low = _Fit(design.new_zeros(design.size(1)), 0.0, 1.0)
    low, high = _bracket_multiplier(solve, low, solve(1.0 / rho), rho)
    weight = 0.0
    if low.shortfall > high.shortfall:
        weight = min(
            max((rho - high.shortfall) / (low.shortfall - high.shortfall), 0.0), 1.0
        )
    combined = weight * low.coefficients + (1.0 - weight) * high.coefficients
    return to_orthonormal @ combined


def _bracket_multiplier(
    solve: Callable[[float], _Fit], low: _Fit, high: _Fit, rho: float
) -> tuple[_Fit, _Fit]:
    """Return minimizers on either side of the optimal multiplier.

    The dual ``g(lambda) = min_p f + lambda v`` is concave and piecewise
    linear, and each fit's ``f + lambda v`` is a line touching it. The next
    multiplier is where the bracket's two lines meet; once a solve there
    touches that point, both fits minimize the Lagrangian at the optimal
    multiplier. Short of it, the meeting value less the solve's bounds the
    combined fit's excess objective.

    Args:
        solve: Returns the Lagrangian's minimizer at a multiplier.
        low: A fit with shortfall above rho.
        high: A fit with shortfall at most rho.
        rho: The tolerance.
    """
    for _ in range(MAX_MULTIPLIER_STEPS):
        if high.shortfall >= low.shortfall:
            break
        multiplier = (high.mass - low.mass) / (low.shortfall - high.shortfall)
        meeting = low.mass + multiplier * low.shortfall
        fit = solve(multiplier)
        value = fit.mass + multiplier * fit.shortfall
        if value >= meeting - MULTIPLIER_TOLERANCE * (1.0 + abs(meeting)):
            break
        if fit.shortfall > rho:
            low = fit
        else:
            high = fit
    return low, high


def _shortfall(values: ch.Tensor) -> float:
    """Return ``mean max(1 - p, 0)``, Program 18's constraint slack."""
    return float((1.0 - values).clamp_min(0.0).mean())


def pool_sampler(pool: ch.Tensor) -> Sampler:
    """Return a sampler drawing uniformly, with replacement, from a pool.

    Args:
        pool: Draws from a reference distribution, shaped (m, d).

    Returns:
        The sampler of the pool's empirical distribution.
    """

    def sample(count: int) -> ch.Tensor:
        return pool[ch.randint(pool.size(0), (count,), device=pool.device)]

    return sample


def _restricted(sample: Sampler, inside: delphi.oracle.Oracle, mass: float) -> Sampler:
    """Return a sampler of the reference restricted to a set, by rejection.

    Args:
        sample: The reference's sampler.
        inside: The set.
        mass: An estimate of the set's mass under the reference.
    """

    def restricted(count: int) -> ch.Tensor:
        kept: list[ch.Tensor] = []
        total = 0
        batch = math.ceil(1.2 * count / mass) + 100
        for _ in range(MAX_REJECTION_BATCHES):
            draws = sample(batch)
            draws = draws[inside(draws)]
            kept.append(draws)
            total += draws.size(0)
            if total >= count:
                break
        return ch.cat(kept)[:count]

    return restricted


class PositiveOnlyLearner(delphi.oracle.Oracle):  # pylint: disable=too-many-instance-attributes
    """A set learned from positive samples against reference distributions.

    Usage::

        learner = PositiveOnlyLearner(hermite.Hermite(4, 2, prob=True, norm=True))
        learner.fit(positive, lambda n: reference.sample((n,)))
        inside = learner(points)

    Attributes:
        polynomial: The basis; each fitted set holds its own copy.
        hyperparameters: The ``PositiveOnlyLearnerConfig``.
        hypotheses_: With one reference, the set of each round; with a list,
            the learner of each reference kept.
        learners_: With a list of references, each one's fitted learner.
        kept_: With a list of references, the indices of those kept.
        rounds_: The rounds run; with a list, one count per reference.
        reference_mass_: The learned set's mass under the reference,
            estimated on fresh draws; with a list, one per reference.
        coverage_: The fraction of the positives the learned set holds; with
            a list, of the held-out positives.
    """

    def __init__(
        self,
        polynomial: delphi.torchnomials.polynomial.Polynomial,
        *,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize an unfit learner.

        Args:
            polynomial: The basis to fit over, in the samples' coordinates;
                its span must hold the constants.
            hyperparameters: Overrides of ``PositiveOnlyLearnerConfig``.
        """
        self.polynomial = polynomial
        self.hyperparameters = positive_only_configs.PositiveOnlyLearnerConfig(
            **(hyperparameters or {})
        )
        self.hypotheses_: list[delphi.oracle.Oracle] = []
        self.learners_: list[PositiveOnlyLearner] = []
        self.kept_: tuple[int, ...] = ()
        self.rounds_: Union[int, tuple[int, ...]] = 0
        self.reference_mass_: Union[float, tuple[float, ...], None] = None
        self.coverage_: Optional[float] = None
        self._set: Optional[delphi.oracle.Intersection] = None

    def fit(
        self, positive: ch.Tensor, reference: Union[Sampler, Sequence[Sampler]]
    ) -> "PositiveOnlyLearner":
        """Learn the set from positives and one reference or a list of them.

        Args:
            positive: Samples from inside the set, shaped (n, d).
            reference: A sampler, ``n -> (n, d)`` draws from the reference;
                or a sequence of them, at least one close to the
                positives' distribution.

        Returns:
            self.

        Raises:
            ValueError: If positive is not shaped (n, d) with n > 0, or the
                sequence of references is empty.
        """
        if positive.dim() != 2 or positive.size(0) == 0:
            raise ValueError(
                f"positive must be shaped (n, d) with n > 0; got {tuple(positive.shape)}."
            )
        positive = positive.to(ch.float64)
        if callable(reference):
            self._fit_one(positive, reference)
        else:
            samplers = list(reference)
            if not samplers:
                raise ValueError("the sequence of references is empty.")
            self._fit_list(positive, samplers)
        return self

    def _fit_one(self, positive: ch.Tensor, sample: Sampler) -> None:
        """Run Algorithm 2 against one reference."""
        hp = self.hyperparameters
        hypotheses: list[delphi.oracle.Oracle] = []
        current = delphi.oracle.Intersection(hypotheses)
        for _ in range(hp.rounds):
            mass = float(current(self._draw(sample)).double().mean())
            inside = positive[current(positive)]
            if hypotheses and (mass <= hp.gamma or inside.size(0) == 0):
                break
            restricted = _restricted(sample, current, mass) if hypotheses else sample
            hypotheses.append(self._boosted(inside, restricted))
            current = delphi.oracle.Intersection(hypotheses)
        self.hypotheses_ = hypotheses
        self._set = current
        self.rounds_ = len(hypotheses)
        self.reference_mass_ = float(current(self._draw(sample)).double().mean())
        self.coverage_ = float(current(positive).double().mean())

    def _boosted(
        self, positive: ch.Tensor, sample: Sampler
    ) -> delphi.oracle.PolynomialThreshold:
        """Run Algorithm 3: return the fit holding the fewest fresh draws."""
        hp = self.hyperparameters
        draws = hp.reference_samples or positive.size(0)
        fits = [self._fit_set(positive, sample(draws)) for _ in range(hp.repetitions)]
        if len(fits) == 1:
            return fits[0]
        fresh = self._draw(sample)
        return min(fits, key=lambda fit: int(fit(fresh).sum()))

    def _fit_set(
        self, positive: ch.Tensor, reference: ch.Tensor
    ) -> delphi.oracle.PolynomialThreshold:
        """Solve Program 18 once and threshold at ``1 - sqrt(rho)``."""
        rho = self.hyperparameters.rho
        # Each fit reassigns only the coefficients; a deep copy would break
        # torch's weak reference from an inverse transform to its forward map.
        hypothesis = delphi.oracle.PolynomialThreshold(copy.copy(self.polynomial))
        basis = hypothesis.polynomial
        basis.coefficients = constrained_regression(
            basis.basis(positive), basis.basis(reference.to(positive)), rho
        )
        hypothesis.threshold = 1.0 - math.sqrt(rho)
        return hypothesis

    def _fit_list(self, positive: ch.Tensor, samplers: list[Sampler]) -> None:
        """Run Algorithm 4: learn against each reference and intersect the good."""
        hp = self.hyperparameters
        order = ch.randperm(positive.size(0), device=positive.device)
        holdout_size = max(1, int(hp.holdout_fraction * positive.size(0)))
        holdout, train = positive[order[:holdout_size]], positive[order[holdout_size:]]
        learners = [
            PositiveOnlyLearner(self.polynomial, hyperparameters=hp.model_dump()).fit(
                train, sample
            )
            for sample in samplers
        ]
        misses = [1.0 - float(learner(holdout).double().mean()) for learner in learners]
        kept = [i for i, miss in enumerate(misses) if miss <= hp.holdout_miss]
        if not kept:
            kept = [min(range(len(misses)), key=misses.__getitem__)]
            warnings.warn(
                f"every reference's set missed more than {hp.holdout_miss} of "
                f"the held-out positives; keeping the one missing {min(misses):.3f}.",
                stacklevel=3,
            )
        self.learners_ = learners
        self.kept_ = tuple(kept)
        self.hypotheses_ = [learners[i] for i in kept]
        self._set = delphi.oracle.Intersection(self.hypotheses_)
        self.rounds_ = tuple(cast(int, learner.rounds_) for learner in learners)
        self.reference_mass_ = tuple(
            float(self._set(self._draw(sample)).double().mean()) for sample in samplers
        )
        self.coverage_ = float(self._set(holdout).double().mean())

    def _draw(self, sample: Sampler) -> ch.Tensor:
        """Return fresh reference draws for evaluation, in float64."""
        return sample(self.hyperparameters.evaluation_samples).to(ch.float64)

    def __call__(self, x: ch.Tensor) -> ch.Tensor:
        """Return membership in the learned set.

        Args:
            x: Samples of shape (n, d).

        Returns:
            Boolean mask over the batch.

        Raises:
            RuntimeError: If called before fit.
        """
        if self._set is None:
            raise RuntimeError("the learner has not been fit.")
        return self._set(x.to(ch.float64))
