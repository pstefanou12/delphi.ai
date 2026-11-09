# Author: pstefanou12@
"""Learning a truncation set from samples inside it.

Lee, Mehrotra and Zampetakis (2024), Algorithm 5. Samples from the truncated
distribution are positive examples only, and no class as simple as
halfspaces is learnable from those alone. Their reduction (Definition 4)
adds unlabeled draws from a reference distribution close to the untruncated
one, labeled 0, at weight rho in the mixture. The positives keep weight
``1 - rho`` and label 1. Labels inside the set are then noisy but mostly 1,
labels outside it are all 0, and a polynomial threshold fit by L1 regression
(``delphi.oracle.PolynomialThreshold``) learns the set agnostically.

How close the reference is matters: where it outweighs the truncated
density inside the set, the mixture labels that region 0 too.

The constrained learner (``learner="constrained"``) instead follows Lee,
Mehrotra and Zampetakis (2026): the polynomial with the least L1 mass under
the reference that still holds the positives, refit in rounds inside the
intersection so far (``delphi.learning.positive_only``). It needs the
reference only smooth with respect to the truth, not close to it.
"""

from typing import Literal, Optional, Union

import torch as ch

from delphi.learning import positive_only
import delphi.oracle
import delphi.torchnomials.polynomial

# Weight of the unlabeled draws in the mixture. At 0.01, marking everything
# inside costs so little that a degree-2 threshold learned a half plane
# wrong on 38% of the population; 0.2 to 0.5 kept every set tested stable.
DEFAULT_RHO = 0.2


def positive_unlabeled_sample(
    positive: ch.Tensor, unlabeled: ch.Tensor, rho: float = DEFAULT_RHO
) -> tuple[ch.Tensor, ch.Tensor, ch.Tensor]:
    """Return Definition 4's mixture as one weighted, labeled sample.

    Args:
        positive: Samples from the truncated distribution, shaped (n, d).
        unlabeled: Draws from the reference distribution, shaped (m, d).
        rho: The unlabeled draws' total weight, in (0, 1).

    Returns:
        X, shaped (n + m, d); labels y, 1 for the positives and 0 for the
        draws; and weights summing to ``1 - rho`` over the positives and
        rho over the draws.

    Raises:
        ValueError: If rho lies outside (0, 1), either sample is empty, or
            their dimensions differ.
    """
    if not 0.0 < rho < 1.0:
        raise ValueError(f"rho must lie in (0, 1); got {rho}.")
    if positive.dim() != 2 or unlabeled.dim() != 2:
        raise ValueError("expected samples shaped (n, d) and (m, d).")
    if positive.size(1) != unlabeled.size(1):
        raise ValueError(
            f"positive has {positive.size(1)} dimensions; unlabeled has "
            f"{unlabeled.size(1)}."
        )
    n, m = positive.size(0), unlabeled.size(0)
    if n == 0 or m == 0:
        raise ValueError("both samples must be non-empty.")
    X = ch.cat([positive, unlabeled.to(positive)])  # pylint: disable=invalid-name
    y = ch.cat([X.new_ones(n), X.new_zeros(m)])
    weights = ch.cat([X.new_full((n,), (1.0 - rho) / n), X.new_full((m,), rho / m)])
    return X, y, weights


def learn_truncation_set(
    positive: ch.Tensor,
    unlabeled: Union[ch.Tensor, positive_only.Sampler],
    polynomial: delphi.torchnomials.polynomial.Polynomial,
    *,
    learner: Literal["pu", "constrained"] = "pu",
    rho: Optional[float] = None,
    hyperparameters: Optional[dict] = None,
) -> delphi.oracle.Oracle:
    """Learn the truncation set as a polynomial threshold, or an intersection.

    Args:
        positive: Samples from the truncated distribution, shaped (n, d).
        unlabeled: Draws from the reference distribution, shaped (m, d), or
            a sampler of it, ``m -> (m, d)``. The positive-unlabeled learner
            draws n from a sampler; the constrained one resamples a tensor.
        polynomial: The basis to regress over, in the samples' coordinates;
            best conditioned when orthonormal under the reference.
        learner: ``"pu"``, the positive-unlabeled reduction, or
            ``"constrained"``, ``positive_only.PositiveOnlyLearner``.
        rho: The positive-unlabeled mixture's reference weight, in (0, 1),
            ``DEFAULT_RHO`` when None; or the constrained learner's
            tolerance, its config's default when None.
        hyperparameters: Overrides of the L1 regression's config for
            ``"pu"``, or of ``PositiveOnlyLearnerConfig`` for
            ``"constrained"``.

    Returns:
        The learned set, as a membership oracle.

    Raises:
        ValueError: As ``positive_unlabeled_sample``, or if learner is
            neither ``"pu"`` nor ``"constrained"``.
    """
    if learner == "constrained":
        sampler = (
            positive_only.pool_sampler(unlabeled)
            if isinstance(unlabeled, ch.Tensor)
            else unlabeled
        )
        overrides = dict(hyperparameters or {})
        if rho is not None:
            overrides["rho"] = rho
        return positive_only.PositiveOnlyLearner(
            polynomial, hyperparameters=overrides
        ).fit(positive, sampler)
    if learner != "pu":
        raise ValueError(f"learner must be 'pu' or 'constrained'; got {learner!r}.")
    if not isinstance(unlabeled, ch.Tensor):
        unlabeled = unlabeled(positive.size(0))
    X, y, weights = positive_unlabeled_sample(  # pylint: disable=invalid-name
        positive, unlabeled, DEFAULT_RHO if rho is None else rho
    )
    return delphi.oracle.PolynomialThreshold(
        polynomial, hyperparameters=hyperparameters
    ).fit(X, y, weights)
