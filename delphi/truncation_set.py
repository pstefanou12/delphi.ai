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
"""

from typing import Optional

import torch as ch

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
    unlabeled: ch.Tensor,
    polynomial: delphi.torchnomials.polynomial.Polynomial,
    *,
    rho: float = DEFAULT_RHO,
    hyperparameters: Optional[dict] = None,
) -> delphi.oracle.PolynomialThreshold:
    """Learn the truncation set as a polynomial threshold.

    Args:
        positive: Samples from the truncated distribution, shaped (n, d).
        unlabeled: Draws from the reference distribution, shaped (m, d).
        polynomial: The basis to regress over, in the samples' coordinates;
            best conditioned when orthonormal under the reference.
        rho: The unlabeled draws' weight in the mixture, in (0, 1).
        hyperparameters: Overrides of the L1 regression's config.

    Returns:
        The learned set, as a membership oracle.

    Raises:
        ValueError: As ``positive_unlabeled_sample``.
    """
    X, y, weights = positive_unlabeled_sample(positive, unlabeled, rho)  # pylint: disable=invalid-name
    return delphi.oracle.PolynomialThreshold(
        polynomial, hyperparameters=hyperparameters
    ).fit(X, y, weights)
