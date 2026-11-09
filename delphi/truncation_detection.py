# Author: pstefanou12@
"""Detecting whether a sample was truncated.

Lee, Mehrotra and Zampetakis (2026), Corollaries 5.2 and 5.3. A sample comes
either from a distribution Q or from Q truncated to a set S* with
``Q(S*) <= 1 - beta``. Learn the sample's support T' from it, against draws
from Q (Corollary 5.2) or from a reference R that Q is smooth with respect
to (Corollary 5.3). Learned to accuracy ``beta / 2``, ``Q(T') > 1 - beta / 2``
holds only when nothing was truncated, so it is the test.

The test reads ``Q(T')`` off fresh draws from the reference given, so it is
exact for Q itself; for an approximate reference, its mass of T' stands in
for Q's and must lie within about ``beta / 4`` of it.
"""

import dataclasses
from typing import Optional, cast

import torch as ch

from delphi.learning import positive_only
import delphi.oracle
import delphi.torchnomials.polynomial


@dataclasses.dataclass(frozen=True)
class TruncationTest:
    """The outcome of ``detect_truncation``.

    Attributes:
        truncated: Whether the sample was judged truncated.
        truncation_set: The learned support of the sample, T'.
        reference_mass: ``Q(T')``, estimated on fresh draws from the
            reference.
    """

    truncated: bool
    truncation_set: delphi.oracle.Oracle
    reference_mass: float


def detect_truncation(
    sample: ch.Tensor,
    reference: positive_only.Sampler,
    polynomial: delphi.torchnomials.polynomial.Polynomial,
    *,
    beta: float,
    hyperparameters: Optional[dict] = None,
) -> TruncationTest:
    """Test whether a sample from the reference was truncated.

    Args:
        sample: The observed samples, shaped (n, d).
        reference: A sampler of Q, or of a reference close to it,
            ``m -> (m, d)``.
        polynomial: The basis to learn the support over.
        beta: The least mass a truncation removes from Q, in (0, 1).
        hyperparameters: Overrides of ``PositiveOnlyLearnerConfig``.

    Returns:
        The test's outcome.

    Raises:
        ValueError: If beta lies outside (0, 1).
    """
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must lie in (0, 1); got {beta}.")
    learner = positive_only.PositiveOnlyLearner(
        polynomial, hyperparameters=hyperparameters
    ).fit(sample, reference)
    mass = cast(float, learner.reference_mass_)
    return TruncationTest(
        truncated=mass <= 1.0 - beta / 2,
        truncation_set=learner,
        reference_mass=mass,
    )
