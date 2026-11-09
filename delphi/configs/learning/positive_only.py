# Author: pstefanou12@
"""Configuration for learning a set from positive samples."""

from typing import Optional

import pydantic


class PositiveOnlyLearnerConfig(pydantic.BaseModel):
    """Configuration for ``PositiveOnlyLearner``.

    The defaults were measured on a Gaussian disc and half plane, 5000
    positives at degree 4, rather than set from the paper's worst-case
    constants. After three rounds the sets erred on 0.07% to 0.22% at rho
    1e-4, 0.23% to 0.47% at 1e-3, and 0.9% to 1.8% at 1e-2, which cuts holes
    where the reference is dense. A second round always helped; a third
    changed nothing there but helped where the reference's support exceeds
    the truth's. Three
    repetitions matched one at three times the cost.

    Attributes:
        rho: Program 18's tolerance: the positives' mean shortfall of
            ``min(p, 1)`` below 1. The learned set keeps at least
            ``1 - sqrt(rho)`` of the positives.
        rounds: Most rounds of Algorithm 2, each refitting inside the
            intersection so far.
        gamma: Algorithm 2 stops once the intersection holds at most this
            fraction of the reference.
        repetitions: Fits per round on fresh reference draws, of which the
            one holding the fewest fresh draws is kept (Algorithm 3).
        reference_samples: Reference draws per fit; as many as the
            positives when None.
        evaluation_samples: Fresh reference draws for each round's stopping
            check, the choice among repetitions, and ``reference_mass_``.
        holdout_fraction: With a list of references, the fraction of the
            positives held out to check each reference's set (Algorithm 4).
        holdout_miss: With a list of references, the most a reference's set
            may miss of the held-out positives and still be intersected.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    rho: float = pydantic.Field(default=1e-4, gt=0.0, lt=1.0)
    rounds: int = pydantic.Field(default=3, ge=1)
    gamma: float = pydantic.Field(default=0.01, gt=0.0, lt=1.0)
    repetitions: int = pydantic.Field(default=1, ge=1)
    reference_samples: Optional[int] = pydantic.Field(default=None, ge=1)
    evaluation_samples: int = pydantic.Field(default=20000, ge=1)
    holdout_fraction: float = pydantic.Field(default=0.2, gt=0.0, lt=1.0)
    holdout_miss: float = pydantic.Field(default=0.05, ge=0.0, le=1.0)
