# Author: pstefanou12@
"""Configuration model for the normal distribution under unknown truncation."""

from typing import Optional

import pydantic

from delphi.configs.base import DelphiConfig


class UnknownTruncationNormalConfig(DelphiConfig):
    """Configuration for the unknown-truncation normal estimators.

    Kontonis, Tzamos and Zampetakis (2019), Algorithm 1: a degree-``degree``
    Hermite expansion of the weighted indicator of the set, then projected
    stochastic gradient descent on the convex objective it defines.

    Attributes:
        degree: Total degree k of the Hermite expansion. The basis has
            ``C(k + dims, dims)`` terms, and the paper's error bound shrinks
            only as ``k ** (-1 / 4)``, so the default trades accuracy for a
            basis small enough to fit in a few seconds in low dimension.
        hermite_fraction: Share of the training samples that estimate the
            Hermite coefficients. The rest drive the descent, which the
            paper draws independently of the expansion.
        min_precision: Smallest eigenvalue the precision matrix may take in
            the whitened frame; from alpha when None.
        max_precision: Largest eigenvalue the precision matrix may take in
            the whitened frame; from alpha when None.
        loc_radius: Bound on the norm of the natural mean parameter in the
            whitened frame; from alpha and max_precision when None.
    """

    model_config = pydantic.ConfigDict(extra="ignore")

    batch_size: int = 100
    val: float = pydantic.Field(default=0.1, ge=0.0, le=1.0)
    record_params_every: int = pydantic.Field(default=1, ge=1)
    iterations: Optional[int] = pydantic.Field(default=2000, ge=1)
    scheduler: Optional[str] = "fisher"

    degree: int = pydantic.Field(default=10, ge=1)
    hermite_fraction: float = pydantic.Field(default=0.5, gt=0.0, lt=1.0)
    min_precision: Optional[float] = pydantic.Field(default=None, gt=0.0)
    max_precision: Optional[float] = pydantic.Field(default=None, gt=0.0)
    loc_radius: Optional[float] = pydantic.Field(default=None, gt=0.0)

    @pydantic.model_validator(mode="after")
    def check_precision_bounds(self) -> "UnknownTruncationNormalConfig":
        """Validate that min_precision does not exceed max_precision."""
        if (
            self.min_precision is not None
            and self.max_precision is not None
            and self.min_precision > self.max_precision
        ):
            raise ValueError(
                f"min_precision ({self.min_precision}) must be <= "
                f"max_precision ({self.max_precision})."
            )
        return self
