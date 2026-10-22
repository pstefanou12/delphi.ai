# Author: pstefanou12@
"""Configuration model for the normal under unknown truncation."""

from typing import Literal, Optional

import pydantic

from delphi.configs.base import DelphiConfig


class UnknownTruncationMultivariateNormalConfig(DelphiConfig):
    """Configuration for the normal under unknown truncation.

    Kontonis et al. (2019), Algorithm 1, minimized by full-batch L-BFGS
    rather than the paper's SGD, whose heavy-tailed minibatch gradients
    stalled short of the minimizer.

    Attributes:
        degree: Hermite degree; by default the largest within max_terms, up
            to max_degree. An explicit degree warns when its basis has more
            terms than coefficient samples.
        max_terms: Term budget for the default degree; by default the number
            of samples estimating the coefficients.
        max_degree: Cap on the default degree.
        hermite_fraction: Share of the training samples that estimate the
            coefficients; the rest drive the descent.
        min_precision: Smallest eigenvalue of the precision matrix in
            standardized coordinates; from alpha by default.
        max_precision: Largest eigenvalue of the precision matrix in
            standardized coordinates; from alpha by default.
        loc_radius: Bound on the natural mean parameter's norm in
            standardized coordinates; from alpha by default.
    """

    model_config = pydantic.ConfigDict(extra="ignore")

    # None is the whole descent sample.
    batch_size: Optional[int] = pydantic.Field(default=None, ge=1)  # type: ignore[assignment]
    val: float = pydantic.Field(default=0.1, ge=0.0, le=1.0)
    record_params_every: int = pydantic.Field(default=1, ge=1)
    iterations: Optional[int] = pydantic.Field(default=100, ge=1)
    optimizer: str = "lbfgs"
    lr: float = pydantic.Field(default=1.0, gt=0.0)
    scheduler: Optional[str] = None
    lbfgs_max_iter: int = pydantic.Field(
        default=50, ge=1, serialization_alias="max_iter"
    )
    lbfgs_line_search_fn: Optional[Literal["strong_wolfe"]] = pydantic.Field(
        default="strong_wolfe", serialization_alias="line_search_fn"
    )

    degree: Optional[int] = pydantic.Field(default=None, ge=1)
    max_terms: Optional[int] = pydantic.Field(default=None, ge=2)
    max_degree: int = pydantic.Field(default=1200, ge=1)
    hermite_fraction: float = pydantic.Field(default=0.5, gt=0.0, lt=1.0)
    min_precision: Optional[float] = pydantic.Field(default=None, gt=0.0)
    max_precision: Optional[float] = pydantic.Field(default=None, gt=0.0)
    loc_radius: Optional[float] = pydantic.Field(default=None, gt=0.0)

    @pydantic.model_validator(mode="after")
    def check_precision_bounds(self) -> "UnknownTruncationMultivariateNormalConfig":
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
