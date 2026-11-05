# Author: pstefanou12@
"""Configurations for the delphi.stats linear models."""

from typing import Literal, Optional

import pydantic

from delphi.configs.base import DelphiConfig, StatisticalModelConfig


class LinearModelConfig(StatisticalModelConfig):
    """Shared configuration for the delphi.stats truncated linear models."""

    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # Must be > 0: fit() reads back best/final params, which the trainer only
    # records when this is enabled.
    record_params_every: int = pydantic.Field(default=1, ge=1)


class LassoConfig(DelphiConfig):
    """Configuration for ``Lasso``.

    Each iteration is one accelerated proximal gradient step, so
    ``iterations`` bounds the steps, not the passes over the data.

    Attributes:
        tolerance: Bound on the duality gap, relative to the objective at
            zero, at which the solver stops; float32 cannot reach the
            default.
        bisection_steps: Halvings of the log-penalty interval that
            ``fit_constrained`` searches.
    """

    iterations: Optional[int] = pydantic.Field(default=20000, ge=1)
    dtype: Literal["float32", "float64"] = "float64"
    tolerance: float = pydantic.Field(default=1e-8, gt=0.0)
    bisection_steps: int = pydantic.Field(default=20, ge=1)
