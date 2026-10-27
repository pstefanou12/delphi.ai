# Author: pstefanou12@
"""Base configuration models for the delphi.stats linear models.

``LinearModelConfig`` covers every linear model; the truncated ones add a
truncation set via ``TruncatedLinearModelConfig``.
"""

from typing import Literal, Optional

import pydantic

from delphi.configs.stats.linear import LinearModelConfig


class TruncatedLinearModelConfig(LinearModelConfig):
    """Shared configuration for the truncated delphi.stats linear models.

    Separate from LinearModelConfig because alpha describes a truncation set,
    which an untruncated linear model such as softmax regression does not
    have. alpha belongs here rather than lower down because both the
    truncated regression and the LQR estimator read it.

    Attributes:
        alpha: Survival probability lower bound of the truncation set.
        num_samples: Rejection-sampling draws per sample in the first round.
        min_accepted: Kept draws each sample collects, in further rounds,
            before its conditional mean is taken.
        eps: Numerical stability constant for the criterion.
        projection_set: The model's own projection set, or "none".
        radius: The projection set's radius; None derives it from alpha as
            ``4 log(2 / alpha) + 7``, taking alpha = 1 when it is unknown.
        rate: Factor the projection radius grows by per phase when alpha
            is not given.
        max_radius: Largest projection radius those phases reach.
        scale_covariates: Divide the design matrix by ``B sqrt(d + 1)``, for ``B``
            its largest entry, so every row has norm at most 1. Estimates
            are reported in the original units either way.
        bisection_steps: Halvings of the segment a projection searches.
        max_quadratic_entries: Largest quadratic form a projection set may
            precompute to speed up its membership test, in float64 entries.
    """

    alpha: float = pydantic.Field(default=1.0, gt=0.0, le=1.0)
    num_samples: int = pydantic.Field(default=1000, ge=1)
    min_accepted: int = pydantic.Field(default=10, ge=1)
    eps: float = pydantic.Field(default=1e-5, gt=0.0)
    projection_set: Literal["default", "none"] = "default"
    radius: Optional[float] = pydantic.Field(default=None, gt=0.0)
    rate: float = pydantic.Field(default=1.5, gt=1.0)
    max_radius: float = pydantic.Field(default=50.0, gt=0.0)
    scale_covariates: bool = True
    bisection_steps: int = pydantic.Field(default=40, ge=1)
    max_quadratic_entries: int = pydantic.Field(default=2**24, ge=0)
