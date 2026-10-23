# Author: pstefanou12@
"""Base configuration models for the delphi.stats linear models.

``LinearModelConfig`` covers every linear model; the truncated ones add a
truncation set via ``TruncatedLinearModelConfig``.
"""

from typing import Literal

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
        num_samples: Rejection-sampling draws per criterion evaluation.
        eps: Numerical stability constant for the criterion.
        project: Keep the iterates in the model's projection set.
        start: Start the descent at the untruncated MLE, or at zero; either
            is projected onto the projection set first.
    """

    alpha: float = pydantic.Field(default=1.0, gt=0.0, le=1.0)
    num_samples: int = pydantic.Field(default=1000, ge=1)
    eps: float = pydantic.Field(default=1e-5, gt=0.0)
    project: bool = True
    start: Literal["untruncated", "zero"] = "untruncated"
