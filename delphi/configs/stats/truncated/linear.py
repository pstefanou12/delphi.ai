# Author: pstefanou12@
"""Base configuration models for the delphi.stats linear models.

``LinearModelConfig`` covers every linear model; the truncated ones add a
truncation set via ``TruncatedLinearModelConfig``.
"""

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
    """

    alpha: float = pydantic.Field(default=1.0, gt=0.0, le=1.0)
