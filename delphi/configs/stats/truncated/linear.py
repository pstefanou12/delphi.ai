# Author: pstefanou12@
"""Base configuration models for the delphi.stats linear models.

``LinearModelConfig`` covers every linear model; the truncated ones add a
truncation set via ``TruncatedLinearModelConfig``.
"""

import pydantic

from delphi.configs.stats.linear import LinearModelConfig


class TruncatedLinearModelConfig(LinearModelConfig):
    """Shared configuration for the truncated delphi.stats linear models.

    Separate from LinearModelConfig because alpha and c_s describe a
    truncation set, which an untruncated linear model such as softmax
    regression does not have.

    Attributes:
        alpha: Survival probability lower bound of the truncation set.
        c_s: Scaling constant for the dependent-data survival threshold.
    """

    alpha: float = pydantic.Field(default=1.0, gt=0.0, le=1.0)
    c_s: float = pydantic.Field(default=10.0, gt=0.0)
