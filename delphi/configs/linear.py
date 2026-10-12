# Author: pstefanou12@
"""Base configuration models for the delphi.stats linear models.

``LinearModelConfig`` covers every linear model; the truncated ones add a
truncation set via ``TruncatedLinearModelConfig``.
"""

import pydantic

from delphi.configs.base import DelphiConfig


class LinearModelConfig(DelphiConfig):
    """Shared configuration for the delphi.stats truncated linear models.

    Transcribed from the pre-pydantic ``*_DEFAULTS`` mappings in
    ``delphi.utils.defaults`` so defaults match the previous
    ``check_and_fill_args`` behaviour exactly.

    Attributes:
        fit_intercept: Fit an intercept term.
    """

    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # Dataset fields (val, workers, shuffle, batch_size, normalize) come from
    # DatasetConfig via DelphiConfig.
    fit_intercept: bool = True

    # Must be > 0: fit() reads back best/final params from the trainer, which
    # only records them when this is enabled. TRUNC_REG_DEFAULTS omitted the
    # key entirely, so the pre-pydantic path got None from
    # Parameters.__getattr__ and failed on "iterations % None".
    record_params_every: int = pydantic.Field(default=1, ge=1)


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
