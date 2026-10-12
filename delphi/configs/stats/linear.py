# Author: pstefanou12@
"""Base configuration model for the delphi.stats linear models."""

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
