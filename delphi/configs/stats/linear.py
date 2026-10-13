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

    fit_intercept: bool = True

    # Must be > 0: fit() reads back best/final params, which the trainer only
    # records when this is enabled.
    record_params_every: int = pydantic.Field(default=1, ge=1)
