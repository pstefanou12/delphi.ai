# Author: pstefanou12@
"""Configuration models for the untruncated delphi.stats classifiers."""

import pydantic

from delphi.configs.stats.linear import LinearModelConfig


class SoftmaxRegressionConfig(LinearModelConfig):
    """Configuration for truncated softmax (multinomial) regression.

    Attributes:
        num_samples: Monte Carlo samples drawn per criterion evaluation.
        multi_class: Multiclass strategy used for the empirical initialization.
    """

    num_samples: int = pydantic.Field(default=1000, ge=1)
    multi_class: str = "ovr"
