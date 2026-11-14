# Author: pstefanou12@
"""Configuration model for the models fit by projected SGD."""

import pydantic

from delphi.configs.models.statistical import StatisticalModelConfig


class PSGDModelConfig(StatisticalModelConfig):
    """Base config of the models fit by projected SGD over growing radii.

    Attributes:
        rate: Factor the projection radius grows by per phase when alpha is
            unknown.
        max_radius: Largest radius those phases reach.
    """

    rate: float = pydantic.Field(default=1.5, gt=1.0)
    max_radius: float = pydantic.Field(default=50.0, gt=0.0)
