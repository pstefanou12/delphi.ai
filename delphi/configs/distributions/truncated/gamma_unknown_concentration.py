# Author: pstefanou12@
"""Configuration model for the truncated Gamma with unknown shape."""

from typing import Optional

import pydantic

from delphi.configs.distributions.truncated.exponential_family import (
    WarmStartedExponentialFamilyDistributionConfig,
)


class TruncatedGammaUnknownConcentrationConfig(
    WarmStartedExponentialFamilyDistributionConfig
):
    """Configuration for truncated Gamma distributions with unknown shape."""

    # Cut to a tail, log x and x make the Hessian's condition number ~470.
    precondition: bool = True
    # A preconditioned step of ten samples ran a two-dimensional fit onto
    # its clamp along the shape-rate ridge.
    batch_size: int = 100
    # The larger batches converge in fewer steps.
    iterations: Optional[int] = pydantic.Field(default=1000, ge=1)
    warm_start_iterations: Optional[int] = pydantic.Field(
        default=250, ge=1, serialization_alias="iterations"
    )
