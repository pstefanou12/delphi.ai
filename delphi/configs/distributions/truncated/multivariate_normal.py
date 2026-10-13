# Author: pstefanou12@
"""Configuration model for the truncated multivariate normal distribution."""

from typing import Optional

import pydantic

from delphi.configs.distributions.truncated.exponential_family import (
    TruncatedExponentialFamilyDistributionConfig,
)


class TruncatedMultivariateNormalConfig(TruncatedExponentialFamilyDistributionConfig):
    """Configuration for truncated multivariate normal distributions.

    Attributes:
        eigenvalue_lower_bound: Minimum eigenvalue enforced during the
            negative-definite cone projection of the precision matrix T.
        covariance_matrix_lr: Optional separate learning rate for the
            covariance matrix parameter; falls back to lr when None.
    """

    eigenvalue_lower_bound: float = pydantic.Field(default=1e-2, gt=0.0)
    covariance_matrix_lr: Optional[float] = pydantic.Field(default=None, gt=0.0)
