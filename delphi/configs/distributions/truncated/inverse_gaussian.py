# Author: pstefanou12@
"""Configuration model for the truncated inverse Gaussian distribution."""

from delphi.configs.distributions.truncated.exponential_family import (
    TruncatedExponentialFamilyDistributionConfig,
)


class TruncatedInverseGaussianConfig(TruncatedExponentialFamilyDistributionConfig):
    """Configuration for truncated inverse Gaussian distributions."""

    # Cut to a tail, x and 1 / x make the Hessian's condition number ~350.
    precondition: bool = True
