# Author: pstefanou12@
"""Configuration model for the truncated Gumbel distribution."""

from delphi.configs.distributions.truncated.exponential_family import (
    TruncatedExponentialFamilyDistributionConfig,
)


class TruncatedGumbelConfig(TruncatedExponentialFamilyDistributionConfig):
    """Configuration for truncated Gumbel distributions with known scale."""

    # exp(-x / scale) spans orders of magnitude across coordinates of
    # different scale, so one step size left a cut coordinate where it began.
    precondition: bool = True
