# Author: pstefanou12@
"""Truncated normal distribution with known variance."""

from delphi.truncated.distributions.truncated_multivariate_normal_known_covariance import (
    TruncatedMultivariateNormalKnownCovariance,
)


class TruncatedNormalKnownVariance(TruncatedMultivariateNormalKnownCovariance):
    """Truncated normal distribution with known variance.

    Inherits all constructor arguments from
    TruncatedMultivariateNormalKnownCovariance:
        args (Parameters): hyperparameter object
        phi (Callable): truncation set oracle
        alpha (float): survival probability lower bound
        dims (int): number of dimensions
        covariance_matrix (Optional[Tensor]): known variance as a 1x1 matrix
        sampler (Callable): optional sampler override
    """

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated normal distribution known variance"
