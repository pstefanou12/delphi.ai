# Author: pstefanou12@
"""Poisson distribution in natural parameterization."""

import torch as ch
import torch.distributions as distributions


class ExponentialFamilyPoisson(distributions.Poisson):  # pylint: disable=abstract-method
    """Poisson distribution parameterized by natural parameters."""

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        lambda_ = ch.exp(theta)
        super().__init__(lambda_)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return sufficient statistics for Poisson distribution."""
        return x

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert canonical rate parameter to natural log form."""
        return ch.log(theta)

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert natural parameters to canonical rate parameter."""
        return ch.exp(theta)

    def log_prob(self, value):
        """Compute summed log probability over all dimensions."""
        result = super().log_prob(value)
        return result.sum(-1)
