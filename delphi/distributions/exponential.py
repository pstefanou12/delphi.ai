# Author: pstefanou12@
"""Exponential distribution in natural parameterization."""

import torch as ch
import torch.distributions


class ExponentialFamilyExponential(ch.distributions.Exponential):  # pylint: disable=abstract-method
    """Exponential distribution parameterized by natural parameters."""

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        lambda_ = -theta
        super().__init__(lambda_)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return sufficient statistics for exponential distribution."""
        return x

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert canonical rate parameter to natural form."""
        return -theta

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert natural parameters to canonical rate parameter."""
        return -theta

    def log_prob(self, value):
        """Compute summed log probability over all dimensions."""
        result = super().log_prob(value)
        return result.sum(-1)
