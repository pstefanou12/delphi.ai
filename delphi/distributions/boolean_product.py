# Author: pstefanou12@
"""Boolean product distribution in natural parameterization."""

import torch as ch
import torch.distributions as distributions


class ExponentialFamilyBooleanProduct(distributions.Bernoulli):  # pylint: disable=abstract-method
    """Boolean product distribution parameterized by natural parameters."""

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        p = ch.exp(theta) / (1 + ch.exp(theta))
        super().__init__(p)

    @staticmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Return sufficient statistics for boolean product distribution."""
        return x

    @staticmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert canonical probability to natural log-odds parameter."""
        return ch.log(theta / (1 - theta))

    @staticmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert natural log-odds to canonical probability parameter."""
        return ch.exp(theta) / (1 + ch.exp(theta))

    def log_prob(self, value):
        """Compute summed log probability over all dimensions."""
        result = super().log_prob(value)
        return result.sum(-1)
