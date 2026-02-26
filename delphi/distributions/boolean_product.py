# Author: pstefanou12@
"""Boolean product distribution in natural parameterization."""

import torch as ch
from torch.distributions import Bernoulli


def calc_bool_prod_suff_stat(x):
    """Return sufficient statistics for boolean product distribution."""
    return x


class ExponentialFamilyBooleanProduct(Bernoulli):  # pylint: disable=abstract-method
    """Boolean product distribution parameterized by natural parameters."""

    def __init__(self, theta: ch.Tensor, dims: int):
        """Initialize with natural parameter theta and dimension."""
        self.dims = dims
        p = ch.exp(theta) / (1 + ch.exp(theta))
        super().__init__(p)

    def log_prob(self, value):
        """Compute summed log probability over all dimensions."""
        result = super().log_prob(value)
        return result.sum(-1)
