# Author: pstefanou12@
"""Abstract base class for delphi exponential family distributions."""

import abc

import torch as ch
from torch.distributions import exp_family


class ExponentialFamilyDistribution(exp_family.ExponentialFamily, abc.ABC):
    """Abstract base class for exponential family distributions in natural parameterization."""

    @staticmethod
    @abc.abstractmethod
    def calc_suff_stat(x: ch.Tensor) -> ch.Tensor:
        """Compute sufficient statistics of the distribution.

        Args:
            x: Observed samples of shape (num_samples, dims).

        Returns:
            Sufficient statistics tensor of shape (num_samples, stat_dims).
        """

    @staticmethod
    @abc.abstractmethod
    def to_natural(theta: ch.Tensor) -> ch.Tensor:
        """Convert canonical parameters to natural parameterization.

        Args:
            theta: Canonical parameters.

        Returns:
            Natural parameters.
        """

    @staticmethod
    @abc.abstractmethod
    def to_canonical(theta: ch.Tensor) -> ch.Tensor:
        """Convert natural parameters to canonical parameterization.

        Args:
            theta: Natural parameters.

        Returns:
            Canonical parameters.
        """
