# Author: pstefanou12@
"""Abstract base class for delphi exponential family distributions."""

import abc
from typing import Any

import torch as ch
from torch.distributions import exp_family


class ExponentialFamilyDistribution(exp_family.ExponentialFamily, abc.ABC):
    """Abstract base class for exponential family distributions in natural parameterization."""

    @staticmethod
    @abc.abstractmethod
    def calc_suff_stat(*args: Any) -> ch.Tensor:
        """Compute sufficient statistics of the distribution.

        Args:
            *args: Distribution-specific arguments. The first positional
                argument is conventionally the observed samples tensor of
                shape (num_samples, dims); subclasses may require additional
                known parameters (e.g. shape parameter k for Weibull).

        Returns:
            Sufficient statistics tensor of shape (num_samples, stat_dims).
        """

    @staticmethod
    @abc.abstractmethod
    def to_natural(*args: Any) -> ch.Tensor:
        """Convert canonical parameters to natural parameterization.

        Args:
            *args: Distribution-specific canonical parameters. The first
                positional argument is conventionally the canonical parameter
                tensor; subclasses may require additional known parameters
                (e.g. variance, covariance matrix, or shape).

        Returns:
            Natural parameters.
        """

    @staticmethod
    @abc.abstractmethod
    def to_canonical(*args: Any) -> ch.Tensor:
        """Convert natural parameters to canonical parameterization.

        Args:
            *args: Distribution-specific natural parameters. The first
                positional argument is conventionally the natural parameter
                tensor; subclasses may require additional known parameters.

        Returns:
            Canonical parameters.
        """
