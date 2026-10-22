# Author: pstefanou12@
"""Stateful truncation sets.

A known truncation set is a predicate the caller supplies as a plain
callable. The sets here carry state: ``Identity`` admits everything, and
``RecoveredSet`` is not given but estimated from the data by the
unknown-truncation normal.
"""

from abc import ABC
from collections.abc import Callable

import torch as ch
from torch import distributions as torch_distributions

from delphi.torchnomials import hermite


class oracle(ABC):  # pylint: disable=invalid-name,too-few-public-methods
    """Base class for stateful truncation sets."""

    def __call__(self, x):
        """Return membership of each sample.

        Args:
            x: Samples to check membership of.
        """


class Identity(oracle):  # pylint: disable=too-few-public-methods
    """Oracle admitting every sample: nothing is truncated.

    Survival mass is 1, so a procedure given this set has nothing to correct
    for and reduces to the untruncated MLE.
    """

    def __call__(self, x: ch.Tensor) -> ch.Tensor:
        """Return membership for every sample.

        Args:
            x: Samples of shape (n, dims).

        Returns:
            Mask over the batch, all true.
        """
        return ch.ones(x.size(0), dtype=ch.bool)


class RecoveredSet(oracle):  # pylint: disable=too-few-public-methods
    """The set estimate of Kontonis et al. (2019), Theorem 6.

    In the frame the expansion was fit in, ``psi_k N(0, I) / N(mu*, Sigma*)``
    approximates ``1_S / alpha*``, which exceeds one on the set and is zero
    off it. With the fitted Gaussian in place of the truth, thresholding at
    one half recovers the set without knowing alpha.

    Attributes:
        expansion: The Hermite expansion ``psi_k``, in frame coordinates.
        to_frame: Maps original coordinates into the frame.
        fitted: The fitted Gaussian, in frame coordinates.
    """

    def __init__(
        self,
        expansion: hermite.HermiteExpansion,
        to_frame: Callable[[ch.Tensor], ch.Tensor],
        fitted: torch_distributions.MultivariateNormal,
    ):
        """Initialize the set estimate.

        Args:
            expansion: The Hermite expansion ``psi_k``, in frame coordinates.
            to_frame: Maps original coordinates into the frame.
            fitted: The fitted Gaussian, in frame coordinates.
        """
        self.expansion = expansion
        self.to_frame = to_frame
        self.fitted = fitted
        dims = fitted.loc.numel()
        self._reference = torch_distributions.MultivariateNormal(
            ch.zeros(dims, dtype=fitted.loc.dtype),
            ch.eye(dims, dtype=fitted.loc.dtype),
        )

    def __call__(self, x: ch.Tensor) -> ch.Tensor:
        """Return membership in the recovered set.

        Args:
            x: Samples of shape (n, dims), in original coordinates.

        Returns:
            Boolean mask over the batch.
        """
        z = self.to_frame(x).to(self.fitted.loc.dtype)
        log_ratio = self._reference.log_prob(z) - self.fitted.log_prob(z)
        return self.expansion(z) * log_ratio.exp() > 0.5
