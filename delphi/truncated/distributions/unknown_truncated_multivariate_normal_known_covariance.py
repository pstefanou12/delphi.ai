# Author: pstefanou12@
"""Multivariate normal with known covariance, truncated to an unknown set.

Kontonis et al. (2019)'s estimator restricted to the mean: see
``unknown_truncated_multivariate_normal`` for the procedure.
"""

from typing import Optional

import torch as ch

from delphi.transformations import standardization
from delphi.truncated.distributions.unknown_truncated_multivariate_normal import (
    UnknownTruncationMultivariateNormalBase,
)


class UnknownTruncationMultivariateNormalKnownCovariance(
    UnknownTruncationMultivariateNormalBase
):
    """Mean of a normal with known covariance truncated to an unknown set.

    The sample is centered at its mean and standardized by the known
    covariance, so the standardized truth is ``N(mu, I)``, exactly isotropic as
    Theorem 2 asks, and ``B`` stays fixed at the identity: only ``u`` is
    learned.

    Attributes:
        covariance_matrix: The known covariance.
    """

    estimate_suffixes = ("loc_",)

    def __init__(
        self,
        hyperparameters: dict,
        dims: int,
        covariance_matrix: ch.Tensor,
        alpha: Optional[float] = None,
    ):
        """Initialize the estimator.

        Args:
            hyperparameters: Hyperparameter dict.
            dims: Number of dimensions.
            covariance_matrix: Known covariance, shaped (dims, dims).
            alpha: Survival mass of the set in (0, 1), if known.

        Raises:
            ValueError: As the base class, or if covariance_matrix is not
                shaped (dims, dims).
        """
        super().__init__(hyperparameters, dims, alpha)
        if covariance_matrix.shape != (dims, dims):
            raise ValueError(
                f"covariance_matrix must be shaped ({dims}, {dims}); got "
                f"{tuple(covariance_matrix.shape)}."
            )
        self.covariance_matrix = covariance_matrix

    def _num_params(self) -> int:
        return self.dims

    def _standardization(self) -> standardization.Standardization:
        return standardization.Standardization(self.covariance_matrix)

    def _statistic(self, z: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
        # With B = I, h + log N(0, I) = -<u, z>: the sample is centered.
        return -z, z.new_zeros(z.size(0))

    def _natural(self, theta: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
        return theta, ch.eye(self.dims, dtype=theta.dtype)

    def _largest_precision(self) -> float:
        return 1.0

    def _project(self, theta: ch.Tensor) -> ch.Tensor:
        return self._project_loc(theta)

    def _initial_theta(self) -> ch.Tensor:
        return ch.zeros(self.dims, dtype=ch.float64)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return (
            "multivariate normal distribution with known covariance under "
            "unknown truncation"
        )
