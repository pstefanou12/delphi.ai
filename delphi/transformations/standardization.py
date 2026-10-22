# Author: pstefanou12@
"""Standardization: zero mean and identity covariance."""

from typing import Optional

import torch as ch

from delphi.transformations.transformation import Transformation


class Standardization(Transformation):
    """Maps x to ``L^-1 (x - mean)``, with ``L`` the Cholesky factor of a covariance.

    Fit on data, the result has zero mean and identity covariance. Given a
    known covariance, only the mean is fit. Computes in float64.

    Attributes:
        covariance_matrix: The known covariance, or None to fit it.
        mean: The fitted mean, shaped (dims,).
        scale: The lower-triangular ``L``, shaped (dims, dims).
    """

    def __init__(self, covariance_matrix: Optional[ch.Tensor] = None):
        """Initialize an unfit standardization.

        Args:
            covariance_matrix: Known covariance, shaped (dims, dims); fit
                from the data when None.
        """
        self.covariance_matrix = covariance_matrix
        self.mean: Optional[ch.Tensor] = None
        self.scale: Optional[ch.Tensor] = None

    def fit(self, x: ch.Tensor) -> "Standardization":
        """Fit the mean, and the covariance unless it is known.

        Args:
            x: Data of shape (n, dims).

        Returns:
            self.
        """
        x = x.to(ch.float64)
        dims = x.size(1)
        covariance_matrix = (
            ch.cov(x.T).reshape(dims, dims)
            if self.covariance_matrix is None
            else self.covariance_matrix.to(ch.float64)
        )
        self.mean = x.mean(0)
        self.scale = ch.linalg.cholesky(covariance_matrix)  # pylint: disable=not-callable
        return self

    def transform(self, x: ch.Tensor) -> ch.Tensor:
        """Return ``L^-1 (x - mean)`` row by row, in float64."""
        mean, scale = self._fitted()
        centered = (x.to(ch.float64) - mean).T
        return ch.linalg.solve_triangular(scale, centered, upper=False).T  # pylint: disable=not-callable

    def inverse_transform(self, z: ch.Tensor) -> ch.Tensor:
        """Return ``L z + mean`` row by row."""
        mean, scale = self._fitted()
        return z.to(ch.float64) @ scale.T + mean

    def inverse_transform_gaussian(
        self, loc: ch.Tensor, covariance_matrix: ch.Tensor
    ) -> tuple[ch.Tensor, ch.Tensor]:
        """Map a Gaussian's parameters back to the original coordinates.

        Args:
            loc: Mean in standardized coordinates.
            covariance_matrix: Covariance in standardized coordinates.

        Returns:
            ``(L loc + mean, L covariance_matrix L^T)``.
        """
        mean, scale = self._fitted()
        return scale @ loc + mean, scale @ covariance_matrix @ scale.T

    def _fitted(self) -> tuple[ch.Tensor, ch.Tensor]:
        """Return the mean and scale.

        Raises:
            RuntimeError: If called before fit.
        """
        if self.mean is None or self.scale is None:
            raise RuntimeError("the standardization has not been fit.")
        return self.mean, self.scale
