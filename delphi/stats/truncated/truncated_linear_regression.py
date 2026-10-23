# Author: pstefanou12@
"""Truncated linear regression with known noise variance.

Daskalakis, Gouleakis, Tzamos and Zampetakis, "Computationally and
Statistically Efficient Truncated Regression" (COLT 2019): projected SGD
without replacement on the truncated NLL, kept in the set D_{r,B} of their
Definition 2, returning the averaged iterate.
"""

from collections.abc import Callable
import math
from typing import Optional, Union

import torch as ch
import torch.distributions

from delphi import configs
from delphi.stats import linear_model
from delphi.stats.truncated import projection_sets
from delphi.stats.truncated.truncated_linear_model import TruncatedLinearModel


class TruncatedLinearRegression(
    TruncatedLinearModel[configs.TruncatedLinearRegressionConfig]
):
    """Truncated linear regression with known noise variance.

    ``y = w^T x + b + e`` with ``e ~ N(0, noise_var)``, observed only where
    ``phi(y)``. The descent starts at OLS (or zero) projected onto D_{r,B},
    which OLS anchors.

    Attributes:
        noise_var: The known noise variance.
    """

    defaults = configs.TruncatedLinearRegressionConfig

    def __init__(  # pylint: disable=too-many-arguments
        self,
        hyperparameters: dict,
        phi: Callable,
        alpha: float,
        *,
        noise_var: Union[float, ch.Tensor],
        fit_intercept: bool = True,
        rand_seed: int = 0,
    ):
        """Initialize TruncatedLinearRegression.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation oracle over responses shaped (n, m, 1).
            alpha: Survival probability in (0, 1].
            noise_var: The known noise variance, a positive scalar.
            fit_intercept: Whether to fit an intercept.
            rand_seed: Seed applied at the start of every run.

        Raises:
            ValueError: If noise_var is not a positive scalar.
        """
        super().__init__(hyperparameters, phi, alpha, fit_intercept, rand_seed)
        noise_var = float(ch.as_tensor(noise_var))
        if noise_var <= 0.0:
            raise ValueError(f"noise_var must be positive; got {noise_var}.")
        self.noise_var = noise_var

    @property
    def noise(self) -> torch.distributions.Distribution:
        """Gaussian noise with the known variance."""
        return torch.distributions.Normal(
            ch.zeros(1), ch.full((1,), math.sqrt(self.noise_var))
        )

    def untruncated_fit(self, design: ch.Tensor, targ: ch.Tensor) -> ch.Tensor:
        """Return the OLS weight."""
        return ch.linalg.lstsq(design, targ).solution  # pylint: disable=not-callable

    def build_projection_set(
        self, design: ch.Tensor, targ: ch.Tensor
    ) -> projection_sets.ProjectionSet:
        """Return D_{r,B}, anchored at OLS and widened to contain it.

        Raises:
            ValueError: If OLS lies outside the ball of radius weight_bound.
        """
        hp = self.hyperparameters
        r = (
            projection_sets.ResidualSpectralSet.paper_radius(self.alpha)
            if hp.r is None
            else hp.r
        )
        ols = self.required_untruncated_weight
        if hp.weight_bound is not None and float(ols.norm()) > hp.weight_bound:
            raise ValueError(
                f"OLS has norm {float(ols.norm()):.4f} in the normalized "
                f"coordinates, outside weight_bound {hp.weight_bound}."
            )
        projection = projection_sets.ResidualSpectralSet(
            design, targ, self.noise_var, r, ols, hp.weight_bound
        )
        at_ols = projection.spectral_radius(ols)
        if at_ols > r:
            self.logger.warning(
                f"OLS lies outside D_{{r,B}} (lambda_max {at_ols:.4f} > r = "
                f"{r:.4f}); widening r to contain it."
            )
            projection.r = at_ols
        return projection

    @property
    def ols_coef_(self) -> ch.Tensor:
        """Coefficients of the OLS fit to the truncated sample."""
        return linear_model.require_fitted(
            self.coefficients(self.required_untruncated_weight.flatten())[0],
            "ols_coef_",
        )

    @property
    def ols_intercept_(self) -> Optional[ch.Tensor]:
        """Intercept of the OLS fit to the truncated sample."""
        return self.coefficients(self.required_untruncated_weight.flatten())[1]

    def __str__(self):
        """Return a human-readable name for this model."""
        return "truncated linear regression, known variance"
