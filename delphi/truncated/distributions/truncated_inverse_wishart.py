# Author: pstefanou12@
"""Truncated inverse Wishart distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import inverse_wishart
from delphi.truncated.distributions import (
    truncated_exponential_family_distributions,
    truncated_wishart,
)
from delphi.utils import helpers


class TruncatedInverseWishart(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated inverse Wishart distributions, df and scale unknown.

    Samples are rows of ``vech(X)``, so ``dims`` here is the matrix side while
    the base class's ``dims`` is the row width ``size * (size + 1) / 2``.

    Attributes:
        size: Side of the sampled matrices.
    """

    dist = inverse_wishart.InverseWishart
    # The Wishart's statistic of X^{-1}, so it couples as the Wishart's does.
    defaults = configs.TruncatedGammaUnknownConcentrationConfig
    estimate_suffixes = ("df_", "scale_")
    # Every statistic shares the log determinant, so the Hessian is dense.
    independent_coordinates = False

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedInverseWishart.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle, over vech rows.
            dims: Side of the sampled matrices.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, helpers.vech_size(dims), alpha)
        self.size = dims

    def _calc_emp_model(self):
        """Start at the Wishart's rough start on ``X^{-1}``, then warm start."""
        dataset_s = self.train_loader_.dataset.S
        inverses = helpers.vech(ch.linalg.inv(helpers.unvech(dataset_s, self.size)))  # pylint: disable=not-callable
        canonical = truncated_wishart.rough_start(inverses, self.size)
        scale = ch.linalg.inv(helpers.unvech(canonical[1:], self.size))  # pylint: disable=not-callable
        self.emp_theta = inverse_wishart.InverseWishart.to_natural(
            ch.cat([canonical[:1], helpers.vech(scale)])
        )
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Keep df above ``size - 1`` and the matrix part negative definite."""
        values, vectors = ch.linalg.eigh(helpers.unvech(theta[1:], self.size))  # pylint: disable=not-callable
        matrix = vectors @ ch.diag_embed(values.clamp(max=-1e-6)) @ vectors.mT
        return ch.cat([theta[:1].clamp(min=self.size + 1e-6), helpers.vech(matrix)])

    def _canonical(self, theta: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
        """Split theta's canonical form into df and the scale matrix."""
        canonical = inverse_wishart.InverseWishart.to_canonical(theta)
        return canonical[0], helpers.unvech(canonical[1:], self.size)

    @property
    def best_df_(self):
        """Return the best degrees-of-freedom estimate."""
        return self._canonical(self.best_params)[0]

    @property
    def final_df_(self):
        """Return the final degrees-of-freedom estimate."""
        return self._canonical(self.final_params)[0]

    @property
    def ema_df_(self):
        """Return the EMA degrees-of-freedom estimate."""
        return self._canonical(self.ema_params)[0]

    @property
    def avg_df_(self):
        """Return the averaged degrees-of-freedom estimate."""
        return self._canonical(self.avg_params)[0]

    @property
    def best_scale_(self):
        """Return the best scale-matrix estimate."""
        return self._canonical(self.best_params)[1]

    @property
    def final_scale_(self):
        """Return the final scale-matrix estimate."""
        return self._canonical(self.final_params)[1]

    @property
    def ema_scale_(self):
        """Return the EMA scale-matrix estimate."""
        return self._canonical(self.ema_params)[1]

    @property
    def avg_scale_(self):
        """Return the averaged scale-matrix estimate."""
        return self._canonical(self.avg_params)[1]

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated inverse wishart distribution"
