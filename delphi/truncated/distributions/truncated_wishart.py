# Author: pstefanou12@
"""Truncated Wishart distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import wishart
from delphi.truncated.distributions import truncated_exponential_family_distributions
from delphi.utils import helpers


class TruncatedWishart(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Wishart distributions with unknown df and scale.

    Samples are rows of ``vech(X)``, the lower triangle of each matrix, so
    ``dims`` here is the matrix side while the base class's ``dims`` is the
    row width ``size * (size + 1) / 2``.

    Attributes:
        size: Side of the sampled matrices.
    """

    dist = wishart.Wishart
    # A one-by-one Wishart is the two-parameter Gamma, whose log x and x couple
    # as log det X and X do.
    defaults = configs.TruncatedGammaUnknownConcentrationConfig
    estimate_suffixes = ("df_", "covariance_matrix_")
    # Every statistic shares the log determinant, so the Hessian is dense.
    independent_coordinates = False

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedWishart.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle, over vech rows.
            dims: Side of the sampled matrices.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, helpers.vech_size(dims), alpha)
        self.size = dims

    def _calc_emp_model(self):
        """Start from the diagonal, refined by the warm start.

        Each ``X_ii`` is Gamma with shape ``df / 2``, so Minka's (2002)
        approximation to the Gamma shape MLE gives a df per diagonal entry;
        they are averaged, and ``scale = mean(X) / df``. The moment estimate
        ``2 E[X_ii]^2 / Var[X_ii]`` is not used: a tail cut shrinks the
        variance, which sent it to 6e11.
        """
        dataset_s = self.train_loader_.dataset.S
        mean = dataset_s.mean(0)
        rows, cols = ch.tril_indices(self.size, self.size)
        diagonal = dataset_s[:, rows == cols]
        spread = diagonal.mean(0).log() - diagonal.log().mean(0)
        shape = (3.0 - spread + ch.sqrt((spread - 3.0) ** 2 + 24.0 * spread)) / (
            12.0 * spread
        )
        df = (2.0 * shape).mean().clamp(min=self.size + 1e-3)
        self.emp_theta = wishart.Wishart.to_natural(ch.cat([df.view(1), mean / df]))
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Keep df above ``size - 1`` and the matrix part negative definite."""
        values, vectors = ch.linalg.eigh(helpers.unvech(theta[1:], self.size))  # pylint: disable=not-callable
        matrix = vectors @ ch.diag_embed(values.clamp(max=-1e-6)) @ vectors.mT
        return ch.cat([theta[:1].clamp(min=-1.0 + 1e-6), helpers.vech(matrix)])

    def _canonical(self, theta: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
        """Split theta's canonical form into df and the scale matrix."""
        canonical = wishart.Wishart.to_canonical(theta)
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
    def best_covariance_matrix_(self):
        """Return the best scale-matrix estimate."""
        return self._canonical(self.best_params)[1]

    @property
    def final_covariance_matrix_(self):
        """Return the final scale-matrix estimate."""
        return self._canonical(self.final_params)[1]

    @property
    def ema_covariance_matrix_(self):
        """Return the EMA scale-matrix estimate."""
        return self._canonical(self.ema_params)[1]

    @property
    def avg_covariance_matrix_(self):
        """Return the averaged scale-matrix estimate."""
        return self._canonical(self.avg_params)[1]

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated wishart distribution"
