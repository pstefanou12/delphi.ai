# Author: pstefanou12@
"""Truncated Wishart distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
import torch.nn

from delphi.configs.distributions.truncated import gamma_unknown_concentration
from delphi.exponential_family import wishart
from delphi.truncated.distributions import truncated_exponential_family_distributions
from delphi.utils import helpers


def rough_start(samples: ch.Tensor, size: int) -> ch.Tensor:
    """Return a Wishart's canonical ``[df, vech(scale)]`` read off its diagonal.

    Each ``X_ii`` is Gamma with shape ``df / 2``, so Minka's (2002)
    approximation to the Gamma shape MLE gives a df per diagonal entry; they
    are averaged, and ``scale = mean(X) / df``. The moment estimate
    ``2 E[X_ii]^2 / Var[X_ii]`` is not used: a tail cut shrinks the variance,
    which sent it to 6e11.

    Args:
        samples: Wishart samples as vech rows, shaped (n, vech_size(size)).
        size: Side of the sampled matrices.

    Returns:
        The canonical parameters, shaped (1 + vech_size(size),).
    """
    mean = samples.mean(0)
    rows, cols = ch.tril_indices(size, size)
    diagonal = samples[:, rows == cols]
    spread = diagonal.mean(0).log() - diagonal.log().mean(0)
    shape = (3.0 - spread + ch.sqrt((spread - 3.0) ** 2 + 24.0 * spread)) / (
        12.0 * spread
    )
    df = (2.0 * shape).mean().clamp(min=size + 1e-3)
    return ch.cat([df.view(1), mean / df])


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
    defaults = gamma_unknown_concentration.TruncatedGammaUnknownConcentrationConfig
    estimate_suffixes = ("df_", "covariance_matrix_")
    # Every statistic shares the log determinant, so the Hessian is dense.
    independent_coordinates = False

    def __init__(
        self,
        phi: Optional[Callable],
        dims: int,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize TruncatedWishart.

        Args:
            phi: Truncation set oracle, over vech rows; None learns it from
                the sample.
            dims: Side of the sampled matrices.
            alpha: Survival probability lower bound.
            hyperparameters: Hyperparameter dict.
        """
        super().__init__(
            phi, helpers.vech_size(dims), alpha, hyperparameters=hyperparameters
        )
        self.size = dims

    def _calc_emp_model(self):
        """Start from ``rough_start``, refined by the warm start."""
        self.emp_theta = wishart.Wishart.to_natural(
            rough_start(self.train_loader_.dataset.S, self.size)
        )
        self.register_parameter("theta", torch.nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Keep df above ``size - 1`` and the matrix part negative definite."""
        values, vectors = ch.linalg.eigh(helpers.unvech(theta[1:], self.size))  # pylint: disable=not-callable
        matrix = vectors @ ch.diag_embed(values.clamp(max=-1e-6)) @ vectors.mT
        return ch.cat([theta[:1].clamp(min=-1.0 + 1e-6), helpers.vech(matrix)])

    def _canonical(self, theta: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
        """Split theta's canonical form into df and the scale matrix."""
        canonical = wishart.Wishart.to_canonical(theta)
        return canonical[0], helpers.unvech(canonical[1:], self.size)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated wishart distribution"
