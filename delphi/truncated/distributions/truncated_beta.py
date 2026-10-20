# Author: pstefanou12@
"""Truncated Beta distribution."""

from collections.abc import Callable
from typing import Optional

import torch as ch
from torch import nn

from delphi import configs
from delphi.exponential_family import beta
from delphi.truncated.distributions import truncated_exponential_family_distributions

# Newton's method on the Beta likelihood converges quadratically once close;
# this is far more than the moment start needs.
NEWTON_STEPS = 50


def _newton_step(first, second, log_x, log_1mx):
    """Take one damped Newton step towards the Beta MLE, per coordinate.

    Args:
        first: Current ``a``, one per coordinate.
        second: Current ``b``, one per coordinate.
        log_x: Mean of ``log x``.
        log_1mx: Mean of ``log(1 - x)``.

    Returns:
        The updated ``(a, b)``, kept positive.
    """
    total = ch.polygamma(1, first + second)
    gradient_a = ch.digamma(first) - ch.digamma(first + second) - log_x
    gradient_b = ch.digamma(second) - ch.digamma(first + second) - log_1mx
    h_aa, h_bb, h_ab = (
        ch.polygamma(1, first) - total,
        ch.polygamma(1, second) - total,
        -total,
    )
    det = h_aa * h_bb - h_ab * h_ab
    step_a = (h_bb * gradient_a - h_ab * gradient_b) / det
    step_b = (h_aa * gradient_b - h_ab * gradient_a) / det
    # Halve a step that would leave the positive orthant.
    scale = ch.ones_like(first)
    while bool(
        ((first - scale * step_a) <= 0).any() or ((second - scale * step_b) <= 0).any()
    ):
        scale = ch.where(
            (first - scale * step_a <= 0) | (second - scale * step_b <= 0),
            scale / 2,
            scale,
        )
    return first - scale * step_a, second - scale * step_b


def untruncated_mle(samples: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
    """Return the Beta MLE of samples, ignoring any truncation.

    Solved by Newton's method from the method of moments, since the MLE has
    no closed form.

    Args:
        samples: Observations of shape (n, dims), inside (0, 1).

    Returns:
        ``(a, b)``, one entry per coordinate.
    """
    dims = samples.size(-1)
    stat = beta.Beta.calc_suff_stat(samples).mean(0)
    log_x, log_1mx = stat[:dims], stat[dims:]
    mean, var = samples.mean(0), samples.var(0)
    common = mean * (1 - mean) / var - 1
    first = (mean * common).clamp(min=1e-3)
    second = ((1 - mean) * common).clamp(min=1e-3)
    for _ in range(NEWTON_STEPS):
        first, second = _newton_step(first, second, log_x, log_1mx)
    return first, second


class TruncatedBeta(
    truncated_exponential_family_distributions.TruncatedExponentialFamilyDistribution
):
    """Model for truncated Beta distributions."""

    dist = beta.Beta
    defaults = configs.TruncatedExponentialFamilyDistributionConfig
    estimate_suffixes = ("a_", "b_")
    # Each coordinate's a and b are coupled, so the Hessian is block-diagonal
    # and its spectrum needs eigenvalues rather than the diagonal.
    independent_coordinates = False

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
    ):
        """Initialize TruncatedBeta.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            alpha: Survival probability lower bound.
        """
        super().__init__(hyperparameters, phi, dims, alpha)

    def _calc_emp_model(self):
        """Initialize theta at the untruncated MLE, the point every family starts from."""
        first, second = untruncated_mle(self.train_loader_.dataset.S)
        self.emp_theta = beta.Beta.to_natural(ch.cat([first, second]))
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))

    def _constraints(self, theta):
        """Clamp theta above -1, so both shape parameters are positive."""
        return ch.clamp(theta, min=-1.0 + 1e-6)

    @property
    def best_a_(self):
        """Return the best first shape estimate."""
        return beta.Beta.to_canonical(self.best_params)[: self.dims]

    @property
    def final_a_(self):
        """Return the final first shape estimate."""
        return beta.Beta.to_canonical(self.final_params)[: self.dims]

    @property
    def ema_a_(self):
        """Return the EMA first shape estimate."""
        return beta.Beta.to_canonical(self.ema_params)[: self.dims]

    @property
    def avg_a_(self):
        """Return the averaged first shape estimate."""
        return beta.Beta.to_canonical(self.avg_params)[: self.dims]

    @property
    def best_b_(self):
        """Return the best second shape estimate."""
        return beta.Beta.to_canonical(self.best_params)[self.dims :]

    @property
    def final_b_(self):
        """Return the final second shape estimate."""
        return beta.Beta.to_canonical(self.final_params)[self.dims :]

    @property
    def ema_b_(self):
        """Return the EMA second shape estimate."""
        return beta.Beta.to_canonical(self.ema_params)[self.dims :]

    @property
    def avg_b_(self):
        """Return the averaged second shape estimate."""
        return beta.Beta.to_canonical(self.avg_params)[self.dims :]

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated beta distribution"
