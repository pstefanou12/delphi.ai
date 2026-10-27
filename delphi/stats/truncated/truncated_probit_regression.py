# Author: pstefanou12@
"""Truncated probit regression: Gaussian latent noise."""

from delphi import configs
from delphi.stats.truncated import losses
from delphi.stats.truncated.truncated_binary_regression import (
    TruncatedBinaryRegression,
)
from delphi.truncated import sampling


class TruncatedProbitRegression(
    TruncatedBinaryRegression[configs.TruncatedProbitRegressionConfig]
):
    """Truncated probit regression, ``y = 1{w^T x + b + e >= 0}``, ``e ~ N(0, 1)``.

    Observed only where ``phi`` admits the latent; see
    ``TruncatedBinaryRegression``.
    """

    defaults = configs.TruncatedProbitRegressionConfig
    noise = sampling.GAUSSIAN_NOISE
    loss = losses.TruncatedProbitNLL

    def __str__(self):
        """Return a human-readable name for this model."""
        return "truncated probit regression"
