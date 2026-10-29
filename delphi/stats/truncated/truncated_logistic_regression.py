# Author: pstefanou12@
"""Truncated logistic regression: logistic latent noise."""

from delphi import configs
from delphi.stats.truncated import losses
from delphi.stats.truncated.truncated_binary_regression import (
    TruncatedBinaryRegression,
)
from delphi.truncated import sampling


class TruncatedLogisticRegression(
    TruncatedBinaryRegression[configs.TruncatedLogisticRegressionConfig]
):
    """Truncated logistic regression, ``y = 1{w^T x + b + e >= 0}``.

    ``e ~ Logistic(0, 1)``, observed only where ``phi`` admits the latent;
    see ``TruncatedBinaryRegression``.
    """

    defaults = configs.TruncatedLogisticRegressionConfig
    noise = sampling.LOGISTIC_NOISE
    loss = losses.TruncatedBCE

    def __str__(self):
        """Return a human-readable name for this model."""
        return "truncated logistic regression"
