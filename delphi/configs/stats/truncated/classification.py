# Author: pstefanou12@
"""Configuration models for the truncated classification estimators."""

from typing import Optional

import pydantic

from delphi.configs.stats.truncated.linear import TruncatedLinearModelConfig


class TruncatedClassificationConfig(TruncatedLinearModelConfig):
    """Fields shared by the truncated classification estimators.

    Logistic and probit regression differ only in their link function, so they
    are siblings here rather than one deriving from the other, and each
    overrides only the defaults it needs.

    Attributes:
        l1: L1 regularization weight.
        r: Initial projection set radius.
        rate: Multiplicative radius expansion factor.
    """

    l1: float = pydantic.Field(default=0.0, ge=0.0)
    r: float = pydantic.Field(default=1.0, gt=0.0)
    rate: float = pydantic.Field(default=1.5, gt=0.0)
    batch_size: int = 10


class TruncatedLogisticRegressionConfig(TruncatedClassificationConfig):
    """Configuration for truncated logistic regression."""

    epochs: Optional[int] = pydantic.Field(default=1, ge=1)


class TruncatedProbitRegressionConfig(TruncatedClassificationConfig):
    """Configuration for truncated probit regression.

    Attributes:
        tol: Convergence tolerance.
    """

    tol: float = pydantic.Field(default=1e-3, ge=0.0)
