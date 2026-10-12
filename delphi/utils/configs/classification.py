# Author: pstefanou12@
"""Configuration models for the truncated classification estimators."""

from typing import Optional

import pydantic

from delphi.utils.configs.base import LinearModelConfig


class TruncatedLogisticRegressionConfig(LinearModelConfig):
    """Configuration for truncated logistic regression (TRUNC_LOG_REG_DEFAULTS).

    Attributes:
        l1: L1 regularization weight.
        eps: Numerical stability constant for the criterion.
        r: Initial projection set radius.
        rate: Multiplicative radius expansion factor.
        num_samples: Monte Carlo samples drawn per criterion evaluation.
    """

    l1: float = pydantic.Field(default=0.0, ge=0.0)
    eps: float = pydantic.Field(default=1e-5, gt=0.0)
    r: float = pydantic.Field(default=1.0, gt=0.0)
    rate: float = pydantic.Field(default=1.5, gt=0.0)
    num_samples: int = pydantic.Field(default=1000, ge=1)

    # Override parent defaults for logistic regression.
    epochs: Optional[int] = pydantic.Field(default=1, ge=1)
    batch_size: int = pydantic.Field(default=10, ge=1)


class TruncatedProbitRegressionConfig(TruncatedLogisticRegressionConfig):
    """Configuration for truncated probit regression (TRUNC_PROB_REG_DEFAULTS).

    Attributes:
        tol: Convergence tolerance.
    """

    tol: float = pydantic.Field(default=1e-3, ge=0.0)

    # The probit defaults did not override epochs.
    epochs: Optional[int] = pydantic.Field(default=None, ge=1)


class SoftmaxRegressionConfig(LinearModelConfig):
    """Configuration for truncated softmax (multinomial) regression.

    Attributes:
        num_samples: Monte Carlo samples drawn per criterion evaluation.
        multi_class: Multiclass strategy used for the empirical initialization.
    """

    num_samples: int = pydantic.Field(default=1000, ge=1)
    multi_class: str = "ovr"
