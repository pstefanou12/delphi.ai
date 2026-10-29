# Author: pstefanou12@
"""Configuration models for the truncated classification estimators."""

from typing import Optional

import pydantic

from delphi.configs.stats.truncated.linear import TruncatedLinearModelConfig


class TruncatedClassificationConfig(TruncatedLinearModelConfig):
    """Configuration shared by the truncated classifiers.

    The descent is projected SGD with steps of 1 / (lambda i) from the
    Fisher schedule and every iterate averaged; Fisher scoring steps on the
    full sample then take that average to the empirical MLE.

    Attributes:
        num_samples: Draws per sample and label estimating the loss.
        radius: ``log(1 / alpha')`` for the smallest per-label mass alpha'
            R_alpha' admits; None takes alpha' = alpha^2, or starts from
            the descent's start when alpha is unknown.
        refine_steps: Fisher scoring steps after the descent, as
            ``configs.RefinementConfig``.
    """

    batch_size: int = 10
    epochs: Optional[int] = pydantic.Field(default=1, ge=1)
    val: float = pydantic.Field(default=0.0, ge=0.0, le=1.0)
    scheduler: Optional[str] = "fisher"
    avg_burn_in: float = pydantic.Field(default=0.0, ge=0.0, lt=1.0)
    num_samples: int = pydantic.Field(default=200, ge=1)
    radius: Optional[float] = pydantic.Field(default=None, gt=0.0)
    refine_steps: int = pydantic.Field(default=10, ge=0)

    @pydantic.model_validator(mode="after")
    def drop_default_schedule_off_sgd(self) -> "TruncatedClassificationConfig":
        """Leave a non-SGD optimizer unscheduled unless a schedule is set.

        The 1 / (lambda i) schedule is SGD's; L-BFGS and Adam scale their
        own steps, and decaying them as well stalls the descent.
        """
        if self.optimizer.lower() != "sgd" and "scheduler" not in (
            self.model_fields_set
        ):
            object.__setattr__(self, "scheduler", None)
        return self


class TruncatedBinaryRegressionConfig(TruncatedClassificationConfig):
    """Configuration shared by truncated probit and logistic regression.

    Attributes:
        interval_grid: Latent means the set's interval is searched over.
    """

    interval_grid: int = pydantic.Field(default=2001, ge=2)


class TruncatedLogisticRegressionConfig(TruncatedBinaryRegressionConfig):
    """Configuration for truncated logistic regression."""


class TruncatedProbitRegressionConfig(TruncatedBinaryRegressionConfig):
    """Configuration for truncated probit regression."""


class TruncatedMultinomialLogisticRegressionConfig(TruncatedClassificationConfig):
    """Configuration for truncated multinomial logistic regression.

    Attributes:
        iterations: Gradient steps of the descent.
    """

    epochs: Optional[int] = pydantic.Field(default=None, ge=1)
    iterations: Optional[int] = pydantic.Field(default=1000, ge=1)
