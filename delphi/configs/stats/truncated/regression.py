# Author: pstefanou12@
"""Configuration models for the truncated regression estimators."""

from typing import Literal, Optional

import pydantic

from delphi.configs.stats.truncated.linear import TruncatedLinearModelConfig


class TruncatedRegressionConfig(TruncatedLinearModelConfig):
    """Configuration for truncated linear regression with known variance.

    The descent follows Daskalakis et al. (2020), Algorithm 1: one shuffled
    pass, one sample per step, steps of 1 / (lambda i) from the Fisher
    schedule, every sample used, and every iterate averaged. Newton steps on
    the full sample then take that average to the empirical MLE.

    Attributes:
        weight_bound: The B of D_{r,B}, on the weight in the normalized
            coordinates the descent runs in; None drops the ball.
        refine_steps: Newton steps after the descent; their iterates are
            the estimates, or the descent's when 0.
    """

    batch_size: int = 10
    epochs: Optional[int] = pydantic.Field(default=1, ge=1)
    val: float = pydantic.Field(default=0.0, ge=0.0, le=1.0)
    scheduler: Optional[str] = "fisher"
    fisher_refresh_every: int = pydantic.Field(
        default=0, ge=0, serialization_alias="refresh_every"
    )
    avg_burn_in: float = pydantic.Field(default=0.0, ge=0.0, lt=1.0)
    weight_bound: Optional[float] = pydantic.Field(default=None, gt=0.0)
    refine_steps: int = pydantic.Field(default=30, ge=0)

    @pydantic.model_validator(mode="after")
    def drop_default_schedule_off_sgd(self) -> "TruncatedRegressionConfig":
        """Leave a non-SGD optimizer unscheduled unless a schedule is set.

        The 1 / (lambda i) schedule is SGD's; L-BFGS and Adam scale their
        own steps, and decaying them as well stalls the descent.
        """
        if self.optimizer.lower() != "sgd" and "scheduler" not in (
            self.model_fields_set
        ):
            object.__setattr__(self, "scheduler", None)
        return self


class TruncatedUnknownVarianceRegressionConfig(TruncatedRegressionConfig):
    """Configuration for truncated regression with the variance learned.

    The defaults are the ones this estimator had before the known-variance
    rework: projected SGD in (v, lambda) with a held-out split, no step
    schedule and no Newton refinement.

    Attributes:
        var_lr: Learning rate for lambda; None trains it at lr.
        r: Scale of the variance upper bound above the start's residual
            variance.
    """

    batch_size: int = 50
    epochs: Optional[int] = pydantic.Field(default=None, ge=1)
    iterations: Optional[int] = pydantic.Field(default=1500, ge=1)
    val: float = pydantic.Field(default=0.2, ge=0.0, le=1.0)
    val_interval: Optional[int] = pydantic.Field(default=50, ge=1)
    scheduler: Optional[str] = None
    avg_burn_in: float = pydantic.Field(default=0.5, ge=0.0, lt=1.0)
    refine_steps: int = pydantic.Field(default=0, ge=0)
    projection_set: Literal["default", "none"] = "none"
    var_lr: Optional[float] = pydantic.Field(default=1e-2, gt=0.0)
    r: float = pydantic.Field(default=1.0, gt=0.0)


class TruncatedLinearDynamicalSystemConfig(TruncatedRegressionConfig):
    """Configuration for truncated linear dynamical systems (Plevrakis 2021).

    The defaults are the ones this estimator had before the known-variance
    rework: data consumed in order, a constant step derived from alpha, and
    no projection or refinement.

    Attributes:
        c_s: Scaling constant for the dependent-data survival threshold.
        c_eta: Exponent used to derive the learning rate from alpha.
        c_gamma: Scaling constant for the gamma bound.
    """

    c_s: float = pydantic.Field(default=10.0, gt=0.0)
    c_eta: float = pydantic.Field(default=0.5, gt=0.0)
    c_gamma: float = pydantic.Field(default=2.0, gt=0.0)

    num_samples: int = pydantic.Field(default=50, ge=1)
    batch_size: int = 50
    epochs: Optional[int] = pydantic.Field(default=None, ge=1)
    val: float = pydantic.Field(default=0.2, ge=0.0, le=1.0)
    shuffle: bool = False
    constant: bool = True
    scheduler: Optional[str] = None
    avg_burn_in: float = pydantic.Field(default=0.5, ge=0.0, lt=1.0)
    refine_steps: int = pydantic.Field(default=0, ge=0)
    projection_set: Literal["default", "none"] = "none"


class TruncatedLassoRegressionConfig(TruncatedLinearModelConfig):
    """Configuration for truncated LASSO regression (TRUNC_LASSO_DEFAULTS).

    Attributes:
        r: Initial projection set radius.
    """

    iterations: Optional[int] = pydantic.Field(default=1500, ge=1)
    val_interval: Optional[int] = pydantic.Field(default=50, ge=1)
    num_samples: int = pydantic.Field(default=10000, ge=1)
    r: float = pydantic.Field(default=1.0, gt=0.0)
