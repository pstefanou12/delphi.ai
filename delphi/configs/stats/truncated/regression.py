# Author: pstefanou12@
"""Configuration models for the truncated regression estimators."""

from typing import Optional

import pydantic

from delphi.configs.stats.truncated.linear import TruncatedLinearModelConfig


class TruncatedRegressionFieldsConfig(TruncatedLinearModelConfig):
    """Fields shared by the truncated regression and dynamical-system variants.

    ``TruncatedLinearRegression`` reads from both variants depending on its
    ``dependent`` flag, so every field either one needs is declared here and
    the concrete subclasses override only default values. That keeps a single
    type parameter sufficient for the estimator.

    Attributes:
        var_lr: Learning rate for the variance parameter when noise_var is
            unknown.
        r: Initial projection set radius.
        rate: Multiplicative radius expansion factor.
        l1: L1 regularization weight.
    """

    # None disables the separate variance learning rate; parameters_() in
    # TruncatedLinearRegression branches on that explicitly.
    var_lr: Optional[float] = pydantic.Field(default=1e-2, gt=0.0)
    r: float = pydantic.Field(default=1.0, gt=0.0)
    rate: float = pydantic.Field(default=1.5, gt=0.0)
    l1: float = pydantic.Field(default=0.0, ge=0.0)


class TruncatedRegressionConfig(TruncatedRegressionFieldsConfig):
    """Configuration for truncated linear regression (TRUNC_REG_DEFAULTS)."""

    iterations: Optional[int] = pydantic.Field(default=1500, ge=1)
    val_interval: Optional[int] = pydantic.Field(default=50, ge=1)


class TruncatedLassoRegressionConfig(TruncatedRegressionConfig):
    """Configuration for truncated LASSO regression (TRUNC_LASSO_DEFAULTS)."""

    num_samples: int = pydantic.Field(default=10000, ge=1)


class TruncatedLinearDynamicalSystemConfig(TruncatedRegressionFieldsConfig):
    """Configuration for truncated linear dynamical systems (TRUNC_LDS_DEFAULTS).

    c_s, c_eta and c_gamma live here rather than on a shared parent: each is
    read only from TruncatedLinearRegression's dependent branches, and
    TRUNC_LDS_DEFAULTS was the only pre-pydantic spec that declared them.

    Attributes:
        c_s: Scaling constant for the dependent-data survival threshold.
        c_eta: Exponent used to derive the learning rate from alpha.
        c_gamma: Scaling constant for the gamma bound.
    """

    c_s: float = pydantic.Field(default=10.0, gt=0.0)
    c_eta: float = pydantic.Field(default=0.5, gt=0.0)
    c_gamma: float = pydantic.Field(default=2.0, gt=0.0)

    num_samples: int = pydantic.Field(default=50, ge=1)
    shuffle: bool = False  # Dependent data is consumed in order.
    constant: bool = True


class TruncatedLinearRegressionConfig(TruncatedLinearModelConfig):
    """Configuration for truncated linear regression with known noise variance.

    The defaults follow Daskalakis et al. (2020), Algorithm 1: one shuffled
    pass, one sample per step, steps of 1 / (lambda i) from the Fisher
    schedule, every sample used for the descent, and the average of every
    iterate as the estimate.

    Attributes:
        r: The r of the projection set D_{r,B}; None uses the paper's
            r* = 4 log(2 / alpha) + 7.
        weight_bound: The B of D_{r,B}, on the weight in the normalized
            coordinates the descent runs in; None drops the ball.
    """

    batch_size: int = 1
    epochs: Optional[int] = pydantic.Field(default=1, ge=1)
    val: float = pydantic.Field(default=0.0, ge=0.0, le=1.0)
    scheduler: Optional[str] = "fisher"
    fisher_refresh_every: int = pydantic.Field(
        default=0, ge=0, serialization_alias="refresh_every"
    )
    avg_burn_in: float = pydantic.Field(default=0.0, ge=0.0, lt=1.0)
    r: Optional[float] = pydantic.Field(default=None, gt=0.0)
    weight_bound: Optional[float] = pydantic.Field(default=None, gt=0.0)

    @pydantic.model_validator(mode="after")
    def drop_default_schedule_off_sgd(self) -> "TruncatedLinearRegressionConfig":
        """Leave a non-SGD optimizer unscheduled unless a schedule is set.

        The 1 / (lambda i) schedule is SGD's; L-BFGS and Adam scale their
        own steps, and decaying them as well stalls the descent.
        """
        if self.optimizer.lower() != "sgd" and "scheduler" not in (
            self.model_fields_set
        ):
            object.__setattr__(self, "scheduler", None)
        return self
