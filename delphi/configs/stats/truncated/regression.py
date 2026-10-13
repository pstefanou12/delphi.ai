# Author: pstefanou12@
"""Configuration models for the truncated regression estimators."""

from typing import Optional

import pydantic
import torch as ch

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
        eps: Numerical stability constant for the truncated MSE criterion.
        r: Initial projection set radius.
        rate: Multiplicative radius expansion factor.
        num_samples: Monte Carlo samples drawn per criterion evaluation.
        l1: L1 regularization weight.
    """

    # None disables the separate variance learning rate; parameters_() in
    # TruncatedLinearRegression branches on that explicitly.
    var_lr: Optional[float] = pydantic.Field(default=1e-2, gt=0.0)
    eps: float = pydantic.Field(default=1e-5, gt=0.0)
    r: float = pydantic.Field(default=1.0, gt=0.0)
    rate: float = pydantic.Field(default=1.5, gt=0.0)
    num_samples: int = pydantic.Field(default=1000, ge=1)
    l1: float = pydantic.Field(default=0.0, ge=0.0)


class TruncatedRegressionConfig(TruncatedRegressionFieldsConfig):
    """Configuration for truncated linear regression (TRUNC_REG_DEFAULTS)."""

    iterations: Optional[int] = pydantic.Field(default=1500, ge=1)
    val_interval: Optional[int] = pydantic.Field(default=50, ge=1)


class TruncatedLassoRegressionConfig(TruncatedRegressionConfig):
    """Configuration for truncated LASSO regression (TRUNC_LASSO_DEFAULTS)."""

    num_samples: int = pydantic.Field(default=10000, ge=1)


class TruncatedElasticNetRegressionConfig(TruncatedLassoRegressionConfig):
    """Configuration for truncated elastic net regression.

    The pre-pydantic code derived its defaults from TRUNC_LASSO_DEFAULTS, so
    this adds no fields of its own; weight_decay comes from OptimizerConfig.
    """

    # Selects the known- or unknown-variance estimator in the elastic net
    # wrapper.
    noise_var: Optional[ch.Tensor] = None


class TruncatedRidgeRegressionConfig(TruncatedRegressionConfig):
    """Configuration for truncated ridge regression.

    The pre-pydantic code derived its defaults from TRUNC_REG_DEFAULTS, so
    this adds no fields of its own; weight_decay comes from OptimizerConfig.
    """

    # Selects the known- or unknown-variance estimator in the ridge wrapper.
    noise_var: Optional[ch.Tensor] = None


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
