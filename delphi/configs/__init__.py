# Author: pstefanou12@
"""Pydantic configuration models for delphi.ai algorithms.

Split by concern so no single module has to hold every config: the trainer,
optimizer, scheduler and dataset groups are separate from the per-estimator
model configs. Every name is re-exported here, so ``from delphi import
configs`` followed by ``configs.SomeConfig`` keeps working regardless of which
submodule a config lives in.
"""

from delphi.configs.base import (
    DelphiConfig,
    LinearModelConfig,
    TrainerConfig,
    _ConfigT,
)
from delphi.configs.classification import (
    SoftmaxRegressionConfig,
    TruncatedLogisticRegressionConfig,
    TruncatedProbitRegressionConfig,
)
from delphi.configs.dataset import DatasetConfig
from delphi.configs.distributions import (
    TruncatedExponentialFamilyDistributionConfig,
    TruncatedMultivariateNormalConfig,
)
from delphi.configs.lqr import TruncatedLQRConfig
from delphi.configs.optimizers import (
    AdamConfig,
    LBFGSConfig,
    OptimizerConfig,
    SGDConfig,
)
from delphi.configs.regression import (
    TruncatedElasticNetRegressionConfig,
    TruncatedLassoRegressionConfig,
    TruncatedLinearDynamicalSystemConfig,
    TruncatedRegressionConfig,
    TruncatedRegressionFieldsConfig,
    TruncatedRidgeRegressionConfig,
)
from delphi.configs.schedulers import (
    CosineLRConfig,
    CyclicLRConfig,
    ExponentialLRConfig,
    LinearLRConfig,
    MultiStepLRConfig,
    PlateauLRConfig,
    StepLRConfig,
    WarmupConfig,
)

__all__ = [
    "AdamConfig",
    "CosineLRConfig",
    "CyclicLRConfig",
    "DatasetConfig",
    "DelphiConfig",
    "ExponentialLRConfig",
    "LBFGSConfig",
    "LinearLRConfig",
    "LinearModelConfig",
    "MultiStepLRConfig",
    "OptimizerConfig",
    "PlateauLRConfig",
    "SGDConfig",
    "SoftmaxRegressionConfig",
    "StepLRConfig",
    "TrainerConfig",
    "TruncatedElasticNetRegressionConfig",
    "TruncatedExponentialFamilyDistributionConfig",
    "TruncatedLQRConfig",
    "TruncatedLassoRegressionConfig",
    "TruncatedLinearDynamicalSystemConfig",
    "TruncatedLogisticRegressionConfig",
    "TruncatedMultivariateNormalConfig",
    "TruncatedProbitRegressionConfig",
    "TruncatedRegressionConfig",
    "TruncatedRegressionFieldsConfig",
    "TruncatedRidgeRegressionConfig",
    "WarmupConfig",
    "_ConfigT",
]
