# Author: pstefanou12@
"""Pydantic configuration models for delphi.ai algorithms.

The framework-level configs are flat; the model configs are nested by domain,
with truncation as a modifier within each domain. Every name is re-exported
here, so ``from delphi import configs`` followed by ``configs.SomeConfig``
keeps working regardless of which submodule a config lives in -- this module
is the single re-export point, and the subpackage ``__init__`` files are
deliberately empty.
"""

from delphi.configs.base import DelphiConfig, TrainerConfig, _ConfigT
from delphi.configs.dataset import DatasetConfig
from delphi.configs.distributions.truncated.exponential_family import (
    TruncatedExponentialFamilyDistributionConfig,
)
from delphi.configs.distributions.truncated.multivariate_normal import (
    TruncatedMultivariateNormalConfig,
)
from delphi.configs.optimizers import (
    AdamConfig,
    LBFGSConfig,
    OptimizerConfig,
    SGDConfig,
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
from delphi.configs.stats.classification import SoftmaxRegressionConfig
from delphi.configs.stats.linear import LinearModelConfig
from delphi.configs.stats.truncated.classification import (
    TruncatedClassificationConfig,
    TruncatedLogisticRegressionConfig,
    TruncatedProbitRegressionConfig,
)
from delphi.configs.stats.truncated.linear import TruncatedLinearModelConfig
from delphi.configs.stats.truncated.lqr import TruncatedLQRConfig
from delphi.configs.stats.truncated.regression import (
    TruncatedElasticNetRegressionConfig,
    TruncatedLassoRegressionConfig,
    TruncatedLinearDynamicalSystemConfig,
    TruncatedRegressionConfig,
    TruncatedRegressionFieldsConfig,
    TruncatedRidgeRegressionConfig,
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
    "TruncatedClassificationConfig",
    "TruncatedElasticNetRegressionConfig",
    "TruncatedExponentialFamilyDistributionConfig",
    "TruncatedLQRConfig",
    "TruncatedLassoRegressionConfig",
    "TruncatedLinearDynamicalSystemConfig",
    "TruncatedLinearModelConfig",
    "TruncatedLogisticRegressionConfig",
    "TruncatedMultivariateNormalConfig",
    "TruncatedProbitRegressionConfig",
    "TruncatedRegressionConfig",
    "TruncatedRegressionFieldsConfig",
    "TruncatedRidgeRegressionConfig",
    "WarmupConfig",
    "_ConfigT",
]
