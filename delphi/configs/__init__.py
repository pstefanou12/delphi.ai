# Author: pstefanou12@
"""Pydantic configuration models for delphi.ai algorithms.

The framework-level configs are flat; the model configs are nested by domain,
with truncation as a modifier within each domain. Every name is re-exported
here, so ``from delphi import configs`` followed by ``configs.SomeConfig``
keeps working regardless of which submodule a config lives in -- this module
is the single re-export point, and the subpackage ``__init__`` files are
deliberately empty.
"""

from delphi.configs.base import DelphiConfig, TrainerConfig, WarmStartConfig, _ConfigT
from delphi.configs.dataset import DatasetConfig
from delphi.configs.distributions.truncated.exponential_family import (
    TruncatedExponentialFamilyDistributionConfig,
    TruncatedGammaUnknownConcentrationConfig,
    TruncatedInverseGaussianConfig,
    TruncatedWishartConfig,
    WarmStartedExponentialFamilyDistributionConfig,
)
from delphi.configs.distributions.truncated.multivariate_normal import (
    TruncatedMultivariateNormalConfig,
)
from delphi.configs.distributions.truncated.unknown_truncation_multivariate_normal import (
    UnknownTruncationMultivariateNormalConfig,
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
    FisherLRConfig,
    InverseTimeLRConfig,
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
    TruncatedLassoRegressionConfig,
    TruncatedLinearDynamicalSystemConfig,
    TruncatedRegressionConfig,
    TruncatedRegressionFieldsConfig,
)

__all__ = [
    "AdamConfig",
    "CosineLRConfig",
    "CyclicLRConfig",
    "DatasetConfig",
    "DelphiConfig",
    "ExponentialLRConfig",
    "FisherLRConfig",
    "LBFGSConfig",
    "InverseTimeLRConfig",
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
    "TruncatedExponentialFamilyDistributionConfig",
    "TruncatedGammaUnknownConcentrationConfig",
    "TruncatedInverseGaussianConfig",
    "TruncatedWishartConfig",
    "WarmStartConfig",
    "WarmStartedExponentialFamilyDistributionConfig",
    "TruncatedLQRConfig",
    "TruncatedLassoRegressionConfig",
    "TruncatedLinearDynamicalSystemConfig",
    "TruncatedLinearModelConfig",
    "TruncatedLogisticRegressionConfig",
    "TruncatedMultivariateNormalConfig",
    "TruncatedProbitRegressionConfig",
    "TruncatedRegressionConfig",
    "TruncatedRegressionFieldsConfig",
    "UnknownTruncationMultivariateNormalConfig",
    "WarmupConfig",
    "_ConfigT",
]
